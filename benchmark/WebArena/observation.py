# TODO

import abc
import logging
import platform

from copy import deepcopy
from dataclasses import asdict, dataclass
from textwrap import dedent
from typing import Literal
from warnings import warn

from browsergym.core.action.base import AbstractActionSet
from browsergym.core.action.highlevel import HighLevelActionSet
from browsergym.core.action.python import PythonActionSet

from .utils.llm_utils import ParseError
from .utils.llm_utils import (
    count_tokens,
    image_to_jpg_base64_url,
    parse_html_tags_raise,
)


@dataclass
class Flags:
    use_html: bool = True
    use_ax_tree: bool = False
    drop_ax_tree_first: bool = True  # This flag is no longer active TODO delete
    use_thinking: bool = False
    use_error_logs: bool = False
    use_past_error_logs: bool = False
    use_history: bool = False
    use_action_history: bool = False
    use_memory: bool = False
    use_diff: bool = False
    html_type: str = "pruned_html"
    use_concrete_example: bool = True
    use_abstract_example: bool = False
    multi_actions: bool = False
    action_space: Literal[
        "python", "bid", "coord", "bid+coord", "bid+nav", "coord+nav", "bid+coord+nav"
    ] = "bid"
    is_strict: bool = False
    # This flag will be automatically disabled `if not chat_model_args.has_vision()`
    use_screenshot: bool = True
    enable_chat: bool = False
    max_prompt_tokens: int = None
    extract_visible_tag: bool = False
    extract_coords: Literal["False", "center", "box"] = "False"
    extract_visible_elements_only: bool = False
    demo_mode: Literal["off", "default", "only_visible_elements"] = "off"
    memory_path: str = None
    observation_path: str = None
    prompt_path: str = None

    def copy(self):
        return deepcopy(self)

    def asdict(self):
        """Helper for JSON serializble requirement."""
        return asdict(self)

    @classmethod
    def from_dict(self, flags_dict):
        """Helper for JSON serializble requirement."""
        if isinstance(flags_dict, Flags):
            return flags_dict

        if not isinstance(flags_dict, dict):
            raise ValueError(f"Unregcognized type for flags_dict of type {type(flags_dict)}.")
        return Flags(**flags_dict)


class PromptElement:
    """Base class for all prompt elements. Prompt elements can be hidden.

    Prompt elements are used to build the prompt. Use flags to control which
    prompt elements are visible. We use class attributes as a convenient way
    to implement static prompts, but feel free to override them with instance
    attributes or @property decorator."""

    _prompt = ""
    _abstract_ex = ""
    _concrete_ex = ""

    def __init__(self, visible: bool = True) -> None:
        """Prompt element that can be hidden.

        Parameters
        ----------
        visible : bool, optional
            Whether the prompt element should be visible, by default True. Can
            be a callable that returns a bool. This is useful when a specific
            flag changes during a shrink iteration.
        """
        self._visible = visible

    @property
    def prompt(self):
        """Avoid overriding this method. Override _prompt instead."""
        return self._hide(self._prompt)

    @property
    def abstract_ex(self):
        """Useful when this prompt element is requesting an answer from the llm.
        Provide an abstract example of the answer here. See Memory for an
        example.

        Avoid overriding this method. Override _abstract_ex instead
        """
        return self._hide(self._abstract_ex)

    @property
    def concrete_ex(self):
        """Useful when this prompt element is requesting an answer from the llm.
        Provide a concrete example of the answer here. See Memory for an
        example.

        Avoid overriding this method. Override _concrete_ex instead
        """
        return self._hide(self._concrete_ex)

    @property
    def is_visible(self):
        """Handle the case where visible is a callable."""
        visible = self._visible
        if callable(visible):
            visible = visible()
        return visible

    def _hide(self, value):
        """Return value if visible is True, else return empty string."""
        if self.is_visible:
            return value
        else:
            return ""

    def _parse_answer(self, text_answer) -> dict:
        if self.is_visible:
            return self._parse_answer(text_answer)
        else:
            return {}


class Shrinkable(PromptElement, abc.ABC):
    @abc.abstractmethod
    def shrink(self) -> None:
        """Implement shrinking of this prompt element.

        You need to recursively call all shrinkable elements that are part of
        this prompt. You can also implement a shriking startegy for this prompt.
        Shrinking is can be called multiple times to progressively shrink the
        prompt until it fits max_tokens. Default max shrink iterations is 20.
        """
        pass


class Trunkater(Shrinkable):
    def __init__(self, visible, shrink_speed=0.3, start_trunkate_iteration=10):
        super().__init__(visible=visible)
        self.shrink_speed = shrink_speed
        self.start_trunkate_iteration = start_trunkate_iteration
        self.shrink_calls = 0
        self.deleted_lines = 0

    def shrink(self) -> None:
        if self.is_visible and self.shrink_calls >= self.start_trunkate_iteration:
            # remove the fraction of _prompt
            lines = self._prompt.splitlines()
            new_line_count = int(len(lines) * (1 - self.shrink_speed))
            self.deleted_lines += len(lines) - new_line_count
            self._prompt = "\n".join(lines[:new_line_count])
            self._prompt += f"\n... Deleted {self.deleted_lines} lines to reduce prompt size."

        self.shrink_calls += 1


def fit_tokens(
    shrinkable: Shrinkable, max_prompt_tokens=None, max_iterations=20, model_name="openai/gpt-4"
):
    """Shrink a prompt element until it fits max_tokens.

    Parameters
    ----------
    shrinkable : Shrinkable
        The prompt element to shrink.
    max_tokens : int
        The maximum number of tokens allowed.
    max_iterations : int, optional
        The maximum number of shrink iterations, by default 20.
    model_name : str, optional
        The name of the model used when tokenizing.

    Returns
    -------
    str : the prompt after shrinking.
    """

    if max_prompt_tokens is None:
        return shrinkable.prompt

    for _ in range(max_iterations):
        prompt = shrinkable.prompt
        if isinstance(prompt, str):
            prompt_str = prompt
        elif isinstance(prompt, list):
            prompt_str = "\n".join([p["text"] for p in prompt if p["type"] == "text"])
        else:
            raise ValueError(f"Unrecognized type for prompt: {type(prompt)}")
        if model_name.startswith("claude"):
            return prompt
        n_token = count_tokens(prompt_str, model=model_name)
        if n_token <= max_prompt_tokens:
            return prompt
        shrinkable.shrink()

    logging.info(
        dedent(
            f"""\
            After {max_iterations} shrink iterations, the prompt is still
            {count_tokens(prompt_str)} tokens (greater than {max_prompt_tokens}). Returning the prompt as is."""
        )
    )
    return prompt


class HTML(Trunkater):
    def __init__(self, html, visible: bool = True, prefix="") -> None:
        super().__init__(visible=visible, start_trunkate_iteration=5)
        self._prompt = f"\n{prefix}HTML:\n{html}\n"


class AXTree(Trunkater):
    def __init__(self, ax_tree, visible: bool = True, coord_type=None, prefix="") -> None:
        super().__init__(visible=visible, start_trunkate_iteration=10)
        if coord_type == "center":
            coord_note = """\
Note: center coordinates are provided in parenthesis and are
  relative to the top left corner of the page.\n\n"""
        elif coord_type == "box":
            coord_note = """\
Note: bounding box of each object are provided in parenthesis and are
  relative to the top left corner of the page.\n\n"""
        else:
            coord_note = ""
        self._prompt = f"\n{prefix}AXTree:\n{coord_note}{ax_tree}\n"


class Error(PromptElement):
    def __init__(self, error, visible: bool = True, prefix="") -> None:
        super().__init__(visible=visible)
        self._prompt = f"\n{prefix}Error from previous action:\n{error}\n"


class Observation(Shrinkable):
    """Observation of the current step.

    Contains the html, the accessibility tree and the error logs.
    """

    def __init__(self, obs, flags: Flags) -> None:
        super().__init__()
        self.flags = flags
        self.obs = obs
        self.html = HTML(obs[flags.html_type], visible=lambda: flags.use_html, prefix="## ")
        self.ax_tree = AXTree(
            obs["axtree_txt"],
            visible=lambda: flags.use_ax_tree,
            coord_type=flags.extract_coords,
            prefix="## ",
        )
        self.error = Error(
            obs["last_action_error"],
            visible=lambda: flags.use_error_logs and obs["last_action_error"],
            prefix="## ",
        )

    def shrink(self):
        self.ax_tree.shrink()
        self.html.shrink()

    @property
    def _prompt(self) -> str:
        return f"\n# Observation of current step:\n{self.html.prompt}{self.ax_tree.prompt}{self.error.prompt}\n\n"

    def add_screenshot(self, prompt):
        if self.flags.use_screenshot:
            if isinstance(prompt, str):
                prompt = [{"type": "text", "text": prompt}]
            img_url = image_to_jpg_base64_url(self.obs["screenshot"])
            prompt.append({"type": "image_url", "image_url": {"url": img_url}})

        return prompt


class MacNote(PromptElement):
    def __init__(self) -> None:
        super().__init__(visible=platform.system() == "Darwin")
        self._prompt = (
            "\nNote: you are on mac so you should use Meta instead of Control for Control+C etc.\n"
        )


class ActionSpace(PromptElement):
    def __init__(self, flags: Flags) -> None:
        super().__init__()
        self.flags = flags
        self.action_space = _get_action_space(flags)

        self._prompt = f"# Action space:\n{self.action_space.describe()}{MacNote().prompt}\n"
        self._abstract_ex = f"""
<action>
{self.action_space.example_action(abstract=True)}
</action>
"""
        self._concrete_ex = f"""
<action>
{self.action_space.example_action(abstract=False)}
</action>
"""

    def _parse_answer(self, text_answer):
        ans_dict = parse_html_tags_raise(text_answer, keys=["action"], merge_multiple=True)

        try:
            # just check if action can be mapped to python code but keep action as is
            # the environment will be responsible for mapping it to python
            self.action_space.to_python_code(ans_dict["action"])
        except Exception as e:
            raise ParseError(
                f"Error while parsing action\n: {e}\n"
                "Make sure your answer is restricted to the allowed actions."
            )

        return ans_dict


def _get_action_space(flags: Flags) -> AbstractActionSet:
    match flags.action_space:
        case "python":
            action_space = PythonActionSet(strict=flags.is_strict)
            if flags.multi_actions:
                warn(
                    f"Flag action_space={repr(flags.action_space)} incompatible with multi_actions={repr(flags.multi_actions)}."
                )
            if flags.demo_mode != "off":
                warn(
                    f"Flag action_space={repr(flags.action_space)} incompatible with demo_mode={repr(flags.demo_mode)}."
                )
            return action_space
        case "bid":
            action_subsets = ["chat", "bid"]
        case "coord":
            action_subsets = ["chat", "coord"]
        case "bid+coord":
            action_subsets = ["chat", "bid", "coord"]
        case "bid+nav":
            action_subsets = ["chat", "bid", "nav"]
        case "coord+nav":
            action_subsets = ["chat", "coord", "nav"]
        case "bid+coord+nav":
            action_subsets = ["chat", "bid", "coord", "nav"]
        case _:
            raise NotImplementedError(f"Unknown action_space {repr(flags.action_space)}")

    action_space = HighLevelActionSet(
        subsets=action_subsets,
        multiaction=flags.multi_actions,
        strict=flags.is_strict,
        demo_mode=flags.demo_mode,
    )

    return action_space
