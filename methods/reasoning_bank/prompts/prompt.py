from __future__ import annotations

import difflib
import logging
from pathlib import Path
from typing import Any

from benchmark.WebArena import observation as benchmark_prompting


DEFAULT_SYSTEM_PROMPT = (
    "You are an agent trying to solve a web task based on the content of the page and "
    "a user instructions. You can interact with the page and explore. Each time you "
    "submit an action it will be sent to the browser and you will receive a new page."
)


def _read_text_file(path: str) -> str:
    data = Path(path).read_text(encoding="utf-8")
    return data.strip()


def build_system_prompt(flags: Any, dynamic_prompt: str | None = None) -> str:
    """Build method-level system prompt.

    `dynamic_prompt` is used by step-level memory mode.
    When absent, static memory text from `flags.memory_path` is appended when available.
    """

    custom_base = None
    prompt_path = getattr(flags, "prompt_path", None)
    if isinstance(prompt_path, str) and prompt_path.strip():
        try:
            custom_base = _read_text_file(prompt_path)
        except OSError:
            custom_base = None

    sys_msg = custom_base or DEFAULT_SYSTEM_PROMPT

    if dynamic_prompt:
        return sys_msg + "\n\n" + dynamic_prompt

    memory_path = getattr(flags, "memory_path", None)
    if isinstance(memory_path, (str, Path)) and str(memory_path).strip():
        memory_text = _read_text_file(str(memory_path))
        if memory_text:
            sys_msg += (
                "\n\nBelow are some memory items that I accumulated from past interaction from "
                "the environment that may be helpful to solve the task. You can use it when you "
                "feel it's relevant. In each step, please first explicitly discuss if you want to "
                "use each memory item or not, and then take action."
            )
            sys_msg += "\n\n" + memory_text
    return sys_msg


class GoalInstructions(benchmark_prompting.PromptElement):
    def __init__(self, goal: str, visible: bool = True) -> None:
        super().__init__(visible)
        self._prompt = f"""\
# Instructions
Review the current state of the page and all other information to find the best
possible next action to accomplish your goal. Your answer will be interpreted
and executed by a program, make sure to follow the formatting instructions.

## Goal:
{goal}
"""


class ChatInstructions(benchmark_prompting.PromptElement):
    def __init__(self, chat_messages: list[dict[str, str]], visible: bool = True) -> None:
        super().__init__(visible)
        self._prompt = """\
# Instructions

You are a UI Assistant, your goal is to help the user perform tasks using a web browser. You can
communicate with the user via a chat, in which the user gives you instructions and in which you
can send back messages. You have access to a web browser that both you and the user can see,
and with which only you can interact via specific commands.

Review the instructions from the user, the current state of the page and all other information
to find the best possible next action to accomplish your goal. Your answer will be interpreted
and executed by a program, make sure to follow the formatting instructions.

## Chat messages:

"""
        self._prompt += "\n".join(
            [f" - [{msg['role']}] {msg['message']}" for msg in chat_messages]
        )


class Memory(benchmark_prompting.PromptElement):
    _prompt = ""
    _abstract_ex = """
<memory>
Write down anything you need to remember for next steps. You will be presented
with the list of previous memories and past actions.
</memory>
"""
    _concrete_ex = """
<memory>
I clicked on bid 32 to activate tab 2. The accessibility tree should mention
focusable for elements of the form at next step.
</memory>
"""

    def _parse_answer(self, text_answer: str):
        return benchmark_prompting.parse_html_tags_raise(
            text_answer,
            optional_keys=["memory"],
            merge_multiple=True,
        )


class Think(benchmark_prompting.PromptElement):
    _prompt = ""
    _abstract_ex = """
<think>
Think step by step. If you need to make calculations such as coordinates, write them here. Describe the effect
that your previous action had on the current content of the page.
</think>
"""
    _concrete_ex = """
<think>
My memory says that I filled the first name and last name, but I can't see any
content in the form. I need to explore different ways to fill the form. Perhaps
the form is not visible yet or some fields are disabled. I need to replan.
</think>
"""

    def _parse_answer(self, text_answer: str):
        return benchmark_prompting.parse_html_tags_raise(
            text_answer,
            optional_keys=["think"],
            merge_multiple=True,
        )


def diff(previous: str | None, new: str | None):
    if previous == new:
        return "Identical", []
    if not previous:
        return "previous is empty", []

    diff_gen = difflib.ndiff(previous.splitlines(), (new or "").splitlines())
    diff_lines: list[str] = []
    plus_count = 0
    minus_count = 0
    for line in diff_gen:
        if line.strip().startswith("+"):
            diff_lines.append(line)
            plus_count += 1
        elif line.strip().startswith("-"):
            diff_lines.append(line)
            minus_count += 1

    header = f"{plus_count} lines added and {minus_count} lines removed:"
    return header, diff_lines


class Diff(benchmark_prompting.Shrinkable):
    def __init__(
        self,
        previous: str,
        new: str,
        prefix: str = "",
        max_line_diff: int = 20,
        shrink_speed: int = 2,
        visible: bool = True,
    ) -> None:
        super().__init__(visible=visible)
        self.max_line_diff = max_line_diff
        self.header, self.diff_lines = diff(previous, new)
        self.shrink_speed = shrink_speed
        self.prefix = prefix

    def shrink(self):
        self.max_line_diff -= self.shrink_speed
        self.max_line_diff = max(1, self.max_line_diff)

    @property
    def _prompt(self) -> str:
        diff_str = "\n".join(self.diff_lines[: self.max_line_diff])
        if len(self.diff_lines) > self.max_line_diff:
            original_count = len(self.diff_lines)
            diff_str = (
                f"{diff_str}\nDiff truncated, {original_count - self.max_line_diff} changes now shown."
            )
        return f"{self.prefix}{self.header}\n{diff_str}\n"


class HistoryStep(benchmark_prompting.Shrinkable):
    def __init__(
        self,
        previous_obs: dict[str, Any],
        current_obs: dict[str, Any],
        action: str,
        memory: str | None,
        thought: str | None,
        flags: Any,
        shrink_speed: int = 1,
    ) -> None:
        super().__init__()
        self.html_diff = Diff(
            previous_obs[flags.html_type],
            current_obs[flags.html_type],
            prefix="\n### HTML diff:\n",
            shrink_speed=shrink_speed,
            visible=lambda: flags.use_html and flags.use_diff,
        )
        self.ax_tree_diff = Diff(
            previous_obs["axtree_txt"],
            current_obs["axtree_txt"],
            prefix="\n### Accessibility tree diff:\n",
            shrink_speed=shrink_speed,
            visible=lambda: flags.use_ax_tree and flags.use_diff,
        )
        self.error = benchmark_prompting.Error(
            current_obs["last_action_error"],
            visible=(
                lambda: flags.use_error_logs
                and current_obs["last_action_error"]
                and flags.use_past_error_logs
            ),
            prefix="### ",
        )
        self.action = action
        self.memory = memory
        self.thought = thought
        self.flags = flags

    def shrink(self):
        super().shrink()
        self.html_diff.shrink()
        self.ax_tree_diff.shrink()

    @property
    def _prompt(self) -> str:
        prompt = ""
        prompt += f"\n### Thought:\n{self.thought}\n"
        if self.flags.use_action_history:
            prompt += f"\n### Action:\n{self.action}\n"
        prompt += f"{self.error.prompt}{self.html_diff.prompt}{self.ax_tree_diff.prompt}"
        if self.flags.use_memory and self.memory is not None:
            prompt += f"\n### Memory:\n{self.memory}\n"
        return prompt


class History(benchmark_prompting.Shrinkable):
    def __init__(
        self,
        history_obs: list[dict[str, Any]],
        actions: list[str],
        memories: list[str | None],
        thoughts: list[str | None],
        flags: Any,
        shrink_speed: int = 1,
    ) -> None:
        super().__init__(visible=lambda: flags.use_history)
        assert len(history_obs) == len(actions) + 1
        assert len(history_obs) == len(memories) + 1
        self.history_steps: list[HistoryStep] = []
        for i in range(1, len(history_obs)):
            self.history_steps.append(
                HistoryStep(
                    history_obs[i - 1],
                    history_obs[i],
                    actions[i - 1],
                    memories[i - 1],
                    thoughts[i - 1],
                    flags,
                    shrink_speed=shrink_speed,
                )
            )

    def shrink(self):
        super().shrink()
        for step in self.history_steps:
            step.shrink()

    @property
    def _prompt(self):
        prompts = ["# History of interaction with the task:\n"]
        for i, step in enumerate(self.history_steps):
            prompts.append(f"## step {i}")
            prompts.append(step.prompt)
        return "\n".join(prompts) + "\n"


class MainPrompt(benchmark_prompting.Shrinkable):
    """Method-owned prompt composition.

    Benchmark-owned blocks used here:
    - Observation
    - ActionSpace
    """

    def __init__(
        self,
        obs_history: list[dict[str, Any]],
        actions: list[str],
        memories: list[str | None],
        thoughts: list[str | None],
        flags: Any,
    ) -> None:
        super().__init__()
        self.flags = flags
        self.history = History(obs_history, actions, memories, thoughts, flags)
        if self.flags.enable_chat:
            self.instructions = ChatInstructions(obs_history[-1]["chat_messages"])
        else:
            if sum(msg["role"] == "user" for msg in obs_history[-1]["chat_messages"]) > 1:
                logging.warning(
                    "Agent is in goal mode, but multiple user messages are present in the chat. "
                    "Consider switching to enable_chat=True."
                )
            self.instructions = GoalInstructions(obs_history[-1]["goal"])

        self.obs = benchmark_prompting.Observation(obs_history[-1], flags)
        self.action_space = benchmark_prompting.ActionSpace(flags)
        self.think = Think(visible=lambda: flags.use_thinking)
        self.memory = Memory(visible=lambda: flags.use_memory)

    @property
    def _prompt(self) -> str:
        prompt = f"""\
{self.instructions.prompt}\
{self.obs.prompt}\
{self.history.prompt}\
{self.action_space.prompt}\
{self.think.prompt}\
{self.memory.prompt}\
"""

        if self.flags.use_abstract_example:
            prompt += f"""
# Abstract Example

Here is an abstract version of the answer with description of the content of
each tag. Make sure you follow this structure, but replace the content with your
answer:
{self.think.abstract_ex}\
{self.memory.abstract_ex}\
{self.action_space.abstract_ex}\
"""

        if self.flags.use_concrete_example:
            prompt += f"""
# Concrete Example

Here is a concrete example of how to format your answer.
Make sure to follow the template with proper tags:
{self.think.concrete_ex}\
{self.memory.concrete_ex}\
{self.action_space.concrete_ex}\
"""
        return self.obs.add_screenshot(prompt)

    def shrink(self):
        self.history.shrink()
        self.obs.shrink()

    def _parse_answer(self, text_answer: str):
        ans_dict = {}
        ans_dict.update(self.think._parse_answer(text_answer))
        ans_dict.update(self.memory._parse_answer(text_answer))
        ans_dict.update(self.action_space._parse_answer(text_answer))
        return ans_dict
