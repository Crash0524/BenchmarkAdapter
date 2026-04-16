
import json
import logging
from dataclasses import asdict, dataclass, field
import traceback
from typing import List, Dict
from warnings import warn
from langchain.schema import HumanMessage, SystemMessage

from browsergym.core.action.base import AbstractActionSet
from browsergym.utils.obs import flatten_axtree_to_str, flatten_dom_to_str, prune_html
from browsergym.experiments import Agent, AbstractAgentArgs

from . import observation
from .utils.llm_utils import ParseError, retry
from .utils.chat_api import ChatModelArgs 
from methods.reasoning_bank.prompts.prompt import MainPrompt, build_system_prompt


@dataclass
class WebArenaAgentArgs(AbstractAgentArgs):
    chat_model_args: ChatModelArgs = None
    flags: observation.Flags = field(default_factory=lambda: observation.Flags())
    max_retry: int = 4

    def make_agent(self):
        return WebArenaAgent(
            chat_model_args=self.chat_model_args, flags=self.flags, max_retry=self.max_retry
        )


class WebArenaAgent(Agent):
    def obs_preprocessor(self, obs: dict) -> dict:
        """
        Augment observations with text HTML and AXTree representations, which will be stored in
        the experiment traces.
        """

        obs = obs.copy()
        obs["dom_txt"] = flatten_dom_to_str(
            obs["dom_object"],
            with_visible=self.flags.extract_visible_tag,
            with_center_coords=self.flags.extract_coords == "center",
            with_bounding_box_coords=self.flags.extract_coords == "box",
            filter_visible_only=self.flags.extract_visible_elements_only,
        )
        obs["axtree_txt"] = flatten_axtree_to_str(
            obs["axtree_object"],
            with_visible=self.flags.extract_visible_tag,
            with_center_coords=self.flags.extract_coords == "center",
            with_bounding_box_coords=self.flags.extract_coords == "box",
            filter_visible_only=self.flags.extract_visible_elements_only,
        )
        obs["pruned_html"] = prune_html(obs["dom_txt"])

        return obs

    def __init__(
        self,
        chat_model_args: ChatModelArgs = None,
        flags: observation.Flags = None,
        max_retry: int = 4,
    ):
        self.chat_model_args = chat_model_args if chat_model_args is not None else ChatModelArgs()
        self.flags = flags if flags is not None else observation.Flags()
        self.max_retry = max_retry

        self.chat_llm = self.chat_model_args.make_chat_model()
        self.action_set = observation._get_action_space(self.flags)

        # consistency check
        if self.flags.use_screenshot:
            if not self.chat_model_args.has_vision():
                warn(
                    """\

Warning: use_screenshot is set to True, but the chat model \
does not support vision. Disabling use_screenshot."""
                )
                self.flags.use_screenshot = False

        # reset episode memory
        self.obs_history = []
        self.actions = []
        self.memories = []
        self.thoughts = []

        # Initialize new memory system only if step_level mode
        self.thoughts = []

    def get_action(self, obs):

        self.obs_history.append(obs)

        sys_msg = build_system_prompt(self.flags)

        # Method-owned prompt composition + benchmark-owned observation/action-space blocks.
        main_prompt = MainPrompt(
            obs_history=self.obs_history,
            actions=self.actions,
            memories=self.memories,
            thoughts=self.thoughts,
            flags=self.flags,
        )

        maxes = (
            self.flags.max_prompt_tokens,
            self.chat_model_args.max_total_tokens,
            self.chat_model_args.max_input_tokens,
        )
        maxes = [m for m in maxes if m is not None]
        max_prompt_tokens = min(maxes) if maxes else None
        
        tokenizer_id = self.chat_model_args.model_name
        if "qwen" in tokenizer_id.lower():
            tokenizer_id = "Qwen/Qwen2.5-7B-Instruct"
        elif "local" in tokenizer_id.lower():
            tokenizer_id = "/mnt/data/workspace/chuntingmen/model/Qwen3.5-9B"

        prompt = observation.fit_tokens(
            main_prompt,
            max_prompt_tokens=max_prompt_tokens,
            model_name=tokenizer_id,
        )

        llm_type = self.chat_llm._llm_type() if callable(getattr(self.chat_llm, "_llm_type", None)) else ""
        if llm_type == "claude":
            chat_messages = [
                ("system", sys_msg),
                ("human", prompt),
            ]
        else:
            chat_messages = [
                SystemMessage(content=sys_msg),
                HumanMessage(content=prompt),
            ]

        def parser(text):
            try:
                ans_dict = main_prompt._parse_answer(text)
                ans_dict["cleaned_answer"] = text
            except ParseError as e:
                # these parse errors will be caught by the retry function and
                # the chat_llm will have a chance to recover
                return None, False, str(e)
            print(ans_dict)
            return ans_dict, True, ""

        try:
            ans_dict = retry(self.chat_llm, chat_messages, n_retry=self.max_retry, parser=parser)
            # inferring the number of retries, TODO: make this less hacky
            ans_dict["n_retry"] = (len(chat_messages) - 3) / 2
        except ValueError as e:
            # Likely due to maximum retry. We catch it here to be able to return
            # the list of messages for further analysis
            ans_dict = {"action": None}
            
            ans_dict["err_msg"] = str(e)
            ans_dict["stack_trace"] = traceback.format_exc()
            ans_dict["n_retry"] = self.max_retry

        self.actions.append(ans_dict["action"])
        self.memories.append(ans_dict.get("memory", None))
        self.thoughts.append(ans_dict.get("think", None))

        serialized_messages = [m[1] if isinstance(m, tuple) else m.content for m in chat_messages]
        ans_dict["chat_messages"] = serialized_messages
        ans_dict["chat_model_args"] = asdict(self.chat_model_args)
        

        return ans_dict["action"], ans_dict

    def _detect_page_change(self, obs) -> bool:
        """Simple page change detection (Stage 1: basic heuristic)"""
        if len(self.obs_history) < 2:
            return False
        prev_url = self.obs_history[-2].get("url", "")
        curr_url = obs.get("url", "")
        return prev_url != curr_url

    def _detect_failure(self, obs) -> bool:
        """Detect if last action failed (placeholder)"""
        # Implement based on obs error signals
        return "error" in obs.get("axtree_txt", "").lower()

    def finalize_task(self, trajectory: List[Dict], task_id: str, bucket: str):
        """Task end: Write new memory"""
        if hasattr(self, "memory_writer"):
            self.memory_writer.write_from_trajectory(trajectory, task_id, bucket)
