from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(slots=True)
class MethodConfig:
    max_steps: int = 15
    include_history_turns: int = 3
    stop_on_answer: bool = True
    model_name: str = "qwen3.5-flash"
    api_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key: str | None = None
    temperature: float = 0.0
    max_tokens: int = 512
    enable_thinking: bool = True

    def resolved_api_key(self) -> str | None:
        if self.api_key:
            return self.api_key
        return os.environ.get("DASHSCOPE_API_KEY")

    def resolved_api_base(self) -> str:
        return os.environ.get("OPENAI_API_BASE", self.api_base_url)
