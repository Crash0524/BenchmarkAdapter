# Copyright 2026 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import base64
import openai
import numpy as np
from PIL import Image
from typing import Union, Optional
from google import genai
from functools import partial
from google.genai.types import HttpOptions, GenerateContentConfig
import time
import openai
from openai import OpenAI, ChatCompletion
from anthropic import AnthropicVertex


class GPT_Client:
    def __init__(self, model_name: str = "gpt-3.5-turbo") -> None:
        self.model_name = model_name

    def chat(self, messages, json_mode: bool = False, temperature: float = 0.0) -> tuple[str, ChatCompletion]:
        """
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "hi"},
        ])
        """
        self.client = OpenAI()
        openai.api_key = os.environ.get("OPENAI_API_KEY")  # Make sure to set your OpenAI API key in the environment variable
        chat_completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            response_format={"type": "json_object"} if json_mode else None,
            temperature=temperature,
        )
        response = chat_completion.choices[0].message.content
        return response, chat_completion

    def one_step_chat(
        self, text, system_msg: str = None, json_mode=False, temperature: float = 0.0
    ) -> tuple[str, ChatCompletion]:
        messages = []
        if system_msg is not None:
            messages.append({"role": "system", "content": system_msg})
        messages.append({"role": "user", "content": text})
        return self.chat(messages, json_mode=json_mode, temperature=temperature)


class CLAUDE_Client:
    def __init__(self, model_name: str = "claude-3-7-sonnet@20250219") -> None:
        self.model_name = model_name

    def chat(self, messages, sys_msg, temperature: float = 0.0):
        """
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "hi"},
        ])
        """
        self.client = AnthropicVertex(region=os.environ.get("GOOGLE_CLOUD_LOCATION"), project_id=os.environ.get("GOOGLE_CLOUD_PROJECT"))

        message = self.client.messages.create(
            model=self.model_name,
            system=sys_msg,
            messages=messages,
            temperature=temperature,
        )
        response = message.content[0].text
        return response, message

    def one_step_chat(
        self, text, system_msg: str = None, temperature: float = 0.0
    ):
        messages = []
        messages.append({"role": "user", "content": text})
        return self.chat(messages, system_msg, temperature=temperature)


class GPT4V_Client:
    def __init__(self, model_name: str = "gpt-4o", max_tokens: int = 512):
        self.model_name = model_name
        self.max_tokens = max_tokens

    def encode_image(self, path: str):
        with open(path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
                         
    def one_step_chat(
        self, text, image: Union[Image.Image, np.ndarray], 
        system_msg: Optional[str] = None,
    ):
        jpg_base64_str = self.encode_image(image)
        messages = []
        if system_msg is not None:
            messages.append({"role": "system", "content": system_msg})
        messages += [{
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{jpg_base64_str}"},},
                ],
        }]
        self.client = OpenAI()
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content, response


class GEMINI_Client:
    def __init__(self, model_name: str = "gemini-2.5-flash-lite") -> None:
        self.model_name = model_name

    def chat(self, messages, json_mode: bool = False, temperature: float = 0.0) -> tuple[str, ChatCompletion]:
        """
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "hi"},
        ])
        """
        self.client = genai.Client(http_options=HttpOptions(api_version="v1"))
        chat_completion = self.client.models.generate_content(
            model=self.model_name,
            contents=messages[1]['content'],
            config=GenerateContentConfig(
                temperature=temperature,
                system_instruction=messages[0]['content'],
            )
        )
        response = chat_completion.text
        return response, chat_completion

    def one_step_chat(
        self, text, system_msg: str = None, json_mode=False, temperature: float = 0.0
    ) -> tuple[str, ChatCompletion]:
        messages = []
        if system_msg is not None:
            messages.append({"role": "system", "content": system_msg})
        messages.append({"role": "user", "content": text})
        return self.chat(messages, json_mode=json_mode, temperature=temperature)
    

class QWEN_Client:
    def __init__(self, model_name: str = "qwen3.5-flash") -> None:
        self.model_name = model_name
        # 初始化阿里百炼客户端
        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

    def chat(self, messages, json_mode: bool = False, temperature: float = 0.0) -> tuple[str, any]:
        """
        与 GEMINI_Client 格式对齐，接收消息列表，返回 (响应文本, 原始对象)
        """
        # 调用 Qwen API 并开启深度思考
        chat_completion = self.client.chat.completions.create(
            model=self.model_name,
            messacges=messages,
            temperature=temperature,
            extra_body={"enable_thinking": True}
        )

        response_text = chat_completion.choices[0].message.content
        # 提取深度思考内容 (reasoning_content)
        reasoning = getattr(chat_completion.choices[0].message, "reasoning_content", "")

        # 为了让后续的评估逻辑能读到思考过程，我们将思考内容包装在 <think> 标签内
        if reasoning:
            response_text = f"<think>\n{reasoning}\n</think>\n{response_text}"

        return response_text, chat_completion

    def one_step_chat(
        self, text, system_msg: str = None, json_mode=False, temperature: float = 0.0
    ) -> tuple[str, any]:
        messages = []
        if system_msg is not None:
            messages.append({"role": "system", "content": system_msg})
        messages.append({"role": "user", "content": text})
        return self.chat(messages, json_mode=json_mode, temperature=temperature)
    

class LOCAL_Client:
    def __init__(self, model_name: str = "/mnt/data/workspace/chuntingmen/model/Qwen3.5-9B") -> None:
        """
        model_name 必须与 vllm 启动时的 --model 参数或 --served-model-name 一致
        """
        self.model_name = "/mnt/data/workspace/chuntingmen/model/Qwen3.5-9B"
        # 初始化本地 vLLM 客户端
        self.client = OpenAI(
            api_key="EMPTY", # vLLM 本地调用不需要 API Key
            base_url="http://localhost:8000/v1"
        )

    def chat(self, messages, json_mode: bool = False, temperature: float = 0.0) -> tuple[str, any]:
        """
        与 QWEN_Client 格式对齐，接收消息列表，返回 (响应文本, 原始对象)
        """
        # 注意：本地 vLLM 默认通常不需要 extra_body={"enable_thinking": True}
        # 因为你启动时带了 --reasoning-parser qwen3，它会自动解析思考过程
        chat_completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            # 如果是 json 模式，vLLM 支持直接传 response_format
            response_format={"type": "json_object"} if json_mode else None,
            timeout = 180
        )

        message_obj = chat_completion.choices[0].message
        response_text = message_obj.content
        
        # 提取本地 vLLM 解析出的深度思考内容 (reasoning_content)
        # 启动参数 --reasoning-parser qwen3 会将思考内容放入这个字段
        reasoning = getattr(message_obj, "reasoning_content", "")

        # 格式对齐：将思考内容包装在 <think> 标签内
        if reasoning:
            response_text = f"<think>\n{reasoning}\n</think>\n{response_text}"

        return response_text, chat_completion

    def one_step_chat(
        self, text, system_msg: str = None, json_mode=False, temperature: float = 0.0
    ) -> tuple[str, any]:
        messages = []
        if system_msg is not None:
            messages.append({"role": "system", "content": system_msg})
        messages.append({"role": "user", "content": text})
        return self.chat(messages, json_mode=json_mode, temperature=temperature)


CLIENT_DICT = {
    "gpt-3.5-turbo": GPT_Client,
    "gpt-4": GPT_Client,
    "gpt-4o": GPT4V_Client,
    "gemini-2.5-flash": GEMINI_Client,
    "gemini-2.5-pro": GEMINI_Client,
    "claude-3-7-sonnet@20250219": CLAUDE_Client,
    "qwen3.5-flash": QWEN_Client,
    "qwen3.5-plus": QWEN_Client,
    "qwen3.5-27b": QWEN_Client,
    "local": LOCAL_Client,
}