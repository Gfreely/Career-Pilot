import os
from openai import OpenAI
from typing import Optional, List, Dict, Any

class UnifiedLLMClient:
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.siliconflow.cn/v1"):
        """
        初始化统一调用客户端。
        优先使用传入的 api_key，如果为空则尝试从环境变量读取 SILICONFLOW_API_KEY。
        默认内部处理小模型定为响应速度较快且成本极低的 Qwen/Qwen2.5-7B-Instruct。
        """
        self.api_key = api_key or os.getenv("SILICONFLOW_API_KEY", "")
        self.base_url = base_url
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        
        # 定义内部高速流转的小模型标识
        self.small_model = "Qwen/Qwen2.5-7B-Instruct"

    def call_small_model(self, system_prompt: str, user_query: str = "") -> str:
        """
        专用于系统内部逻辑中转小模型的方法。
        涵盖场景：意图识别、复杂问题判定、查询重写与假想参考生成。
        若外部API调用失败，则返回空串执行静默兜底，防止核心崩溃。
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if user_query:
            messages.append({"role": "user", "content": user_query})
        
        try:
            response = self.client.chat.completions.create(
                messages=messages,
                model=self.small_model,
                stream=False,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"小模型调用异常 [{self.small_model}]: {e}")
            return ""

    def call_large_model(self, messages: List[Dict[str, str]], model_name: str, stream: bool = True):
        """
        暴露大模型最终输出和润色的核心逻辑接口。
        支持流式或非阻塞响应回调。
        """
        try:
            return self.client.chat.completions.create(
                messages=messages,
                model=model_name,
                stream=stream,
            )
        except Exception as e:
            print(f"大模型调用抛出异常 [{model_name}]: {e}")
            raise e
