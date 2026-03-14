"""
LLM客户端封装
统一使用OpenAI格式调用
"""

import json
import re
import time
from typing import Optional, Dict, Any, List
from openai import OpenAI, APIError, APIConnectionError, APITimeoutError, RateLimitError

from ..config import Config
from .logger import get_logger

logger = get_logger("mirofish.llm")

# LLM 调用超时设置（秒）
LLM_TIMEOUT = 500
LLM_CONNECT_TIMEOUT = 30


class LLMClient:
    """LLM客户端"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model = model or Config.LLM_MODEL_NAME

        if not self.api_key:
            logger.error("LLM_API_KEY 未配置")
            raise ValueError("LLM_API_KEY 未配置")

        logger.info(f"LLM客户端初始化: base_url={self.base_url}, model={self.model}")

        self.client = OpenAI(
            api_key=self.api_key, base_url=self.base_url, timeout=LLM_TIMEOUT
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 40960,
        response_format: Optional[Dict] = None,
    ) -> str:
        """
        发送聊天请求

        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数
            response_format: 响应格式（如JSON模式）

        Returns:
            模型响应文本
        """
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if response_format:
            kwargs["response_format"] = response_format

        logger.info(
            f"LLM调用开始: model={self.model}, temperature={temperature}, max_tokens={max_tokens}"
        )
        start_time = time.time()

        try:
            response = self.client.chat.completions.create(**kwargs)
            elapsed_time = time.time() - start_time
            logger.info(f"LLM调用成功: 耗时={elapsed_time:.2f}秒, model={self.model}")

            content = response.choices[0].message.content
            content = re.sub(r"潜意识思考[\s\S]*?$", "", content).strip()
            logger.debug(f"LLM响应长度: {len(content)} 字符")
            return content

        except APITimeoutError as e:
            elapsed_time = time.time() - start_time
            logger.error(f"LLM调用超时: 耗时={elapsed_time:.2f}秒, 错误={str(e)}")
            raise
        except APIConnectionError as e:
            elapsed_time = time.time() - start_time
            logger.error(
                f"LLM连接失败: 耗时={elapsed_time:.2f}秒, base_url={self.base_url}, 错误={str(e)}"
            )
            raise
        except RateLimitError as e:
            elapsed_time = time.time() - start_time
            logger.error(f"LLM速率限制: 耗时={elapsed_time:.2f}秒, 错误={str(e)}")
            raise
        except APIError as e:
            elapsed_time = time.time() - start_time
            logger.error(
                f"LLM API错误: 耗时={elapsed_time:.2f}秒, status_code={getattr(e, 'status_code', 'unknown')}, 错误={str(e)}"
            )
            raise
        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(
                f"LLM调用异常: 耗时={elapsed_time:.2f}秒, 类型={type(e).__name__}, 错误={str(e)}"
            )
            raise

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 40960,
    ) -> Dict[str, Any]:
        """
        发送聊天请求并返回JSON

        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数

        Returns:
            解析后的JSON对象
        """
        logger.info("chat_json: 开始JSON格式调用")

        try:
            response = self.chat(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            logger.error(f"chat_json: LLM调用失败, 错误={str(e)}")
            raise

        # 清理markdown代码块标记
        cleaned_response = response.strip()
        cleaned_response = re.sub(
            r"^```(?:json)?\s*\n?", "", cleaned_response, flags=re.IGNORECASE
        )
        cleaned_response = re.sub(r"\n?```\s*$", "", cleaned_response)
        cleaned_response = cleaned_response.strip()

        logger.debug(f"chat_json: 清理后响应长度={len(cleaned_response)} 字符")

        try:
            result = json.loads(cleaned_response)
            logger.info(f"chat_json: JSON解析成功, keys={list(result.keys())}")
            return result
        except json.JSONDecodeError as e:
            logger.error(
                f"chat_json: JSON解析失败, 错误={str(e)}, 全部响应={cleaned_response}"
            )
            raise ValueError(f"LLM返回的JSON格式无效: {cleaned_response[:500]}")
