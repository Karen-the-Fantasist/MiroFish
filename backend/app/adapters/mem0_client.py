"""
Mem0 客户端封装
提供单例模式的 Memory 实例，支持延迟初始化
"""

import logging
import threading
from typing import Optional

import requests

from mem0 import Memory

from app.config import Config

logger = logging.getLogger(__name__)

# 单例实例和锁
_memory_instance: Optional[Memory] = None
_lock = threading.Lock()


def _check_embedding_service(base_url: str, timeout: int = 5) -> bool:
    """验证 Embedding 服务是否可用"""
    try:
        response = requests.get(f"{base_url.rstrip('/')}/models", timeout=timeout)
        return response.status_code == 200
    except Exception:
        return False


def get_memory_instance() -> Memory:
    """
    获取 mem0 Memory 单例实例

    使用延迟初始化（lazy initialization），首次调用时才创建实例。
    线程安全，支持多线程环境。

    Returns:
        Memory: mem0 Memory 实例

    Raises:
        RuntimeError: 如果 USE_MEM0 未启用或配置无效
    """
    global _memory_instance

    if _memory_instance is not None:
        return _memory_instance

    with _lock:
        # 双重检查锁定
        if _memory_instance is not None:
            return _memory_instance

        # 检查功能开关
        if not Config.USE_MEM0:
            raise RuntimeError("USE_MEM0 未启用，请检查环境变量配置")

        # 验证必要配置
        if not Config.LLM_API_KEY:
            raise RuntimeError("LLM_API_KEY 未配置")
        if not Config.NEO4J_PASSWORD:
            raise RuntimeError("NEO4J_PASSWORD 未配置")

        # 验证 Embedding 服务可用性
        if not _check_embedding_service(Config.EMBEDDING_BASE_URL):
            raise RuntimeError(
                f"Embedding 服务不可用: {Config.EMBEDDING_BASE_URL}\n"
                "请确保 vllm Embedding 服务已启动：\n"
                "  conda activate vllm\n"
                "  bash scripts/start_embedding_service.sh\n"
                "或设置 EMBEDDING_BASE_URL 指向可用的 Embedding 服务"
            )

        logger.info("初始化 mem0 Memory 实例...")

        # 构建配置
        config = {
            "graph_store": {
                "provider": "neo4j",
                "config": {
                    "url": Config.NEO4J_URL,
                    "username": Config.NEO4J_USERNAME,
                    "password": Config.NEO4J_PASSWORD,
                    "database": "neo4j",
                },
            },
            "llm": {
                "provider": "openai",
                "config": {
                    "model": Config.LLM_MODEL_NAME,
                    "openai_base_url": Config.LLM_BASE_URL,
                    "api_key": Config.LLM_API_KEY,
                },
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": Config.EMBEDDING_MODEL,
                    "openai_base_url": Config.EMBEDDING_BASE_URL,
                    "api_key": Config.EMBEDDING_API_KEY,
                },
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "path": "/tmp/qdrant",
                },
            },
        }

        _memory_instance = Memory.from_config(config)
        logger.info("mem0 Memory 实例初始化完成")

        return _memory_instance


def reset_memory_instance() -> None:
    """
    重置 mem0 Memory 单例实例

    用于测试或需要重新初始化的场景
    """
    global _memory_instance

    with _lock:
        _memory_instance = None
        logger.info("mem0 Memory 实例已重置")
