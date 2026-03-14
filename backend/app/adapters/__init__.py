"""
Mem0 适配器模块
封装 mem0 Memory 客户端，提供统一接口
"""

from .mem0_client import get_memory_instance, reset_memory_instance
from .zep_graph_adapter import ZepGraphAdapter, create_zep_compatible_client

__all__ = [
    "get_memory_instance",
    "reset_memory_instance",
    "ZepGraphAdapter",
    "create_zep_compatible_client",
]
