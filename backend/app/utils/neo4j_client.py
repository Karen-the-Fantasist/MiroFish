"""Neo4j 客户端封装。

封装 Neo4j driver 连接管理，提供：
- 单例模式的连接池管理
- 上下文管理器支持
- 参数化查询防止注入
- UUID cursor 分页
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from neo4j import GraphDatabase, Driver, Session
from neo4j.exceptions import ServiceUnavailable, AuthError, TransientError

from app.config import Config
from .logger import get_logger

logger = get_logger("mirofish.neo4j_client")

_DEFAULT_PAGE_SIZE = 100
_MAX_NODES = 2000
_MAX_EDGES = 5000
_MAX_RETRIES = 3
_RETRY_DELAY = 2.0


@dataclass
class EntityNode:
    """实体节点数据结构。"""

    uuid: str
    name: str
    labels: list[str]
    summary: str
    attributes: dict[str, Any]
    related_edges: list[dict[str, Any]] = field(default_factory=list)
    related_nodes: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class EdgeInfo:
    """边信息数据结构。"""

    uuid: str
    name: str
    fact: str
    source_node_uuid: str
    target_node_uuid: str
    source_node_name: str | None = None
    target_node_name: str | None = None
    created_at: str | None = None
    valid_at: str | None = None
    invalid_at: str | None = None
    expired_at: str | None = None


class Neo4jClient:
    """Neo4j 客户端，单例模式管理连接池。"""

    _instance: Neo4jClient | None = None
    _driver: Driver | None = None

    def __new__(cls) -> Neo4jClient:
        """单例模式：确保全局只有一个客户端实例。"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """初始化客户端（仅首次创建时执行）。"""
        if self._driver is not None:
            return

        self._url = Config.NEO4J_URL
        self._username = Config.NEO4J_USERNAME
        self._password = Config.NEO4J_PASSWORD

        if not self._password:
            logger.warning("NEO4J_PASSWORD 未配置，连接可能失败")

        self._connect()

    def _connect(self) -> None:
        """建立连接池。"""
        try:
            self._driver = GraphDatabase.driver(
                self._url,
                auth=(self._username, self._password),
                max_connection_pool_size=50,
                connection_timeout=30.0,
            )
            logger.info(f"Neo4j 连接池已创建: {self._url}")
        except Exception as e:
            logger.error(f"Neo4j 连接池创建失败: {e}")
            raise

    def __enter__(self) -> Neo4jClient:
        """上下文管理器入口。"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """上下文管理器退出（不关闭连接池，保持复用）。"""
        pass

    def _get_session(self) -> Session:
        """获取会话。"""
        if self._driver is None:
            raise RuntimeError("Neo4j driver 未初始化")
        return self._driver.session()

    def _execute_with_retry(
        self,
        query: str,
        parameters: dict[str, Any],
        max_retries: int = _MAX_RETRIES,
        retry_delay: float = _RETRY_DELAY,
    ) -> list[dict[str, Any]]:
        """执行查询，失败时指数退避重试。"""
        last_exception: Exception | None = None
        delay = retry_delay

        for attempt in range(max_retries):
            try:
                with self._get_session() as session:
                    result = session.run(query, parameters)
                    return [dict(record) for record in result]
            except (ServiceUnavailable, TransientError, OSError) as e:
                last_exception = e
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Neo4j 查询重试 {attempt + 1}/{max_retries}: {str(e)[:100]}, "
                        f"{delay:.1f}s 后重试..."
                    )
                    time.sleep(delay)
                    delay *= 2
                else:
                    logger.error(f"Neo4j 查询失败，已重试 {max_retries} 次: {e}")
            except AuthError as e:
                # 认证错误不重试
                logger.error(f"Neo4j 认证失败: {e}")
                raise

        assert last_exception is not None
        raise last_exception

    def test_connection(self) -> bool:
        """测试连接是否正常。

        Returns:
            True 表示连接正常，False 表示连接失败
        """
        try:
            with self._get_session() as session:
                result = session.run("RETURN 1 as test")
                record = result.single()
                return record is not None and record["test"] == 1
        except Exception as e:
            logger.error(f"Neo4j 连接测试失败: {e}")
            return False

    def fetch_all_nodes(
        self,
        graph_id: str,
        limit: int = _MAX_NODES,
        cursor: str | None = None,
        page_size: int = _DEFAULT_PAGE_SIZE,
    ) -> list[EntityNode]:
        """分页获取图谱节点。

        Args:
            graph_id: 图谱 ID
            limit: 最大返回节点数
            cursor: UUID 分页游标（可选）
            page_size: 每页大小

        Returns:
            EntityNode 列表
        """
        all_nodes: list[EntityNode] = []
        current_cursor = cursor
        page_num = 0

        while len(all_nodes) < limit:
            # 构建 Cypher 查询（参数化防止注入）
            query = """
                MATCH (n:Entity {graph_id: $graph_id})
                WHERE $cursor IS NULL OR n.uuid > $cursor
                RETURN n
                ORDER BY n.uuid
                LIMIT $page_size
            """

            parameters = {
                "graph_id": graph_id,
                "cursor": current_cursor,
                "page_size": min(page_size, limit - len(all_nodes)),
            }

            page_num += 1
            records = self._execute_with_retry(query, parameters)

            if not records:
                break

            for record in records:
                node_data = record.get("n", {})
                if not node_data:
                    continue

                if hasattr(node_data, "_properties"):
                    props = dict(node_data._properties)
                else:
                    props = dict(node_data)

                node = EntityNode(
                    uuid=props.get("uuid", ""),
                    name=props.get("name", ""),
                    labels=list(getattr(node_data, "_labels", []))
                    if hasattr(node_data, "_labels")
                    else [],
                    summary=props.get("summary", ""),
                    attributes={
                        k: v
                        for k, v in props.items()
                        if k not in ("uuid", "name", "summary", "graph_id")
                    },
                )
                all_nodes.append(node)

            if records:
                last_record = records[-1].get("n", {})
                if hasattr(last_record, "_properties"):
                    current_cursor = last_record._properties.get("uuid")
                else:
                    current_cursor = last_record.get("uuid")

            if len(records) < page_size:
                break

        logger.info(f"获取图谱 {graph_id} 的节点: {len(all_nodes)} 个")
        return all_nodes[:limit]

    def fetch_all_edges(
        self,
        graph_id: str,
        limit: int = _MAX_EDGES,
        cursor: str | None = None,
        page_size: int = _DEFAULT_PAGE_SIZE,
    ) -> list[EdgeInfo]:
        """分页获取图谱所有边。

        Args:
            graph_id: 图谱 ID
            limit: 最大返回边数
            cursor: UUID 分页游标（可选）
            page_size: 每页大小

        Returns:
            EdgeInfo 列表
        """
        all_edges: list[EdgeInfo] = []
        current_cursor = cursor
        page_num = 0

        while len(all_edges) < limit:
            # 构建 Cypher 查询（参数化防止注入）
            query = """
                MATCH (source)-[r]->(target)
                WHERE r.graph_id = $graph_id
                AND ($cursor IS NULL OR r.uuid > $cursor)
                RETURN r, source.uuid as source_uuid, source.name as source_name,
                       target.uuid as target_uuid, target.name as target_name
                ORDER BY r.uuid
                LIMIT $page_size
            """

            parameters = {
                "graph_id": graph_id,
                "cursor": current_cursor,
                "page_size": min(page_size, limit - len(all_edges)),
            }

            page_num += 1
            records = self._execute_with_retry(query, parameters)

            if not records:
                break

            for record in records:
                edge_data = record.get("r", {})
                if not edge_data:
                    continue

                if hasattr(edge_data, "_properties"):
                    props = dict(edge_data._properties)
                else:
                    props = dict(edge_data)

                edge = EdgeInfo(
                    uuid=props.get("uuid", ""),
                    name=props.get("name", ""),
                    fact=props.get("fact", ""),
                    source_node_uuid=record.get("source_uuid", ""),
                    target_node_uuid=record.get("target_uuid", ""),
                    source_node_name=record.get("source_name"),
                    target_node_name=record.get("target_name"),
                    created_at=props.get("created_at"),
                    valid_at=props.get("valid_at"),
                    invalid_at=props.get("invalid_at"),
                    expired_at=props.get("expired_at"),
                )
                all_edges.append(edge)

            if records:
                last_record = records[-1].get("r", {})
                if hasattr(last_record, "_properties"):
                    current_cursor = last_record._properties.get("uuid")
                else:
                    current_cursor = last_record.get("uuid")

            if len(records) < page_size:
                break

        logger.info(f"获取图谱 {graph_id} 的边: {len(all_edges)} 条")
        return all_edges[:limit]

    def get_entity_edges(self, node_uuid: str) -> list[dict[str, Any]]:
        """获取指定节点的所有边及关联节点。

        Args:
            node_uuid: 节点 UUID

        Returns:
            边信息列表，每项包含 edge 和 related_node
        """
        query = """
            MATCH (n {uuid: $node_uuid})-[r]-(m)
            RETURN r as edge, m.uuid as related_uuid, m.name as related_name,
                   type(r) as edge_type, startNode(r).uuid as source_uuid,
                   endNode(r).uuid as target_uuid
        """

        parameters = {"node_uuid": node_uuid}
        records = self._execute_with_retry(query, parameters)

        edges: list[dict[str, Any]] = []
        for record in records:
            edge_data = record.get("edge", {})

            if hasattr(edge_data, "_properties"):
                props = dict(edge_data._properties)
            else:
                props = dict(edge_data) if edge_data else {}

            edge_info = {
                "uuid": props.get("uuid", ""),
                "name": props.get("name", ""),
                "fact": props.get("fact", ""),
                "edge_type": record.get("edge_type", ""),
                "source_uuid": record.get("source_uuid", ""),
                "target_uuid": record.get("target_uuid", ""),
                "related_uuid": record.get("related_uuid", ""),
                "related_name": record.get("related_name", ""),
                "created_at": props.get("created_at"),
                "valid_at": props.get("valid_at"),
                "invalid_at": props.get("invalid_at"),
                "expired_at": props.get("expired_at"),
            }
            edges.append(edge_info)

        logger.debug(f"获取节点 {node_uuid} 的边: {len(edges)} 条")
        return edges

    def close(self) -> None:
        """关闭连接池。"""
        if self._driver is not None:
            self._driver.close()
            self._driver = None
            logger.info("Neo4j 连接池已关闭")

    @classmethod
    def reset_instance(cls) -> None:
        """重置单例实例（用于测试或重新配置）。"""
        if cls._instance is not None:
            cls._instance.close()
            cls._instance = None


def get_neo4j_client() -> Neo4jClient:
    """获取 Neo4j 客户端单例实例。"""
    return Neo4jClient()
