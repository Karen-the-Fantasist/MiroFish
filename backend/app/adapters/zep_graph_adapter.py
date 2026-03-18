"""
Zep Graph 适配器
将 Zep Cloud SDK 调用转换为 mem0 + Neo4j 操作，保持 API 兼容性。
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from ..utils.logger import get_logger
from ..utils.neo4j_client import Neo4jClient, EntityNode, EdgeInfo, get_neo4j_client
from .mem0_client import get_memory_instance

logger = get_logger("mirofish.zep_graph_adapter")


@dataclass
class EpisodeData:
    """Episode 数据结构，与 Zep SDK 兼容。"""

    data: str
    type: str = "text"
    reference_id: Optional[str] = None


@dataclass
class SearchResult:
    """搜索结果结构。"""

    facts: list[dict[str, Any]] = field(default_factory=list)
    edges: list[EdgeInfo] = field(default_factory=list)
    nodes: list[EntityNode] = field(default_factory=list)


@dataclass
class EpisodeInfo:
    """Episode 信息结构。"""

    uuid_: str
    data: str
    processed: bool = True
    created_at: Optional[str] = None


class ZepGraphAdapter:
    """
    Zep Graph 适配器。

    将 Zep Cloud SDK 调用转换为 mem0 + Neo4j 操作。
    保持与 Zep SDK 的 API 兼容性，便于现有代码无缝迁移。

    使用方式:
        client = ZepGraphAdapter()
        graph_id = client.graph.create(name="My Graph")
        client.graph.add(graph_id=graph_id, type="text", data="Alice works at Google")
        results = client.graph.search(graph_id=graph_id, query="Where does Alice work?")
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        初始化适配器。

        Args:
            api_key: 保留参数，保持与 Zep SDK 兼容，实际不使用。
        """
        self._api_key = api_key  # 不使用，仅为 API 兼容
        self._memory = None
        self._neo4j = None
        self.graph = self._Graph(self)
        self.node = self._Node(self)
        self.edge = self._Edge(self)
        self.episode = self._Episode(self)

    def _get_memory(self):
        """延迟初始化 mem0 客户端。"""
        if self._memory is None:
            self._memory = get_memory_instance()
        return self._memory

    def _get_neo4j(self) -> Neo4jClient:
        """延迟初始化 Neo4j 客户端。"""
        if self._neo4j is None:
            self._neo4j = get_neo4j_client()
        return self._neo4j

    class _Graph:
        """Graph 子接口。"""

        def __init__(self, adapter: ZepGraphAdapter):
            self._adapter = adapter

        def create(
            self,
            graph_id: Optional[str] = None,
            name: str = "",
            description: str = "",
        ) -> str:
            """
            创建图谱。

            Args:
                graph_id: 图谱 ID（可选，自动生成）
                name: 图谱名称
                description: 图谱描述

            Returns:
                创建的图谱 ID
            """
            logger.debug(f"[CREATE_GRAPH] 入口: name={name}, graph_id={graph_id}")
            if graph_id is None:
                graph_id = f"mirofish_{uuid.uuid4().hex[:16]}"

            neo4j = self._adapter._get_neo4j()

            # 在 Neo4j 创建元数据节点
            query = """
                MERGE (meta:GraphMetadata {graph_id: $graph_id})
                SET meta.name = $name,
                    meta.description = $description,
                    meta.created_at = datetime(),
                    meta.entity_types = $entity_types,
                    meta.edge_types = $edge_types
                RETURN meta.graph_id as graph_id
            """

            parameters = {
                "graph_id": graph_id,
                "name": name,
                "description": description,
                "entity_types": json.dumps([]),
                "edge_types": json.dumps([]),
            }

            try:
                neo4j._execute_with_retry(query, parameters)
                logger.info(f"[CREATE_GRAPH] 出口: graph_id={graph_id}, name={name}")
                return graph_id
            except Exception as e:
                logger.error(
                    f"[CREATE_GRAPH] 失败: graph_id={graph_id}, name={name}, error={e!r}",
                    exc_info=True,
                )
                raise

        def delete(self, graph_id: str) -> None:
            """
            删除图谱。

            使用 DETACH DELETE 删除所有关联节点和边。

            Args:
                graph_id: 图谱 ID
            """
            logger.debug(f"[DELETE_GRAPH] 入口: graph_id={graph_id}")
            neo4j = self._adapter._get_neo4j()

            # 删除元数据节点
            query_meta = """
                MATCH (meta:GraphMetadata {graph_id: $graph_id})
                DELETE meta
            """

            # 删除所有实体节点（DETACH DELETE 会同时删除边）
            query_entities = """
                MATCH (n:Entity {graph_id: $graph_id})
                DETACH DELETE n
            """

            # 删除所有边（以防有遗漏）
            query_edges = """
                MATCH ()-[r]->()
                WHERE r.graph_id = $graph_id
                DELETE r
            """

            try:
                neo4j._execute_with_retry(query_edges, {"graph_id": graph_id})
                neo4j._execute_with_retry(query_entities, {"graph_id": graph_id})
                neo4j._execute_with_retry(query_meta, {"graph_id": graph_id})
                logger.info(f"[DELETE_GRAPH] 出口: graph_id={graph_id}")
            except Exception as e:
                logger.error(
                    f"[DELETE_GRAPH] 失败: graph_id={graph_id}, error={e!r}",
                    exc_info=True,
                )
                raise

        def set_ontology(
            self,
            graph_ids: list[str],
            entities: Optional[dict[str, Any]] = None,
            edges: Optional[dict[str, Any]] = None,
        ) -> None:
            """
            设置图谱本体。

            将本体定义存储到 Neo4j 元数据节点。
            注意：mem0 会自动从文本中提取实体和关系，本体主要用于指导提取。

            Args:
                graph_ids: 图谱 ID 列表
                entities: 实体类型定义
                edges: 边类型定义
            """
            if not graph_ids:
                return

            neo4j = self._adapter._get_neo4j()

            # 序列化本体定义
            entity_types = []
            if entities:
                for name, cls in entities.items():
                    entity_types.append(
                        {
                            "name": name,
                            "doc": cls.__doc__ or "",
                        }
                    )

            edge_types = []
            if edges:
                for name, (cls, source_targets) in edges.items():
                    edge_types.append(
                        {
                            "name": name,
                            "doc": cls.__doc__ or "",
                            "source_targets": [
                                {"source": st.source, "target": st.target}
                                for st in source_targets
                            ]
                            if source_targets
                            else [],
                        }
                    )

            query = """
                MATCH (meta:GraphMetadata {graph_id: $graph_id})
                SET meta.entity_types = $entity_types,
                    meta.edge_types = $edge_types,
                    meta.updated_at = datetime()
            """

            for graph_id in graph_ids:
                try:
                    neo4j._execute_with_retry(
                        query,
                        {
                            "graph_id": graph_id,
                            "entity_types": json.dumps(entity_types),
                            "edge_types": json.dumps(edge_types),
                        },
                    )
                    logger.info(
                        f"设置图谱 {graph_id} 的本体: {len(entity_types)} 实体类型, {len(edge_types)} 边类型"
                    )
                except Exception as e:
                    logger.error(
                        f"[SET_ONTOLOGY] 失败: graph_id={graph_id}, entity_types={len(entity_types)}, edge_types={len(edge_types)}, error={e!r}",
                        exc_info=True,
                    )
                    raise

        def add(
            self,
            graph_id: str,
            type: str = "text",
            data: str = "",
            **kwargs,
        ) -> dict[str, Any]:
            """
            添加单条数据到图谱。

            Args:
                graph_id: 图谱 ID
                type: 数据类型（目前只支持 "text"）
                data: 数据内容

            Returns:
                添加结果，包含 uuid
            """
            logger.info(
                f"[MEM0_ADD] 入口: graph_id={graph_id}, type={type}, data_len={len(data)}"
            )

            if type != "text":
                logger.warning(f"[MEM0_ADD] 不支持的数据类型: {type}，将作为文本处理")

            memory = self._adapter._get_memory()

            try:
                logger.debug(f"[MEM0_ADD] 调用 memory.add: user_id={graph_id}")
                result = memory.add(
                    [{"role": "user", "content": data}],
                    user_id=graph_id,
                )
                logger.debug(
                    f"[MEM0_ADD] 原始响应: type={type(result)}, value={result!r}"
                )

                logger.debug(
                    f"[MEM0_ADD] 解析: isinstance(result, dict)={isinstance(result, dict)}"
                )
                if isinstance(result, dict):
                    logger.debug(f"[MEM0_ADD] 解析: keys={list(result.keys())}")
                    results_list = result.get("results", [])
                    relations = result.get("relations", [])
                    logger.debug(
                        f"[MEM0_ADD] 解析: results_count={len(results_list)}, relations_count={len(relations) if relations else 0}"
                    )
                    if relations:
                        logger.debug(
                            f"[MEM0_ADD] 解析: relations_preview={relations[:2]}..."
                        )

                # 提取 uuid - 兼容 mem0 v1.0.0+ API 格式 {"results": [...]}
                uuid_ = None
                if result and isinstance(result, dict):
                    results_list = result.get("results", [])
                elif isinstance(result, list):
                    results_list = result  # 兼容旧版本
                else:
                    results_list = []

                if results_list and len(results_list) > 0:
                    first_result = results_list[0]
                    if isinstance(first_result, dict):
                        uuid_ = first_result.get("id") or first_result.get("uuid")
                    elif hasattr(first_result, "id"):
                        uuid_ = first_result.id
                    elif hasattr(first_result, "uuid"):
                        uuid_ = first_result.uuid

                uuid_final = uuid_ or str(uuid.uuid4())
                logger.info(f"[MEM0_ADD] 出口: uuid={uuid_final}")

                return {"uuid": uuid_final}

            except Exception as e:
                logger.error(
                    f"[MEM0_ADD] 失败: graph_id={graph_id}, type={type}, data_len={len(data)}, error={e!r}",
                    exc_info=True,
                )
                raise

        def add_batch(
            self,
            graph_id: str,
            episodes: list[EpisodeData],
            **kwargs,
        ) -> list[dict[str, Any]]:
            """
            批量添加数据到图谱。

            Args:
                graph_id: 图谱 ID
                episodes: Episode 数据列表

            Returns:
                添加结果列表
            """
            logger.info(
                f"[MEM0_ADD_BATCH] 入口: graph_id={graph_id}, episodes_count={len(episodes)}"
            )
            memory = self._adapter._get_memory()
            results = []
            total_relations = 0
            total = len(episodes)

            for i, episode in enumerate(episodes):
                try:
                    logger.debug(
                        f"[MEM0_ADD_BATCH] 批次 [{i + 1}/{total}]: data_len={len(episode.data)}"
                    )
                    result = memory.add(
                        [{"role": "user", "content": episode.data}],
                        user_id=graph_id,
                    )
                    logger.debug(
                        f"[MEM0_ADD_BATCH] 批次 [{i + 1}] 原始响应: type={type(result)}, value={result!r}"
                    )

                    if isinstance(result, dict):
                        results_list = result.get("results", [])
                        relations = result.get("relations", [])
                        logger.debug(
                            f"[MEM0_ADD_BATCH] 批次 [{i + 1}] 解析: results_count={len(results_list)}, relations_count={len(relations) if relations else 0}"
                        )
                        if relations:
                            total_relations += len(relations)

                    uuid_ = None
                    if result and isinstance(result, dict):
                        results_list = result.get("results", [])
                    elif isinstance(result, list):
                        results_list = result
                    else:
                        results_list = []

                    if results_list and len(results_list) > 0:
                        first_result = results_list[0]
                        if isinstance(first_result, dict):
                            uuid_ = first_result.get("id") or first_result.get("uuid")
                        elif hasattr(first_result, "id"):
                            uuid_ = first_result.id
                        elif hasattr(first_result, "uuid"):
                            uuid_ = first_result.uuid

                    results.append({"uuid": uuid_ or str(uuid.uuid4())})

                except Exception as e:
                    logger.error(
                        f"[MEM0_ADD_BATCH] 批次 [{i + 1}/{total}] 失败: graph_id={graph_id}, data_len={len(episode.data)}, error={e!r}",
                        exc_info=True,
                    )
                    raise

            logger.info(
                f"[MEM0_ADD_BATCH] 出口: results={len(results)}, total_relations={total_relations}"
            )
            return results

        def search(
            self,
            graph_id: str,
            query: str,
            limit: int = 10,
            scope: str = "edges",
            reranker: str = "cross_encoder",
            **kwargs,
        ) -> SearchResult:
            """
            语义搜索图谱。

            Args:
                graph_id: 图谱 ID
                query: 搜索查询
                limit: 返回结果数量限制
                scope: 搜索范围（"edges", "nodes", "both"）
                reranker: 重排序器（mem0 内置处理）

            Returns:
                SearchResult 包含 facts, edges, nodes
            """
            logger.info(
                f"[MEM0_SEARCH] 入口: graph_id={graph_id}, query={query[:50]}..., limit={limit}, scope={scope}"
            )
            memory = self._adapter._get_memory()
            neo4j = self._adapter._get_neo4j()

            facts = []
            edges = []
            nodes = []

            try:
                logger.debug(
                    f"[MEM0_SEARCH] 调用 memory.search: user_id={graph_id}, limit={limit}"
                )
                search_result = memory.search(query, user_id=graph_id, limit=limit)
                logger.debug(
                    f"[MEM0_SEARCH] 原始响应: type={type(search_result)}, value={search_result!r}"
                )

                if search_result and isinstance(search_result, dict):
                    logger.debug(
                        f"[MEM0_SEARCH] 解析: keys={list(search_result.keys())}"
                    )
                    results_list = search_result.get("results", [])
                    logger.debug(
                        f"[MEM0_SEARCH] 解析: results_count={len(results_list)}"
                    )
                elif isinstance(search_result, list):
                    results_list = search_result
                    logger.debug(f"[MEM0_SEARCH] 解析: list_count={len(results_list)}")
                else:
                    results_list = []
                    logger.debug(
                        f"[MEM0_SEARCH] 解析: empty_or_unknown_type, results_count=0"
                    )

                for item in results_list:
                    fact_data = {
                        "fact": item.get("memory", "")
                        if isinstance(item, dict)
                        else str(item),
                        "score": item.get("score", 0) if isinstance(item, dict) else 0,
                    }
                    facts.append(fact_data)

                if scope in ("edges", "both"):
                    edges = self._get_related_edges(neo4j, graph_id, facts[:limit])

                if scope in ("nodes", "both"):
                    nodes = self._get_related_nodes(neo4j, graph_id, facts[:limit])

                logger.info(
                    f"[MEM0_SEARCH] 出口: facts={len(facts)}, edges={len(edges)}, nodes={len(nodes)}"
                )

                return SearchResult(facts=facts, edges=edges, nodes=nodes)

            except Exception as e:
                logger.error(
                    f"[MEM0_SEARCH] 失败: graph_id={graph_id}, query={query}, limit={limit}, error={e!r}",
                    exc_info=True,
                )
                raise

        def _get_related_edges(
            self,
            neo4j: Neo4jClient,
            graph_id: str,
            facts: list[dict],
        ) -> list[EdgeInfo]:
            """根据搜索结果获取相关边。"""
            if not facts:
                return []

            # 从事实中提取可能的实体名称
            entity_names = []
            for fact in facts:
                text = fact.get("fact", "")
                # 简单提取：获取长度大于1的词
                words = [w.strip() for w in text.split() if len(w.strip()) > 1]
                entity_names.extend(words[:5])  # 每条事实取前5个词

            if not entity_names:
                return []

            query = """
                MATCH (source:Entity {graph_id: $graph_id})-[r]->(target:Entity)
                WHERE source.name CONTAINS ANY($names) OR target.name CONTAINS ANY($names)
                RETURN r, source.uuid as source_uuid, source.name as source_name,
                       target.uuid as target_uuid, target.name as target_name
                LIMIT $limit
            """

            try:
                records = neo4j._execute_with_retry(
                    query,
                    {"graph_id": graph_id, "names": entity_names[:20], "limit": 50},
                )

                edges = []
                for record in records:
                    edge_data = record.get("r", {})
                    if hasattr(edge_data, "_properties"):
                        props = dict(edge_data._properties)
                    else:
                        props = dict(edge_data) if edge_data else {}

                    edge = EdgeInfo(
                        uuid=props.get("uuid", str(uuid.uuid4())),
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
                        attributes=props.get("attributes") or {},
                        episodes=props.get("episodes") or [],
                    )
                    edges.append(edge)

                return edges

            except Exception as e:
                logger.error(
                    f"[MEM0_GET_EDGES] 失败: graph_id={graph_id}, facts_count={len(facts)}, error={e!r}",
                    exc_info=True,
                )
                raise

        def _get_related_nodes(
            self,
            neo4j: Neo4jClient,
            graph_id: str,
            facts: list[dict],
        ) -> list[EntityNode]:
            """根据搜索结果获取相关节点。"""
            if not facts:
                return []

            # 从事实中提取可能的实体名称
            entity_names = []
            for fact in facts:
                text = fact.get("fact", "")
                words = [w.strip() for w in text.split() if len(w.strip()) > 1]
                entity_names.extend(words[:5])

            if not entity_names:
                return []

            query = """
                MATCH (n:Entity {graph_id: $graph_id})
                WHERE n.name CONTAINS ANY($names)
                RETURN n
                LIMIT $limit
            """

            try:
                records = neo4j._execute_with_retry(
                    query,
                    {"graph_id": graph_id, "names": entity_names[:20], "limit": 50},
                )

                nodes = []
                for record in records:
                    node_data = record.get("n", {})
                    if hasattr(node_data, "_properties"):
                        props = dict(node_data._properties)
                    else:
                        props = dict(node_data) if node_data else {}

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
                    nodes.append(node)

                return nodes

            except Exception as e:
                logger.error(
                    f"[MEM0_GET_NODES] 失败: graph_id={graph_id}, facts_count={len(facts)}, error={e!r}",
                    exc_info=True,
                )
                raise

    class _Node:
        """Node 子接口。"""

        def __init__(self, adapter: ZepGraphAdapter):
            self._adapter = adapter

        def get(self, uuid_: str) -> Optional[EntityNode]:
            """
            获取单个节点。

            Args:
                uuid_: 节点 UUID

            Returns:
                EntityNode 或 None
            """
            neo4j = self._adapter._get_neo4j()

            query = """
                MATCH (n {uuid: $uuid})
                RETURN n
                LIMIT 1
            """

            try:
                records = neo4j._execute_with_retry(query, {"uuid": uuid_})

                if not records:
                    return None

                node_data = records[0].get("n", {})
                if not node_data:
                    return None

                if hasattr(node_data, "_properties"):
                    props = dict(node_data._properties)
                else:
                    props = dict(node_data)

                return EntityNode(
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

            except Exception as e:
                logger.error(
                    f"[MEM0_NODE_GET] 失败: uuid={uuid_}, error={e!r}", exc_info=True
                )
                raise

        def get_by_graph_id(
            self,
            graph_id: str,
            limit: int = 100,
            uuid_cursor: Optional[str] = None,
            **kwargs,
        ) -> list[EntityNode]:
            """
            分页获取图谱节点。

            Args:
                graph_id: 图谱 ID
                limit: 返回数量限制
                uuid_cursor: UUID 分页游标

            Returns:
                EntityNode 列表
            """
            neo4j = self._adapter._get_neo4j()
            return neo4j.fetch_all_nodes(
                graph_id=graph_id,
                limit=limit,
                cursor=uuid_cursor,
            )

        def get_entity_edges(self, node_uuid: str) -> list[EdgeInfo]:
            """
            获取节点的所有边。

            Args:
                node_uuid: 节点 UUID

            Returns:
                EdgeInfo 列表
            """
            neo4j = self._adapter._get_neo4j()

            # 使用 neo4j_client 的 get_entity_edges 方法
            edge_dicts = neo4j.get_entity_edges(node_uuid)

            # 转换为 EdgeInfo 对象
            edges = []
            for edge_dict in edge_dicts:
                edge = EdgeInfo(
                    uuid=edge_dict.get("uuid", ""),
                    name=edge_dict.get("name", ""),
                    fact=edge_dict.get("fact", ""),
                    source_node_uuid=edge_dict.get("source_uuid", ""),
                    target_node_uuid=edge_dict.get("target_uuid", ""),
                    source_node_name=edge_dict.get("related_name"),
                    target_node_name=None,  # 需要额外查询
                    created_at=edge_dict.get("created_at"),
                    valid_at=edge_dict.get("valid_at"),
                    invalid_at=edge_dict.get("invalid_at"),
                    expired_at=edge_dict.get("expired_at"),
                    attributes=edge_dict.get("attributes") or {},
                    episodes=edge_dict.get("episodes") or [],
                )
                edges.append(edge)

            return edges

    class _Edge:
        """Edge 子接口。"""

        def __init__(self, adapter: ZepGraphAdapter):
            self._adapter = adapter

        def get_by_graph_id(
            self,
            graph_id: str,
            limit: int = 100,
            uuid_cursor: Optional[str] = None,
            **kwargs,
        ) -> list[EdgeInfo]:
            """
            分页获取图谱边。

            Args:
                graph_id: 图谱 ID
                limit: 返回数量限制
                uuid_cursor: UUID 分页游标

            Returns:
                EdgeInfo 列表
            """
            neo4j = self._adapter._get_neo4j()
            return neo4j.fetch_all_edges(
                graph_id=graph_id,
                limit=limit,
                cursor=uuid_cursor,
            )

    class _Episode:
        """Episode 子接口（mem0 同步处理，此接口主要用于兼容）。"""

        def __init__(self, adapter: ZepGraphAdapter):
            self._adapter = adapter

        def get(self, uuid_: str) -> Optional[EpisodeInfo]:
            """
            获取 Episode 信息。

            注意：mem0 是同步处理的，所以 processed 始终为 True。

            Args:
                uuid_: Episode UUID

            Returns:
                EpisodeInfo 或 None
            """
            # mem0 是同步处理，不需要等待
            # 返回一个简单的 EpisodeInfo 表示已完成
            return EpisodeInfo(
                uuid_=uuid_,
                data="",
                processed=True,
                created_at=datetime.now().isoformat(),
            )


def create_zep_compatible_client(api_key: Optional[str] = None) -> ZepGraphAdapter:
    """
    创建 Zep 兼容客户端。

    这是推荐的创建适配器实例的方式，保持与 Zep SDK 的初始化方式兼容。

    Args:
        api_key: API 密钥（保留参数，不实际使用）

    Returns:
        ZepGraphAdapter 实例

    Example:
        # 替换 Zep 客户端
        # from zep_cloud.client import Zep
        # client = Zep(api_key="...")
        client = create_zep_compatible_client()  # 无需 API Key

        # 使用方式与 Zep SDK 完全相同
        graph_id = client.graph.create(name="My Graph")
    """
    return ZepGraphAdapter(api_key=api_key)
