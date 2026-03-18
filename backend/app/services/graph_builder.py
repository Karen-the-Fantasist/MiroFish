"""
图谱构建服务
接口2：使用Zep API构建Standalone Graph
"""

import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from app.adapters import ZepGraphAdapter
from app.adapters.zep_graph_adapter import EpisodeData
from app.utils.logger import get_logger

from ..models.task import TaskManager, TaskStatus

logger = get_logger("mirofish.graph_builder")
from ..utils.zep_paging import fetch_all_edges, fetch_all_nodes
from .text_processor import TextProcessor


@dataclass
class EntityEdgeSourceTarget:
    """Source-target pair for edge definitions, compatible with adapter interface."""

    source: str
    target: str


@dataclass
class GraphInfo:
    """图谱信息"""

    graph_id: str
    node_count: int
    edge_count: int
    entity_types: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "entity_types": self.entity_types,
        }


class GraphBuilderService:
    """
    图谱构建服务
    负责调用Zep API构建知识图谱
    """

    def __init__(self, api_key: str | None = None):
        # api_key 参数保留以保持接口兼容，但 ZepGraphAdapter 不使用它
        self.client = ZepGraphAdapter()
        self.task_manager = TaskManager()

    def build_graph_async(
        self,
        text: str,
        ontology: Dict[str, Any],
        graph_name: str = "MiroFish Graph",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        batch_size: int = 3,
    ) -> str:
        """
        异步构建图谱

        Args:
            text: 输入文本
            ontology: 本体定义（来自接口1的输出）
            graph_name: 图谱名称
            chunk_size: 文本块大小
            chunk_overlap: 块重叠大小
            batch_size: 每批发送的块数量

        Returns:
            任务ID
        """
        # 入口日志
        logger.info(
            f"[GRAPH_BUILD] 入口: graph_name={graph_name}, text_len={len(text)}, chunk_size={chunk_size}, batch_size={batch_size}"
        )

        # 创建任务
        task_id = self.task_manager.create_task(
            task_type="graph_build",
            metadata={
                "graph_name": graph_name,
                "chunk_size": chunk_size,
                "text_length": len(text),
            },
        )
        logger.debug(f"[GRAPH_BUILD] 创建任务: task_id={task_id}")

        # 在后台线程中执行构建
        thread = threading.Thread(
            target=self._build_graph_worker,
            args=(
                task_id,
                text,
                ontology,
                graph_name,
                chunk_size,
                chunk_overlap,
                batch_size,
            ),
        )
        thread.daemon = True
        thread.start()

        return task_id

    def _build_graph_worker(
        self,
        task_id: str,
        text: str,
        ontology: Dict[str, Any],
        graph_name: str,
        chunk_size: int,
        chunk_overlap: int,
        batch_size: int,
    ):
        """图谱构建工作线程"""
        logger.info(f"[GRAPH_BUILD_WORKER] 开始: task_id={task_id}")
        try:
            self.task_manager.update_task(
                task_id,
                status=TaskStatus.PROCESSING,
                progress=5,
                message="开始构建图谱...",
            )

            # 1. 创建图谱
            logger.debug("[GRAPH_BUILD_WORKER] 步骤 1/5: 创建图谱")
            graph_id = self.create_graph(graph_name)
            self.task_manager.update_task(
                task_id, progress=10, message=f"图谱已创建: {graph_id}"
            )

            # 2. 设置本体
            logger.debug(
                f"[GRAPH_BUILD_WORKER] 步骤 2/5: 设置本体, entity_types={len(ontology.get('entity_types', []))}"
            )
            self.set_ontology(graph_id, ontology)
            self.task_manager.update_task(task_id, progress=15, message="本体已设置")

            # 3. 文本分块
            chunks = TextProcessor.split_text(text, chunk_size, chunk_overlap)
            total_chunks = len(chunks)
            logger.debug(
                f"[GRAPH_BUILD_WORKER] 步骤 3/5: 文本分块, chunks={len(chunks)}"
            )
            self.task_manager.update_task(
                task_id, progress=20, message=f"文本已分割为 {total_chunks} 个块"
            )

            # 4. 分批发送数据
            logger.debug("[GRAPH_BUILD_WORKER] 步骤 4/5: 添加数据批次")
            self.add_text_batches(
                graph_id,
                chunks,
                batch_size,
                lambda msg, prog: self.task_manager.update_task(
                    task_id,
                    progress=20 + int(prog * 0.7),
                    message=msg,
                ),
            )

            # 5. 获取图谱信息
            logger.debug("[GRAPH_BUILD_WORKER] 步骤 5/5: 获取图谱信息")
            self.task_manager.update_task(
                task_id, progress=90, message="获取图谱信息..."
            )

            graph_info = self._get_graph_info(graph_id)

            # 完成
            logger.info(
                f"[GRAPH_BUILD_WORKER] 完成: task_id={task_id}, graph_id={graph_id}, nodes={graph_info.node_count}, edges={graph_info.edge_count}"
            )
            self.task_manager.complete_task(
                task_id,
                {
                    "graph_id": graph_id,
                    "graph_info": graph_info.to_dict(),
                    "chunks_processed": total_chunks,
                },
            )

        except Exception as e:
            logger.error(
                f"[GRAPH_BUILD_WORKER] 失败: task_id={task_id}, error={e!r}",
                exc_info=True,
            )
            import traceback

            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            self.task_manager.fail_task(task_id, error_msg)

    def create_graph(self, name: str) -> str:
        """创建Zep图谱（公开方法）"""
        logger.info(f"[CREATE_GRAPH] 入口: name={name}")
        try:
            graph_id = f"mirofish_{uuid.uuid4().hex[:16]}"

            self.client.graph.create(
                graph_id=graph_id,
                name=name,
                description="MiroFish Social Simulation Graph",
            )

            logger.info(f"[CREATE_GRAPH] 出口: graph_id={graph_id}")
            return graph_id
        except Exception as e:
            logger.error(
                f"[CREATE_GRAPH] 失败: name={name}, error={e!r}", exc_info=True
            )
            raise

    def set_ontology(self, graph_id: str, ontology: Dict[str, Any]):
        """设置图谱本体（公开方法）"""
        logger.info(
            f"[SET_ONTOLOGY] 入口: graph_id={graph_id}, entity_types={len(ontology.get('entity_types', []))}, edge_types={len(ontology.get('edge_types', []))}"
        )
        RESERVED_NAMES = {
            "uuid",
            "name",
            "group_id",
            "name_embedding",
            "summary",
            "created_at",
        }

        def safe_attr_name(attr_name: str) -> str:
            if attr_name.lower() in RESERVED_NAMES:
                return f"entity_{attr_name}"
            return attr_name

        # 动态创建实体类型（简化版，仅需 __doc__ 属性）
        entity_types = {}
        for entity_def in ontology.get("entity_types", []):
            name = entity_def["name"]
            description = entity_def.get("description", f"A {name} entity.")
            entity_class = type(name, (), {"__doc__": description})
            entity_types[name] = entity_class

        edge_definitions = {}
        for edge_def in ontology.get("edge_types", []):
            name = edge_def["name"]
            description = edge_def.get("description", f"A {name} relationship.")
            edge_class = type(name, (), {"__doc__": description})

            source_targets = []
            for st in edge_def.get("source_targets", []):
                source_targets.append(
                    EntityEdgeSourceTarget(
                        source=st.get("source", "Entity"),
                        target=st.get("target", "Entity"),
                    )
                )

            if source_targets:
                edge_definitions[name] = (edge_class, source_targets)

        if entity_types or edge_definitions:
            self.client.graph.set_ontology(
                graph_ids=[graph_id],
                entities=entity_types if entity_types else None,
                edges=edge_definitions if edge_definitions else None,
            )
        logger.debug(f"[SET_ONTOLOGY] 完成: graph_id={graph_id}")

    def add_text_batches(
        self,
        graph_id: str,
        chunks: List[str],
        batch_size: int = 3,
        progress_callback: Optional[Callable] = None,
    ) -> List[str]:
        """分批添加文本到图谱，返回所有 episode 的 uuid 列表"""
        logger.info(
            f"[ADD_BATCHES] 入口: graph_id={graph_id}, total_chunks={len(chunks)}, batch_size={batch_size}"
        )
        episode_uuids = []
        total_chunks = len(chunks)

        for i in range(0, total_chunks, batch_size):
            batch_chunks = chunks[i : i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total_chunks + batch_size - 1) // batch_size

            logger.debug(
                f"[ADD_BATCHES] 批次 [{batch_num}/{total_batches}]: 处理 {len(batch_chunks)} 块"
            )

            if progress_callback:
                progress = (i + len(batch_chunks)) / total_chunks
                progress_callback(
                    f"发送第 {batch_num}/{total_batches} 批数据 ({len(batch_chunks)} 块)...",
                    progress,
                )

            # 构建episode数据
            episodes = [EpisodeData(data=chunk, type="text") for chunk in batch_chunks]

            # 发送到Zep
            try:
                batch_result = self.client.graph.add_batch(
                    graph_id=graph_id, episodes=episodes
                )

                logger.debug(
                    f"[ADD_BATCHES] 批次 [{batch_num}] 结果: {len(batch_result)} 条"
                )

                # 收集返回的 episode uuid
                if batch_result and isinstance(batch_result, list):
                    for ep in batch_result:
                        ep_uuid = getattr(ep, "uuid_", None) or getattr(
                            ep, "uuid", None
                        )
                        if ep_uuid:
                            episode_uuids.append(ep_uuid)

                # 避免请求过快
                time.sleep(1)

            except Exception as e:
                logger.error(
                    f"[ADD_BATCHES] 批次 [{batch_num}] 失败: {e!r}", exc_info=True
                )
                if progress_callback:
                    progress_callback(f"批次 {batch_num} 发送失败: {str(e)}", 0)
                raise

        logger.info(
            f"[ADD_BATCHES] 出口: graph_id={graph_id}, total_episodes={len(episode_uuids)}"
        )
        return episode_uuids

    def _get_graph_info(self, graph_id: str) -> GraphInfo:
        """获取图谱信息"""
        logger.debug(f"[GET_GRAPH_INFO] 入口: graph_id={graph_id}")
        # 获取节点（分页）
        nodes = fetch_all_nodes(self.client, graph_id)

        # 获取边（分页）
        edges = fetch_all_edges(self.client, graph_id)

        # 统计实体类型
        entity_types = set()
        for node in nodes:
            if node.labels:
                for label in node.labels:
                    if label not in ["Entity", "Node"]:
                        entity_types.add(label)

        logger.info(
            f"[GET_GRAPH_INFO] 出口: graph_id={graph_id}, nodes={len(nodes)}, edges={len(edges)}, entity_types={entity_types}"
        )
        return GraphInfo(
            graph_id=graph_id,
            node_count=len(nodes),
            edge_count=len(edges),
            entity_types=list(entity_types),
        )

    def get_graph_data(self, graph_id: str) -> Dict[str, Any]:
        """
        获取完整图谱数据（包含详细信息）

        Args:
            graph_id: 图谱ID

        Returns:
            包含nodes和edges的字典，包括时间信息、属性等详细数据
        """
        logger.info(f"[GET_GRAPH_DATA] 入口: graph_id={graph_id}")
        nodes = fetch_all_nodes(self.client, graph_id)
        edges = fetch_all_edges(self.client, graph_id)

        # 创建节点映射用于获取节点名称
        node_map = {}
        for node in nodes:
            node_map[node.uuid_] = node.name or ""

        nodes_data = []
        for node in nodes:
            # 获取创建时间
            created_at = getattr(node, "created_at", None)
            if created_at:
                created_at = str(created_at)

            nodes_data.append(
                {
                    "uuid": node.uuid_,
                    "name": node.name,
                    "labels": node.labels or [],
                    "summary": node.summary or "",
                    "attributes": node.attributes or {},
                    "created_at": created_at,
                }
            )

        edges_data = []
        for edge in edges:
            # 获取时间信息
            created_at = getattr(edge, "created_at", None)
            valid_at = getattr(edge, "valid_at", None)
            invalid_at = getattr(edge, "invalid_at", None)
            expired_at = getattr(edge, "expired_at", None)

            # 获取 episodes
            episodes = getattr(edge, "episodes", None) or getattr(
                edge, "episode_ids", None
            )
            if episodes and not isinstance(episodes, list):
                episodes = [str(episodes)]
            elif episodes:
                episodes = [str(e) for e in episodes]

            # 获取 fact_type
            fact_type = getattr(edge, "fact_type", None) or edge.name or ""

            edges_data.append(
                {
                    "uuid": edge.uuid_,
                    "name": edge.name or "",
                    "fact": edge.fact or "",
                    "fact_type": fact_type,
                    "source_node_uuid": edge.source_node_uuid,
                    "target_node_uuid": edge.target_node_uuid,
                    "source_node_name": node_map.get(edge.source_node_uuid, ""),
                    "target_node_name": node_map.get(edge.target_node_uuid, ""),
                    "attributes": edge.attributes or {},
                    "created_at": str(created_at) if created_at else None,
                    "valid_at": str(valid_at) if valid_at else None,
                    "invalid_at": str(invalid_at) if invalid_at else None,
                    "expired_at": str(expired_at) if expired_at else None,
                    "episodes": episodes or [],
                }
            )

        logger.info(
            f"[GET_GRAPH_DATA] 出口: graph_id={graph_id}, nodes={len(nodes_data)}, edges={len(edges_data)}"
        )
        return {
            "graph_id": graph_id,
            "nodes": nodes_data,
            "edges": edges_data,
            "node_count": len(nodes_data),
            "edge_count": len(edges_data),
        }

    def delete_graph(self, graph_id: str):
        """删除图谱"""
        logger.info(f"[DELETE_GRAPH] 入口: graph_id={graph_id}")
        self.client.graph.delete(graph_id=graph_id)
        logger.info(f"[DELETE_GRAPH] 完成: graph_id={graph_id}")

    def _wait_for_episodes(
        self, episode_uuids: List[str], progress_callback: Optional[Callable] = None
    ):
        """
        等待 episodes 处理完成（mem0 同步处理，无需等待）。

        注意：此方法保留用于 API 兼容性。
        在 Zep Cloud 中，此方法会轮询等待异步处理完成。
        在 mem0 中，数据处理是同步的，直接返回即可。
        """
        # mem0 是同步处理，不需要等待
        if progress_callback:
            progress_callback("mem0 同步处理，无需等待", 1.0)
