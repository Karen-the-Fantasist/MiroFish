"""
Neo4j 集成测试。

测试需要真实 Neo4j 数据库连接，使用 pytest skipif 标记。
运行测试前确保 Neo4j 服务已启动：
    docker compose up -d neo4j

运行集成测试：
    cd backend && uv run pytest tests/integration/ -v

环境变量（可选，覆盖默认值）：
    NEO4J_URL: Neo4j 连接 URL（默认: bolt://127.0.0.1:7687）
    NEO4J_USERNAME: Neo4j 用户名（默认: neo4j）
    NEO4J_PASSWORD: Neo4j 密码（默认: 空）
"""

import os
import uuid
from datetime import datetime

import pytest


# 检查 Neo4j 是否可用的辅助函数
def _check_neo4j_available() -> bool:
    """检查 Neo4j 是否可用。"""
    try:
        from neo4j import GraphDatabase

        url = os.environ.get("NEO4J_URL", "bolt://127.0.0.1:7687")
        username = os.environ.get("NEO4J_USERNAME", "neo4j")
        password = os.environ.get("NEO4J_PASSWORD", "")

        driver = GraphDatabase.driver(
            url,
            auth=(username, password),
            connection_timeout=5.0,
        )
        with driver.session() as session:
            result = session.run("RETURN 1 as test")
            record = result.single()
            driver.close()
            return record is not None and record["test"] == 1
    except Exception:
        return False


# 跳过标记：当 Neo4j 不可用时跳过测试
neo4j_required = pytest.mark.skipif(
    not _check_neo4j_available(),
    reason="Neo4j 服务不可用，跳过集成测试。请启动 Neo4j: docker compose up -d neo4j",
)

# 测试数据前缀，用于隔离测试数据
TEST_PREFIX = f"test_integration_{datetime.now().strftime('%Y%m%d_%H%M%S')}_"


@pytest.fixture(scope="module")
def neo4j_client():
    """创建 Neo4j 客户端实例（模块级别）。"""
    from app.utils.neo4j_client import Neo4jClient

    # 重置单例以确保使用新配置
    Neo4jClient.reset_instance()
    client = Neo4jClient()
    yield client

    # 清理：关闭连接
    client.close()
    Neo4jClient.reset_instance()


@pytest.fixture(scope="module")
def zep_adapter():
    """创建 ZepGraphAdapter 实例（模块级别）。"""
    from app.adapters.zep_graph_adapter import ZepGraphAdapter

    adapter = ZepGraphAdapter()
    yield adapter


@pytest.fixture(scope="function")
def unique_graph_id():
    """生成唯一的图谱 ID（函数级别）。"""
    return f"{TEST_PREFIX}{uuid.uuid4().hex[:8]}"


class TestNeo4jConnection:
    """测试 Neo4j 连接。"""

    @neo4j_required
    def test_connection_success(self, neo4j_client):
        """测试 Neo4j 连接成功。"""
        result = neo4j_client.test_connection()
        assert result is True, "Neo4j 连接测试应该成功"

    @neo4j_required
    def test_connection_with_session(self, neo4j_client):
        """测试通过 session 执行简单查询。"""
        with neo4j_client._get_session() as session:
            result = session.run("RETURN 'hello' as message")
            record = result.single()
            assert record is not None
            assert record["message"] == "hello"


class TestGraphLifecycle:
    """测试图谱完整生命周期：创建 → 添加 → 搜索 → 删除。"""

    @neo4j_required
    def test_create_graph(self, zep_adapter, unique_graph_id):
        """测试创建图谱。"""
        graph_id = zep_adapter.graph.create(
            graph_id=unique_graph_id,
            name="Test Graph",
            description="Integration test graph",
        )

        assert graph_id == unique_graph_id

        # 验证图谱元数据节点已创建
        neo4j = zep_adapter._get_neo4j()
        query = """
            MATCH (meta:GraphMetadata {graph_id: $graph_id})
            RETURN meta.name as name, meta.description as description
        """
        records = neo4j._execute_with_retry(query, {"graph_id": unique_graph_id})

        assert len(records) == 1
        assert records[0]["name"] == "Test Graph"
        assert records[0]["description"] == "Integration test graph"

        # 清理
        zep_adapter.graph.delete(unique_graph_id)

    @neo4j_required
    def test_full_lifecycle(self, zep_adapter, unique_graph_id):
        """测试完整生命周期：创建 → 添加 → 搜索 → 删除。"""
        # 1. 创建图谱
        graph_id = zep_adapter.graph.create(
            graph_id=unique_graph_id,
            name="Lifecycle Test Graph",
        )
        assert graph_id == unique_graph_id

        try:
            # 2. 直接在 Neo4j 中添加测试节点和边
            # 注意：mem0.add() 需要 LLM API，这里直接操作 Neo4j 测试适配层
            neo4j = zep_adapter._get_neo4j()

            # 创建实体节点
            create_nodes_query = """
                CREATE (alice:Entity {
                    uuid: $alice_uuid,
                    name: 'Alice',
                    graph_id: $graph_id,
                    summary: 'A test person',
                    created_at: datetime()
                })
                CREATE (google:Entity {
                    uuid: $google_uuid,
                    name: 'Google',
                    graph_id: $graph_id,
                    summary: 'A technology company',
                    created_at: datetime()
                })
                CREATE (alice)-[r:WORKS_AT {
                    uuid: $edge_uuid,
                    name: 'WORKS_AT',
                    fact: 'Alice works at Google',
                    graph_id: $graph_id,
                    created_at: datetime()
                }]->(google)
                RETURN alice, google
            """

            test_uuid = uuid.uuid4().hex
            neo4j._execute_with_retry(
                create_nodes_query,
                {
                    "graph_id": unique_graph_id,
                    "alice_uuid": f"{test_uuid}_alice",
                    "google_uuid": f"{test_uuid}_google",
                    "edge_uuid": f"{test_uuid}_edge",
                },
            )

            # 3. 验证节点可以查询
            nodes = zep_adapter.node.get_by_graph_id(graph_id=unique_graph_id, limit=10)
            assert len(nodes) == 2, f"应该有2个节点，实际有 {len(nodes)} 个"
            node_names = {n.name for n in nodes}
            assert "Alice" in node_names
            assert "Google" in node_names

            # 4. 验证边可以查询
            edges = zep_adapter.edge.get_by_graph_id(graph_id=unique_graph_id, limit=10)
            assert len(edges) == 1, f"应该有1条边，实际有 {len(edges)} 条"
            assert edges[0].name == "WORKS_AT"
            assert edges[0].fact == "Alice works at Google"

        finally:
            # 5. 删除图谱
            zep_adapter.graph.delete(unique_graph_id)

            # 验证删除成功
            verify_query = """
                MATCH (n {graph_id: $graph_id})
                RETURN count(n) as count
            """
            records = neo4j._execute_with_retry(
                verify_query, {"graph_id": unique_graph_id}
            )
            assert records[0]["count"] == 0, "图谱删除后应该没有残留节点"


class TestPagination:
    """测试节点/边分页查询。"""

    @neo4j_required
    def test_node_pagination(self, neo4j_client, unique_graph_id):
        """测试节点分页查询。"""
        # 创建测试节点
        create_query = """
            UNWIND range(1, 25) as i
            CREATE (n:Entity {
                uuid: $prefix + toString(i),
                name: 'Node_' + toString(i),
                graph_id: $graph_id,
                summary: 'Test node ' + toString(i)
            })
        """

        neo4j_client._execute_with_retry(
            create_query,
            {"graph_id": unique_graph_id, "prefix": f"{unique_graph_id}_node_"},
        )

        try:
            # 测试分页：第一页
            page1 = neo4j_client.fetch_all_nodes(
                graph_id=unique_graph_id, limit=10, page_size=5
            )
            assert len(page1) == 10, f"第一页应该有10个节点，实际有 {len(page1)} 个"

            # 测试分页：使用游标获取下一页
            cursor = page1[-1].uuid
            page2 = neo4j_client.fetch_all_nodes(
                graph_id=unique_graph_id, limit=10, cursor=cursor, page_size=5
            )
            assert len(page2) == 10, f"第二页应该有10个节点，实际有 {len(page2)} 个"

            # 验证没有重复
            page1_uuids = {n.uuid for n in page1}
            page2_uuids = {n.uuid for n in page2}
            assert len(page1_uuids & page2_uuids) == 0, "分页结果不应该有重复"

            # 测试获取全部
            all_nodes = neo4j_client.fetch_all_nodes(
                graph_id=unique_graph_id, limit=100, page_size=10
            )
            assert len(all_nodes) == 25, (
                f"总共应该有25个节点，实际有 {len(all_nodes)} 个"
            )

        finally:
            # 清理
            cleanup_query = """
                MATCH (n:Entity {graph_id: $graph_id})
                DETACH DELETE n
            """
            neo4j_client._execute_with_retry(
                cleanup_query, {"graph_id": unique_graph_id}
            )

    @neo4j_required
    def test_edge_pagination(self, neo4j_client, unique_graph_id):
        """测试边分页查询。"""
        # 创建测试节点和边
        create_query = """
            CREATE (source:Entity {
                uuid: $source_uuid,
                name: 'Source',
                graph_id: $graph_id
            })
            WITH source
            UNWIND range(1, 20) as i
            CREATE (target:Entity {
                uuid: $prefix + toString(i),
                name: 'Target_' + toString(i),
                graph_id: $graph_id
            })
            CREATE (source)-[r:RELATES_TO {
                uuid: $edge_prefix + toString(i),
                name: 'RELATES_TO',
                fact: 'Edge ' + toString(i),
                graph_id: $graph_id
            }]->(target)
        """

        neo4j_client._execute_with_retry(
            create_query,
            {
                "graph_id": unique_graph_id,
                "source_uuid": f"{unique_graph_id}_source",
                "prefix": f"{unique_graph_id}_target_",
                "edge_prefix": f"{unique_graph_id}_edge_",
            },
        )

        try:
            # 测试分页：获取前10条边
            page1 = neo4j_client.fetch_all_edges(
                graph_id=unique_graph_id, limit=10, page_size=5
            )
            assert len(page1) == 10, f"第一页应该有10条边，实际有 {len(page1)} 条"

            # 测试分页：使用游标获取下一页
            cursor = page1[-1].uuid
            page2 = neo4j_client.fetch_all_edges(
                graph_id=unique_graph_id, limit=10, cursor=cursor, page_size=5
            )
            assert len(page2) == 10, f"第二页应该有10条边，实际有 {len(page2)} 条"

            # 验证没有重复
            page1_uuids = {e.uuid for e in page1}
            page2_uuids = {e.uuid for e in page2}
            assert len(page1_uuids & page2_uuids) == 0, "分页结果不应该有重复"

            # 测试获取全部
            all_edges = neo4j_client.fetch_all_edges(
                graph_id=unique_graph_id, limit=100, page_size=10
            )
            assert len(all_edges) == 20, f"总共应该有20条边，实际有 {len(all_edges)} 条"

        finally:
            # 清理
            cleanup_query = """
                MATCH (n:Entity {graph_id: $graph_id})
                DETACH DELETE n
            """
            neo4j_client._execute_with_retry(
                cleanup_query, {"graph_id": unique_graph_id}
            )

    @neo4j_required
    def test_empty_pagination(self, neo4j_client, unique_graph_id):
        """测试空图谱的分页查询。"""
        # 空图谱应该返回空列表
        nodes = neo4j_client.fetch_all_nodes(graph_id=unique_graph_id, limit=10)
        assert nodes == [], "空图谱应该返回空节点列表"

        edges = neo4j_client.fetch_all_edges(graph_id=unique_graph_id, limit=10)
        assert edges == [], "空图谱应该返回空边列表"


class TestEntityEdges:
    """测试节点边查询。"""

    @neo4j_required
    def test_get_entity_edges(self, neo4j_client, unique_graph_id):
        """测试获取节点的所有边。"""
        # 创建测试数据：一个中心节点连接多个其他节点
        create_query = """
            CREATE (center:Entity {
                uuid: $center_uuid,
                name: 'Center',
                graph_id: $graph_id
            })
            CREATE (node1:Entity {
                uuid: $node1_uuid,
                name: 'Node1',
                graph_id: $graph_id
            })
            CREATE (node2:Entity {
                uuid: $node2_uuid,
                name: 'Node2',
                graph_id: $graph_id
            })
            CREATE (center)-[r1:KNOWS {
                uuid: $edge1_uuid,
                name: 'KNOWS',
                fact: 'Center knows Node1',
                graph_id: $graph_id
            }]->(node1)
            CREATE (node2)-[r2:WORKS_WITH {
                uuid: $edge2_uuid,
                name: 'WORKS_WITH',
                fact: 'Node2 works with Center',
                graph_id: $graph_id
            }]->(center)
        """

        center_uuid = f"{unique_graph_id}_center"

        neo4j_client._execute_with_retry(
            create_query,
            {
                "graph_id": unique_graph_id,
                "center_uuid": center_uuid,
                "node1_uuid": f"{unique_graph_id}_node1",
                "node2_uuid": f"{unique_graph_id}_node2",
                "edge1_uuid": f"{unique_graph_id}_edge1",
                "edge2_uuid": f"{unique_graph_id}_edge2",
            },
        )

        try:
            # 获取中心节点的所有边
            edges = neo4j_client.get_entity_edges(center_uuid)

            assert len(edges) == 2, f"中心节点应该有2条边，实际有 {len(edges)} 条"
            edge_names = {e["name"] for e in edges}
            assert "KNOWS" in edge_names
            assert "WORKS_WITH" in edge_names

            # 验证边的方向
            for edge in edges:
                if edge["name"] == "KNOWS":
                    assert edge["source_uuid"] == center_uuid
                    assert edge["related_name"] == "Node1"
                elif edge["name"] == "WORKS_WITH":
                    assert edge["target_uuid"] == center_uuid
                    assert edge["related_name"] == "Node2"

        finally:
            # 清理
            cleanup_query = """
                MATCH (n:Entity {graph_id: $graph_id})
                DETACH DELETE n
            """
            neo4j_client._execute_with_retry(
                cleanup_query, {"graph_id": unique_graph_id}
            )


class TestDataIsolation:
    """测试数据隔离。"""

    @neo4j_required
    def test_graph_isolation(self, neo4j_client, unique_graph_id):
        """测试不同图谱之间的数据隔离。"""
        # 创建第一个图谱的数据
        graph1_id = f"{unique_graph_id}_graph1"
        graph2_id = f"{unique_graph_id}_graph2"

        create_query = """
            CREATE (n1:Entity {
                uuid: $uuid1,
                name: 'Graph1Node',
                graph_id: $graph1_id
            })
            CREATE (n2:Entity {
                uuid: $uuid2,
                name: 'Graph2Node',
                graph_id: $graph2_id
            })
        """

        neo4j_client._execute_with_retry(
            create_query,
            {
                "graph1_id": graph1_id,
                "graph2_id": graph2_id,
                "uuid1": f"{graph1_id}_node",
                "uuid2": f"{graph2_id}_node",
            },
        )

        try:
            # 验证图谱隔离
            nodes1 = neo4j_client.fetch_all_nodes(graph_id=graph1_id, limit=10)
            nodes2 = neo4j_client.fetch_all_nodes(graph_id=graph2_id, limit=10)

            assert len(nodes1) == 1
            assert len(nodes2) == 1
            assert nodes1[0].name == "Graph1Node"
            assert nodes2[0].name == "Graph2Node"

        finally:
            # 清理
            for gid in [graph1_id, graph2_id]:
                cleanup_query = """
                    MATCH (n:Entity {graph_id: $graph_id})
                    DETACH DELETE n
                """
                neo4j_client._execute_with_retry(cleanup_query, {"graph_id": gid})
