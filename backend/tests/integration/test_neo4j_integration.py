"""
Neo4j 集成测试。

测试需要真实服务连接。运行前确保服务已启动：
    docker compose up -d neo4j
    bash scripts/start_embedding_service.sh

运行集成测试：
    cd backend && uv run pytest tests/integration/ -v

环境变量（可选，覆盖默认值）：
    NEO4J_URL: Neo4j 连接 URL（默认: bolt://127.0.0.1:7687）
    NEO4J_USERNAME: Neo4j 用户名（默认: neo4j）
    NEO4J_PASSWORD: Neo4j 密码（默认: 空）
    EMBEDDING_BASE_URL: Embedding 服务 URL（默认: http://127.0.0.1:8001/v1）
"""

import os
import uuid
from datetime import datetime

import pytest


def _check_neo4j_available() -> tuple[bool, str]:
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
            if record is not None and record["test"] == 1:
                return True, ""
            return False, f"Neo4j 返回异常结果: {record}"
    except Exception as e:
        return False, f"Neo4j 连接失败: {e}"


def _check_embedding_available() -> tuple[bool, str]:
    try:
        import requests

        base_url = os.environ.get("EMBEDDING_BASE_URL", "http://127.0.0.1:8001/v1")
        response = requests.get(f"{base_url.rstrip('/')}/models", timeout=5)
        if response.status_code == 200:
            return True, ""
        return False, f"Embedding 服务返回状态码: {response.status_code}"
    except Exception as e:
        return False, f"Embedding 服务连接失败: {e}"


def _check_llm_available() -> tuple[bool, str]:
    llm_api_key = os.environ.get("LLM_API_KEY")
    if llm_api_key is not None and len(llm_api_key) > 0:
        return True, ""
    return False, "LLM_API_KEY 未配置"


def require_neo4j():
    available, error = _check_neo4j_available()
    if not available:
        pytest.fail(
            f"Neo4j 服务不可用: {error}\n请启动 Neo4j: docker compose up -d neo4j"
        )


def require_embedding():
    available, error = _check_embedding_available()
    if not available:
        pytest.fail(
            f"Embedding 服务不可用: {error}\n请启动 vllm: bash scripts/start_embedding_service.sh"
        )


def require_llm():
    available, error = _check_llm_available()
    if not available:
        pytest.fail(f"LLM API 不可用: {error}\n请配置 LLM_API_KEY 环境变量")


def require_mem0_full():
    errors = []
    neo4j_ok, neo4j_err = _check_neo4j_available()
    if not neo4j_ok:
        errors.append(f"Neo4j: {neo4j_err}")

    embed_ok, embed_err = _check_embedding_available()
    if not embed_ok:
        errors.append(f"Embedding: {embed_err}")

    llm_ok, llm_err = _check_llm_available()
    if not llm_ok:
        errors.append(f"LLM: {llm_err}")

    if errors:
        pytest.fail(
            f"mem0 完整功能需要以下服务:\n" + "\n".join(f"  - {e}" for e in errors)
        )


TEST_PREFIX = f"test_integration_{datetime.now().strftime('%Y%m%d_%H%M%S')}_"


@pytest.fixture(scope="module")
def neo4j_client():
    from app.utils.neo4j_client import Neo4jClient

    Neo4jClient.reset_instance()
    client = Neo4jClient()
    yield client
    client.close()
    Neo4jClient.reset_instance()


@pytest.fixture(scope="module")
def zep_adapter():
    from app.adapters.zep_graph_adapter import ZepGraphAdapter

    adapter = ZepGraphAdapter()
    yield adapter


@pytest.fixture(scope="function")
def unique_graph_id():
    return f"{TEST_PREFIX}{uuid.uuid4().hex[:8]}"


class TestNeo4jConnection:
    def test_connection_success(self, neo4j_client):
        require_neo4j()
        result = neo4j_client.test_connection()
        assert result is True, "Neo4j 连接测试应该成功"

    def test_connection_with_session(self, neo4j_client):
        require_neo4j()
        with neo4j_client._get_session() as session:
            result = session.run("RETURN 'hello' as message")
            record = result.single()
            assert record is not None
            assert record["message"] == "hello"


class TestGraphLifecycle:
    def test_create_graph(self, zep_adapter, unique_graph_id):
        require_neo4j()
        graph_id = zep_adapter.graph.create(
            graph_id=unique_graph_id,
            name="Test Graph",
            description="Integration test graph",
        )

        assert graph_id == unique_graph_id

        neo4j = zep_adapter._get_neo4j()
        query = """
            MATCH (meta:GraphMetadata {graph_id: $graph_id})
            RETURN meta.name as name, meta.description as description
        """
        records = neo4j._execute_with_retry(query, {"graph_id": unique_graph_id})

        assert len(records) == 1
        assert records[0]["name"] == "Test Graph"
        assert records[0]["description"] == "Integration test graph"

        zep_adapter.graph.delete(unique_graph_id)

    def test_full_lifecycle(self, zep_adapter, unique_graph_id):
        require_neo4j()
        graph_id = zep_adapter.graph.create(
            graph_id=unique_graph_id,
            name="Lifecycle Test Graph",
        )
        assert graph_id == unique_graph_id

        try:
            neo4j = zep_adapter._get_neo4j()

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

            nodes = zep_adapter.node.get_by_graph_id(graph_id=unique_graph_id, limit=10)
            assert len(nodes) == 2, f"应该有2个节点，实际有 {len(nodes)} 个"
            node_names = {n.name for n in nodes}
            assert "Alice" in node_names
            assert "Google" in node_names

            edges = zep_adapter.edge.get_by_graph_id(graph_id=unique_graph_id, limit=10)
            assert len(edges) == 1, f"应该有1条边，实际有 {len(edges)} 条"
            assert edges[0].name == "WORKS_AT"
            assert edges[0].fact == "Alice works at Google"

        finally:
            zep_adapter.graph.delete(unique_graph_id)

            verify_query = """
                MATCH (n {graph_id: $graph_id})
                RETURN count(n) as count
            """
            records = neo4j._execute_with_retry(
                verify_query, {"graph_id": unique_graph_id}
            )
            assert records[0]["count"] == 0, "图谱删除后应该没有残留节点"


class TestPagination:
    def test_node_pagination(self, neo4j_client, unique_graph_id):
        require_neo4j()
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
            page1 = neo4j_client.fetch_all_nodes(
                graph_id=unique_graph_id, limit=10, page_size=5
            )
            assert len(page1) == 10, f"第一页应该有10个节点，实际有 {len(page1)} 个"

            cursor = page1[-1].uuid
            page2 = neo4j_client.fetch_all_nodes(
                graph_id=unique_graph_id, limit=10, cursor=cursor, page_size=5
            )
            assert len(page2) == 10, f"第二页应该有10个节点，实际有 {len(page2)} 个"

            page1_uuids = {n.uuid for n in page1}
            page2_uuids = {n.uuid for n in page2}
            assert len(page1_uuids & page2_uuids) == 0, "分页结果不应该有重复"

            all_nodes = neo4j_client.fetch_all_nodes(
                graph_id=unique_graph_id, limit=100, page_size=10
            )
            assert len(all_nodes) == 25, (
                f"总共应该有25个节点，实际有 {len(all_nodes)} 个"
            )

        finally:
            cleanup_query = """
                MATCH (n:Entity {graph_id: $graph_id})
                DETACH DELETE n
            """
            neo4j_client._execute_with_retry(
                cleanup_query, {"graph_id": unique_graph_id}
            )

    def test_edge_pagination(self, neo4j_client, unique_graph_id):
        require_neo4j()
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
            page1 = neo4j_client.fetch_all_edges(
                graph_id=unique_graph_id, limit=10, page_size=5
            )
            assert len(page1) == 10, f"第一页应该有10条边，实际有 {len(page1)} 条"

            cursor = page1[-1].uuid
            page2 = neo4j_client.fetch_all_edges(
                graph_id=unique_graph_id, limit=10, cursor=cursor, page_size=5
            )
            assert len(page2) == 10, f"第二页应该有10条边，实际有 {len(page2)} 条"

            page1_uuids = {e.uuid for e in page1}
            page2_uuids = {e.uuid for e in page2}
            assert len(page1_uuids & page2_uuids) == 0, "分页结果不应该有重复"

            all_edges = neo4j_client.fetch_all_edges(
                graph_id=unique_graph_id, limit=100, page_size=10
            )
            assert len(all_edges) == 20, f"总共应该有20条边，实际有 {len(all_edges)} 条"

        finally:
            cleanup_query = """
                MATCH (n:Entity {graph_id: $graph_id})
                DETACH DELETE n
            """
            neo4j_client._execute_with_retry(
                cleanup_query, {"graph_id": unique_graph_id}
            )

    def test_empty_pagination(self, neo4j_client, unique_graph_id):
        require_neo4j()
        nodes = neo4j_client.fetch_all_nodes(graph_id=unique_graph_id, limit=10)
        assert nodes == [], "空图谱应该返回空节点列表"

        edges = neo4j_client.fetch_all_edges(graph_id=unique_graph_id, limit=10)
        assert edges == [], "空图谱应该返回空边列表"


class TestEntityEdges:
    def test_get_entity_edges(self, neo4j_client, unique_graph_id):
        require_neo4j()
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
            edges = neo4j_client.get_entity_edges(center_uuid)

            assert len(edges) == 2, f"中心节点应该有2条边，实际有 {len(edges)} 条"
            edge_names = {e["name"] for e in edges}
            assert "KNOWS" in edge_names
            assert "WORKS_WITH" in edge_names

            for edge in edges:
                if edge["name"] == "KNOWS":
                    assert edge["source_uuid"] == center_uuid
                    assert edge["related_name"] == "Node1"
                elif edge["name"] == "WORKS_WITH":
                    assert edge["target_uuid"] == center_uuid
                    assert edge["related_name"] == "Node2"

        finally:
            cleanup_query = """
                MATCH (n:Entity {graph_id: $graph_id})
                DETACH DELETE n
            """
            neo4j_client._execute_with_retry(
                cleanup_query, {"graph_id": unique_graph_id}
            )


class TestDataIsolation:
    def test_graph_isolation(self, neo4j_client, unique_graph_id):
        require_neo4j()
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
            nodes1 = neo4j_client.fetch_all_nodes(graph_id=graph1_id, limit=10)
            nodes2 = neo4j_client.fetch_all_nodes(graph_id=graph2_id, limit=10)

            assert len(nodes1) == 1
            assert len(nodes2) == 1
            assert nodes1[0].name == "Graph1Node"
            assert nodes2[0].name == "Graph2Node"

        finally:
            for gid in [graph1_id, graph2_id]:
                cleanup_query = """
                    MATCH (n:Entity {graph_id: $graph_id})
                    DETACH DELETE n
                """
                neo4j_client._execute_with_retry(cleanup_query, {"graph_id": gid})


class TestMem0Initialization:
    def test_memory_instance_initialization(self):
        require_neo4j()
        require_embedding()
        from app.adapters.mem0_client import get_memory_instance, reset_memory_instance

        try:
            memory = get_memory_instance()
            assert memory is not None
            assert hasattr(memory, "add")
            assert hasattr(memory, "search")
        finally:
            reset_memory_instance()

    def test_memory_instance_singleton(self):
        require_neo4j()
        require_embedding()
        from app.adapters.mem0_client import get_memory_instance, reset_memory_instance

        try:
            memory1 = get_memory_instance()
            memory2 = get_memory_instance()
            assert memory1 is memory2
        finally:
            reset_memory_instance()


class TestMem0GraphOperations:
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        self.test_graph_ids = []
        yield
        from app.adapters.mem0_client import reset_memory_instance
        from app.utils.neo4j_client import Neo4jClient

        reset_memory_instance()
        neo4j = Neo4jClient()
        for graph_id in self.test_graph_ids:
            try:
                cleanup_query = """
                    MATCH (n {graph_id: $graph_id})
                    DETACH DELETE n
                """
                neo4j._execute_with_retry(cleanup_query, {"graph_id": graph_id})
            except Exception:
                pass

    def test_zep_adapter_create_graph(self, zep_adapter, unique_graph_id):
        require_mem0_full()
        self.test_graph_ids.append(unique_graph_id)

        graph_id = zep_adapter.graph.create(
            graph_id=unique_graph_id,
            name="Test Graph for mem0",
            description="Integration test for mem0 operations",
        )

        assert graph_id == unique_graph_id

        neo4j = zep_adapter._get_neo4j()
        query = """
            MATCH (meta:GraphMetadata {graph_id: $graph_id})
            RETURN meta.name as name
        """
        records = neo4j._execute_with_retry(query, {"graph_id": unique_graph_id})
        assert len(records) == 1
        assert records[0]["name"] == "Test Graph for mem0"

        zep_adapter.graph.delete(unique_graph_id)
        self.test_graph_ids.remove(unique_graph_id)

    def test_zep_adapter_add_and_search(self, zep_adapter, unique_graph_id):
        require_mem0_full()
        self.test_graph_ids.append(unique_graph_id)

        zep_adapter.graph.create(
            graph_id=unique_graph_id,
            name="Test Graph for Search",
        )

        try:
            result = zep_adapter.graph.add(
                graph_id=unique_graph_id,
                type="text",
                data="Alice works at Google as a software engineer. She lives in San Francisco.",
            )

            assert "uuid" in result
            assert result["uuid"] is not None

            import time

            time.sleep(2)

            search_result = zep_adapter.graph.search(
                graph_id=unique_graph_id,
                query="Where does Alice work?",
                limit=5,
            )

            assert search_result is not None
            assert hasattr(search_result, "facts")

        finally:
            zep_adapter.graph.delete(unique_graph_id)
            self.test_graph_ids.remove(unique_graph_id)

    def test_zep_adapter_delete_graph(self, zep_adapter, unique_graph_id):
        require_mem0_full()
        self.test_graph_ids.append(unique_graph_id)

        zep_adapter.graph.create(
            graph_id=unique_graph_id,
            name="Test Graph for Deletion",
        )

        neo4j = zep_adapter._get_neo4j()
        check_query = """
            MATCH (meta:GraphMetadata {graph_id: $graph_id})
            RETURN count(meta) as count
        """
        records = neo4j._execute_with_retry(check_query, {"graph_id": unique_graph_id})
        assert records[0]["count"] >= 1

        zep_adapter.graph.delete(unique_graph_id)

        records = neo4j._execute_with_retry(check_query, {"graph_id": unique_graph_id})
        assert records[0]["count"] == 0

        self.test_graph_ids.remove(unique_graph_id)


class TestEmbeddingServiceIntegration:
    def test_embedding_service_responds(self):
        require_embedding()
        import requests

        base_url = os.environ.get("EMBEDDING_BASE_URL", "http://127.0.0.1:8001/v1")

        response = requests.get(f"{base_url.rstrip('/')}/models", timeout=10)
        assert response.status_code == 200

        data = response.json()
        assert "data" in data or "models" in data or isinstance(data, list)

    def test_embedding_generation(self):
        require_embedding()
        import requests

        base_url = os.environ.get("EMBEDDING_BASE_URL", "http://127.0.0.1:8001/v1")
        model = os.environ.get("EMBEDDING_MODEL", "/home/ying/Qwen3-Embedding-4B")

        response = requests.post(
            f"{base_url.rstrip('/')}/embeddings",
            json={
                "input": "This is a test sentence for embedding.",
                "model": model,
            },
            headers={"Content-Type": "application/json"},
            timeout=30,
        )

        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert len(data["data"]) > 0
        assert "embedding" in data["data"][0]
        assert len(data["data"][0]["embedding"]) > 0
