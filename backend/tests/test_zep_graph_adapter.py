"""
ZepGraphAdapter 单元测试

测试 ZepGraphAdapter 的核心方法，使用 unittest.mock 模拟 Neo4j 和 mem0 依赖。
"""

import sys
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def mock_config():
    """Mock Config for all tests."""
    mock_cfg = MagicMock()
    mock_cfg.NEO4J_URL = "bolt://localhost:7687"
    mock_cfg.NEO4J_USERNAME = "neo4j"
    mock_cfg.NEO4J_PASSWORD = "test_password"
    mock_cfg.LLM_API_KEY = "test_key"
    mock_cfg.LLM_MODEL_NAME = "test_model"
    mock_cfg.LLM_BASE_URL = "http://test.url"
    mock_cfg.EMBEDDING_MODEL = "test_embedding"
    mock_cfg.EMBEDDING_BASE_URL = "http://test.url"
    mock_cfg.EMBEDDING_API_KEY = "test_embedding_key"
    mock_cfg.USE_MEM0 = True
    sys.modules["backend.app.config"] = mock_cfg
    sys.modules["backend"] = MagicMock()
    sys.modules["backend.app"] = MagicMock()
    yield mock_cfg


@pytest.fixture
def mock_neo4j_client():
    """Create mock Neo4j client."""
    mock_client = MagicMock()
    mock_client._execute_with_retry.return_value = []
    mock_client.fetch_all_nodes.return_value = []
    mock_client.fetch_all_edges.return_value = []
    return mock_client


@pytest.fixture
def mock_memory():
    """Create mock mem0 Memory instance."""
    mock_mem = MagicMock()
    mock_mem.add.return_value = [{"id": "test-uuid"}]
    mock_mem.search.return_value = []
    return mock_mem


class TestZepGraphAdapterInit:
    def test_init_without_api_key(self, mock_config):
        from app.adapters.zep_graph_adapter import ZepGraphAdapter

        adapter = ZepGraphAdapter()
        assert adapter._api_key is None
        assert adapter._memory is None
        assert adapter._neo4j is None
        assert hasattr(adapter, "graph")
        assert hasattr(adapter, "node")
        assert hasattr(adapter, "edge")
        assert hasattr(adapter, "episode")

    def test_init_with_api_key(self, mock_config):
        from app.adapters.zep_graph_adapter import ZepGraphAdapter

        adapter = ZepGraphAdapter(api_key="test-key")
        assert adapter._api_key == "test-key"


class TestGraphCreate:
    @patch("app.adapters.zep_graph_adapter.get_neo4j_client")
    def test_create_with_auto_graph_id(
        self, mock_get_neo4j, mock_config, mock_neo4j_client
    ):
        mock_neo4j_client._execute_with_retry.return_value = [
            {"graph_id": "auto_generated_id"}
        ]
        mock_get_neo4j.return_value = mock_neo4j_client

        from app.adapters.zep_graph_adapter import ZepGraphAdapter

        adapter = ZepGraphAdapter()
        result = adapter.graph.create(name="Test Graph", description="Test Description")

        assert result.startswith("mirofish_")
        mock_neo4j_client._execute_with_retry.assert_called_once()
        call_args = mock_neo4j_client._execute_with_retry.call_args
        assert call_args[0][1]["name"] == "Test Graph"
        assert call_args[0][1]["description"] == "Test Description"

    @patch("app.adapters.zep_graph_adapter.get_neo4j_client")
    def test_create_with_custom_graph_id(
        self, mock_get_neo4j, mock_config, mock_neo4j_client
    ):
        mock_neo4j_client._execute_with_retry.return_value = [{"graph_id": "custom_id"}]
        mock_get_neo4j.return_value = mock_neo4j_client

        from app.adapters.zep_graph_adapter import ZepGraphAdapter

        adapter = ZepGraphAdapter()
        result = adapter.graph.create(
            graph_id="custom_id", name="Test Graph", description="Test Description"
        )

        assert result == "custom_id"
        call_args = mock_neo4j_client._execute_with_retry.call_args
        assert call_args[0][1]["graph_id"] == "custom_id"

    @patch("app.adapters.zep_graph_adapter.get_neo4j_client")
    def test_create_raises_on_neo4j_error(
        self, mock_get_neo4j, mock_config, mock_neo4j_client
    ):
        mock_neo4j_client._execute_with_retry.side_effect = Exception(
            "Neo4j connection error"
        )
        mock_get_neo4j.return_value = mock_neo4j_client

        from app.adapters.zep_graph_adapter import ZepGraphAdapter

        adapter = ZepGraphAdapter()
        with pytest.raises(Exception, match="Neo4j connection error"):
            adapter.graph.create(name="Test Graph")


class TestGraphDelete:
    @patch("app.adapters.zep_graph_adapter.get_neo4j_client")
    def test_delete_calls_all_queries(
        self, mock_get_neo4j, mock_config, mock_neo4j_client
    ):
        mock_get_neo4j.return_value = mock_neo4j_client

        from app.adapters.zep_graph_adapter import ZepGraphAdapter

        adapter = ZepGraphAdapter()
        adapter.graph.delete(graph_id="test_graph_id")

        assert mock_neo4j_client._execute_with_retry.call_count == 3

        call_args_list = mock_neo4j_client._execute_with_retry.call_args_list
        for call_args in call_args_list:
            assert call_args[0][1]["graph_id"] == "test_graph_id"

    @patch("app.adapters.zep_graph_adapter.get_neo4j_client")
    def test_delete_raises_on_error(
        self, mock_get_neo4j, mock_config, mock_neo4j_client
    ):
        mock_neo4j_client._execute_with_retry.side_effect = Exception("Delete failed")
        mock_get_neo4j.return_value = mock_neo4j_client

        from app.adapters.zep_graph_adapter import ZepGraphAdapter

        adapter = ZepGraphAdapter()
        with pytest.raises(Exception, match="Delete failed"):
            adapter.graph.delete(graph_id="test_graph_id")


class TestNodeGetByGraphId:
    @patch("app.adapters.zep_graph_adapter.get_neo4j_client")
    def test_get_by_graph_id_returns_nodes(
        self, mock_get_neo4j, mock_config, mock_neo4j_client
    ):
        from app.utils.neo4j_client import EntityNode
        from app.adapters.zep_graph_adapter import ZepGraphAdapter

        expected_nodes = [
            EntityNode(
                uuid="node-1",
                name="Alice",
                labels=["Entity", "Person"],
                summary="A person",
                attributes={"age": 30},
            ),
            EntityNode(
                uuid="node-2",
                name="Bob",
                labels=["Entity", "Person"],
                summary="Another person",
                attributes={"age": 25},
            ),
        ]
        mock_neo4j_client.fetch_all_nodes.return_value = expected_nodes
        mock_get_neo4j.return_value = mock_neo4j_client

        adapter = ZepGraphAdapter()
        result = adapter.node.get_by_graph_id(graph_id="test_graph", limit=10)

        assert len(result) == 2
        assert result[0].uuid == "node-1"
        assert result[0].name == "Alice"
        assert result[1].uuid == "node-2"
        assert result[1].name == "Bob"
        mock_neo4j_client.fetch_all_nodes.assert_called_once_with(
            graph_id="test_graph", limit=10, cursor=None
        )

    @patch("app.adapters.zep_graph_adapter.get_neo4j_client")
    def test_get_by_graph_id_with_cursor(
        self, mock_get_neo4j, mock_config, mock_neo4j_client
    ):
        from app.adapters.zep_graph_adapter import ZepGraphAdapter

        mock_neo4j_client.fetch_all_nodes.return_value = []
        mock_get_neo4j.return_value = mock_neo4j_client

        adapter = ZepGraphAdapter()
        adapter.node.get_by_graph_id(
            graph_id="test_graph", limit=50, uuid_cursor="cursor-123"
        )

        mock_neo4j_client.fetch_all_nodes.assert_called_once_with(
            graph_id="test_graph", limit=50, cursor="cursor-123"
        )

    @patch("app.adapters.zep_graph_adapter.get_neo4j_client")
    def test_get_by_graph_id_returns_empty_on_error(
        self, mock_get_neo4j, mock_config, mock_neo4j_client
    ):
        from app.adapters.zep_graph_adapter import ZepGraphAdapter

        mock_neo4j_client.fetch_all_nodes.return_value = []
        mock_get_neo4j.return_value = mock_neo4j_client

        adapter = ZepGraphAdapter()
        result = adapter.node.get_by_graph_id(graph_id="nonexistent")

        assert result == []


class TestEdgeGetByGraphId:
    @patch("app.adapters.zep_graph_adapter.get_neo4j_client")
    def test_get_by_graph_id_returns_edges(
        self, mock_get_neo4j, mock_config, mock_neo4j_client
    ):
        from app.utils.neo4j_client import EdgeInfo
        from app.adapters.zep_graph_adapter import ZepGraphAdapter

        expected_edges = [
            EdgeInfo(
                uuid="edge-1",
                name="WORKS_AT",
                fact="Alice works at Google",
                source_node_uuid="node-1",
                target_node_uuid="node-3",
                source_node_name="Alice",
                target_node_name="Google",
            ),
            EdgeInfo(
                uuid="edge-2",
                name="KNOWS",
                fact="Alice knows Bob",
                source_node_uuid="node-1",
                target_node_uuid="node-2",
                source_node_name="Alice",
                target_node_name="Bob",
            ),
        ]
        mock_neo4j_client.fetch_all_edges.return_value = expected_edges
        mock_get_neo4j.return_value = mock_neo4j_client

        adapter = ZepGraphAdapter()
        result = adapter.edge.get_by_graph_id(graph_id="test_graph", limit=20)

        assert len(result) == 2
        assert result[0].uuid == "edge-1"
        assert result[0].name == "WORKS_AT"
        assert result[1].uuid == "edge-2"
        assert result[1].name == "KNOWS"
        mock_neo4j_client.fetch_all_edges.assert_called_once_with(
            graph_id="test_graph", limit=20, cursor=None
        )

    @patch("app.adapters.zep_graph_adapter.get_neo4j_client")
    def test_get_by_graph_id_with_cursor(
        self, mock_get_neo4j, mock_config, mock_neo4j_client
    ):
        from app.adapters.zep_graph_adapter import ZepGraphAdapter

        mock_neo4j_client.fetch_all_edges.return_value = []
        mock_get_neo4j.return_value = mock_neo4j_client

        adapter = ZepGraphAdapter()
        adapter.edge.get_by_graph_id(
            graph_id="test_graph", limit=100, uuid_cursor="edge-cursor-456"
        )

        mock_neo4j_client.fetch_all_edges.assert_called_once_with(
            graph_id="test_graph", limit=100, cursor="edge-cursor-456"
        )

    @patch("app.adapters.zep_graph_adapter.get_neo4j_client")
    def test_get_by_graph_id_returns_empty_on_no_edges(
        self, mock_get_neo4j, mock_config, mock_neo4j_client
    ):
        from app.adapters.zep_graph_adapter import ZepGraphAdapter

        mock_neo4j_client.fetch_all_edges.return_value = []
        mock_get_neo4j.return_value = mock_neo4j_client

        adapter = ZepGraphAdapter()
        result = adapter.edge.get_by_graph_id(graph_id="empty_graph")

        assert result == []


class TestGraphAdd:
    @patch("app.adapters.zep_graph_adapter.get_memory_instance")
    def test_add_returns_uuid(self, mock_get_memory, mock_config, mock_memory):
        mock_memory.add.return_value = [{"id": "memory-uuid-123"}]
        mock_get_memory.return_value = mock_memory

        from app.adapters.zep_graph_adapter import ZepGraphAdapter

        adapter = ZepGraphAdapter()
        result = adapter.graph.add(
            graph_id="test_graph", type="text", data="Alice works at Google"
        )

        assert "uuid" in result
        assert result["uuid"] == "memory-uuid-123"
        mock_memory.add.assert_called_once_with(
            [{"role": "user", "content": "Alice works at Google"}], user_id="test_graph"
        )

    @patch("app.adapters.zep_graph_adapter.get_memory_instance")
    def test_add_generates_uuid_if_missing(
        self, mock_get_memory, mock_config, mock_memory
    ):
        mock_memory.add.return_value = [{}]
        mock_get_memory.return_value = mock_memory

        from app.adapters.zep_graph_adapter import ZepGraphAdapter

        adapter = ZepGraphAdapter()
        result = adapter.graph.add(graph_id="test_graph", type="text", data="Some data")

        assert "uuid" in result
        assert len(result["uuid"]) > 0

    @patch("app.adapters.zep_graph_adapter.get_memory_instance")
    def test_add_raises_on_memory_error(
        self, mock_get_memory, mock_config, mock_memory
    ):
        mock_memory.add.side_effect = Exception("Memory error")
        mock_get_memory.return_value = mock_memory

        from app.adapters.zep_graph_adapter import ZepGraphAdapter

        adapter = ZepGraphAdapter()
        with pytest.raises(Exception, match="Memory error"):
            adapter.graph.add(graph_id="test_graph", data="data")


class TestGraphSearch:
    @patch("app.adapters.zep_graph_adapter.get_neo4j_client")
    @patch("app.adapters.zep_graph_adapter.get_memory_instance")
    def test_search_returns_facts(
        self,
        mock_get_memory,
        mock_get_neo4j,
        mock_config,
        mock_neo4j_client,
        mock_memory,
    ):
        from app.adapters.zep_graph_adapter import ZepGraphAdapter, SearchResult

        mock_memory.search.return_value = [
            {"memory": "Alice works at Google", "score": 0.9},
            {"memory": "Bob knows Alice", "score": 0.8},
        ]
        mock_get_memory.return_value = mock_memory
        mock_get_neo4j.return_value = mock_neo4j_client

        adapter = ZepGraphAdapter()
        result = adapter.graph.search(
            graph_id="test_graph", query="Where does Alice work?", limit=10
        )

        assert isinstance(result, SearchResult)
        assert len(result.facts) == 2
        assert result.facts[0]["fact"] == "Alice works at Google"
        assert result.facts[0]["score"] == 0.9

    @patch("app.adapters.zep_graph_adapter.get_neo4j_client")
    @patch("app.adapters.zep_graph_adapter.get_memory_instance")
    def test_search_empty_result(
        self,
        mock_get_memory,
        mock_get_neo4j,
        mock_config,
        mock_neo4j_client,
        mock_memory,
    ):
        from app.adapters.zep_graph_adapter import ZepGraphAdapter, SearchResult

        mock_get_memory.return_value = mock_memory
        mock_get_neo4j.return_value = mock_neo4j_client

        adapter = ZepGraphAdapter()
        result = adapter.graph.search(graph_id="test_graph", query="nonexistent query")

        assert isinstance(result, SearchResult)
        assert result.facts == []
        assert result.edges == []
        assert result.nodes == []

    @patch("app.adapters.zep_graph_adapter.get_neo4j_client")
    @patch("app.adapters.zep_graph_adapter.get_memory_instance")
    def test_search_handles_memory_error(
        self,
        mock_get_memory,
        mock_get_neo4j,
        mock_config,
        mock_neo4j_client,
        mock_memory,
    ):
        from app.adapters.zep_graph_adapter import ZepGraphAdapter, SearchResult

        mock_memory.search.side_effect = Exception("Search failed")
        mock_get_memory.return_value = mock_memory
        mock_get_neo4j.return_value = mock_neo4j_client

        adapter = ZepGraphAdapter()
        result = adapter.graph.search(graph_id="test_graph", query="test query")

        assert isinstance(result, SearchResult)
        assert result.facts == []
        assert result.edges == []
        assert result.nodes == []


class TestNodeGet:
    @patch("app.adapters.zep_graph_adapter.get_neo4j_client")
    def test_get_returns_node(self, mock_get_neo4j, mock_config, mock_neo4j_client):
        from app.adapters.zep_graph_adapter import ZepGraphAdapter

        mock_node_data = MagicMock()
        mock_node_data._properties = {
            "uuid": "node-uuid-1",
            "name": "Alice",
            "summary": "A person",
        }
        mock_node_data._labels = ["Entity", "Person"]

        mock_neo4j_client._execute_with_retry.return_value = [{"n": mock_node_data}]
        mock_get_neo4j.return_value = mock_neo4j_client

        adapter = ZepGraphAdapter()
        result = adapter.node.get(uuid_="node-uuid-1")

        assert result is not None
        assert result.uuid == "node-uuid-1"
        assert result.name == "Alice"
        assert "Entity" in result.labels

    @patch("app.adapters.zep_graph_adapter.get_neo4j_client")
    def test_get_returns_none_when_not_found(
        self, mock_get_neo4j, mock_config, mock_neo4j_client
    ):
        from app.adapters.zep_graph_adapter import ZepGraphAdapter

        mock_neo4j_client._execute_with_retry.return_value = []
        mock_get_neo4j.return_value = mock_neo4j_client

        adapter = ZepGraphAdapter()
        result = adapter.node.get(uuid_="nonexistent-uuid")

        assert result is None


class TestEpisodeGet:
    def test_get_returns_processed_episode(self, mock_config):
        from app.adapters.zep_graph_adapter import ZepGraphAdapter, EpisodeInfo

        adapter = ZepGraphAdapter()
        result = adapter.episode.get(uuid_="episode-uuid-123")

        assert isinstance(result, EpisodeInfo)
        assert result.uuid_ == "episode-uuid-123"
        assert result.processed is True
        assert result.data == ""


class TestDataClasses:
    def test_episode_data(self, mock_config):
        from app.adapters.zep_graph_adapter import EpisodeData

        episode = EpisodeData(data="test content", type="text", reference_id="ref-1")
        assert episode.data == "test content"
        assert episode.type == "text"
        assert episode.reference_id == "ref-1"

    def test_search_result_defaults(self, mock_config):
        from app.adapters.zep_graph_adapter import SearchResult

        result = SearchResult()
        assert result.facts == []
        assert result.edges == []
        assert result.nodes == []

    def test_episode_info_defaults(self, mock_config):
        from app.adapters.zep_graph_adapter import EpisodeInfo

        info = EpisodeInfo(uuid_="test-uuid", data="test data")
        assert info.uuid_ == "test-uuid"
        assert info.data == "test data"
        assert info.processed is True


class TestCreateZepCompatibleClient:
    def test_creates_adapter_without_api_key(self, mock_config):
        from app.adapters.zep_graph_adapter import (
            create_zep_compatible_client,
            ZepGraphAdapter,
        )

        client = create_zep_compatible_client()
        assert isinstance(client, ZepGraphAdapter)
        assert client._api_key is None

    def test_creates_adapter_with_api_key(self, mock_config):
        from app.adapters.zep_graph_adapter import (
            create_zep_compatible_client,
            ZepGraphAdapter,
        )

        client = create_zep_compatible_client(api_key="test-key")
        assert isinstance(client, ZepGraphAdapter)
        assert client._api_key == "test-key"
