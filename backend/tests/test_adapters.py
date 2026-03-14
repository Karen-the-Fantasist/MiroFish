"""
Unit tests for neo4j_client.py and mem0_client.py.

Tests use unittest.mock to simulate external dependencies (Neo4j driver, mem0 Memory).
No real database connections are required.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from dataclasses import fields

# Import modules under test
from app.utils.neo4j_client import (
    EntityNode,
    EdgeInfo,
    Neo4jClient,
    get_neo4j_client,
)
from app.adapters.mem0_client import (
    get_memory_instance,
    reset_memory_instance,
)


class TestEntityNode:
    """Test EntityNode dataclass."""

    def test_entity_node_required_fields(self):
        """Test EntityNode with only required fields."""
        node = EntityNode(
            uuid="test-uuid-123",
            name="Test Entity",
            labels=["Person", "Entity"],
            summary="A test entity",
            attributes={"age": 25, "city": "Beijing"},
        )

        assert node.uuid == "test-uuid-123"
        assert node.name == "Test Entity"
        assert node.labels == ["Person", "Entity"]
        assert node.summary == "A test entity"
        assert node.attributes == {"age": 25, "city": "Beijing"}
        assert node.related_edges == []
        assert node.related_nodes == []

    def test_entity_node_with_optional_fields(self):
        """Test EntityNode with optional related_edges and related_nodes."""
        node = EntityNode(
            uuid="test-uuid-456",
            name="Test Entity 2",
            labels=["Company"],
            summary="A company entity",
            attributes={"industry": "tech"},
            related_edges=[{"uuid": "edge-1", "name": "WORKS_AT"}],
            related_nodes=[{"uuid": "node-2", "name": "Company A"}],
        )

        assert node.related_edges == [{"uuid": "edge-1", "name": "WORKS_AT"}]
        assert node.related_nodes == [{"uuid": "node-2", "name": "Company A"}]

    def test_entity_node_field_types(self):
        """Test EntityNode field types are correct."""
        node = EntityNode(
            uuid="type-test-uuid",
            name="Type Test",
            labels=["Test"],
            summary="Type test",
            attributes={},
        )

        assert isinstance(node.uuid, str)
        assert isinstance(node.name, str)
        assert isinstance(node.labels, list)
        assert isinstance(node.summary, str)
        assert isinstance(node.attributes, dict)
        assert isinstance(node.related_edges, list)
        assert isinstance(node.related_nodes, list)

    def test_entity_node_empty_attributes(self):
        """Test EntityNode with empty attributes."""
        node = EntityNode(
            uuid="empty-attrs",
            name="Empty",
            labels=[],
            summary="",
            attributes={},
        )

        assert node.attributes == {}
        assert node.labels == []


class TestEdgeInfo:
    """Test EdgeInfo dataclass."""

    def test_edge_info_required_fields(self):
        """Test EdgeInfo with only required fields."""
        edge = EdgeInfo(
            uuid="edge-uuid-123",
            name="WORKS_AT",
            fact="John works at Microsoft",
            source_node_uuid="node-uuid-1",
            target_node_uuid="node-uuid-2",
        )

        assert edge.uuid == "edge-uuid-123"
        assert edge.name == "WORKS_AT"
        assert edge.fact == "John works at Microsoft"
        assert edge.source_node_uuid == "node-uuid-1"
        assert edge.target_node_uuid == "node-uuid-2"
        assert edge.source_node_name is None
        assert edge.target_node_name is None
        assert edge.created_at is None
        assert edge.valid_at is None
        assert edge.invalid_at is None
        assert edge.expired_at is None

    def test_edge_info_with_optional_fields(self):
        """Test EdgeInfo with all optional fields."""
        edge = EdgeInfo(
            uuid="edge-full",
            name="KNOWS",
            fact="Alice knows Bob",
            source_node_uuid="alice-uuid",
            target_node_uuid="bob-uuid",
            source_node_name="Alice",
            target_node_name="Bob",
            created_at="2024-01-01",
            valid_at="2024-01-01",
            invalid_at="2025-01-01",
            expired_at="2025-12-31",
        )

        assert edge.source_node_name == "Alice"
        assert edge.target_node_name == "Bob"
        assert edge.created_at == "2024-01-01"
        assert edge.valid_at == "2024-01-01"
        assert edge.invalid_at == "2025-01-01"
        assert edge.expired_at == "2025-12-31"

    def test_edge_info_field_types(self):
        """Test EdgeInfo field types are correct."""
        edge = EdgeInfo(
            uuid="type-test",
            name="TEST",
            fact="Test fact",
            source_node_uuid="src",
            target_node_uuid="tgt",
        )

        assert isinstance(edge.uuid, str)
        assert isinstance(edge.name, str)
        assert isinstance(edge.fact, str)
        assert isinstance(edge.source_node_uuid, str)
        assert isinstance(edge.target_node_uuid, str)
        # Optional fields should be None or str
        assert edge.source_node_name is None or isinstance(edge.source_node_name, str)


class TestNeo4jClient:
    """Test Neo4jClient class with mocked driver."""

    def setup_method(self):
        """Reset singleton before each test."""
        Neo4jClient.reset_instance()

    def teardown_method(self):
        """Reset singleton after each test."""
        Neo4jClient.reset_instance()

    @patch("app.utils.neo4j_client.GraphDatabase")
    @patch("app.utils.neo4j_client.Config")
    def test_singleton_pattern(self, mock_config, mock_graph_db):
        """Test that Neo4jClient implements singleton pattern correctly."""
        mock_config.NEO4J_URL = "bolt://localhost:7687"
        mock_config.NEO4J_USERNAME = "neo4j"
        mock_config.NEO4J_PASSWORD = "test_password"

        mock_driver = Mock()
        mock_graph_db.driver.return_value = mock_driver

        client1 = Neo4jClient()
        client2 = Neo4jClient()

        assert client1 is client2
        # Driver should only be created once
        assert mock_graph_db.driver.call_count == 1

    @patch("app.utils.neo4j_client.GraphDatabase")
    @patch("app.utils.neo4j_client.Config")
    def test_initialization_with_config(self, mock_config, mock_graph_db):
        """Test Neo4jClient initialization reads config correctly."""
        mock_config.NEO4J_URL = "bolt://test-server:7687"
        mock_config.NEO4J_USERNAME = "test_user"
        mock_config.NEO4J_PASSWORD = "test_pass"

        mock_driver = Mock()
        mock_graph_db.driver.return_value = mock_driver

        client = Neo4jClient()

        mock_graph_db.driver.assert_called_once_with(
            "bolt://test-server:7687",
            auth=("test_user", "test_pass"),
            max_connection_pool_size=50,
            connection_timeout=30.0,
        )

    @patch("app.utils.neo4j_client.GraphDatabase")
    @patch("app.utils.neo4j_client.Config")
    def test_context_manager(self, mock_config, mock_graph_db):
        """Test Neo4jClient works as context manager."""
        mock_config.NEO4J_URL = "bolt://localhost:7687"
        mock_config.NEO4J_USERNAME = "neo4j"
        mock_config.NEO4J_PASSWORD = "password"

        mock_driver = Mock()
        mock_graph_db.driver.return_value = mock_driver

        with Neo4jClient() as client:
            assert client is not None

    @patch("app.utils.neo4j_client.GraphDatabase")
    @patch("app.utils.neo4j_client.Config")
    def test_test_connection_success(self, mock_config, mock_graph_db):
        """Test test_connection returns True on successful connection."""
        mock_config.NEO4J_URL = "bolt://localhost:7687"
        mock_config.NEO4J_USERNAME = "neo4j"
        mock_config.NEO4J_PASSWORD = "password"

        mock_driver = Mock()
        mock_session = Mock()
        mock_result = Mock()
        mock_result.single.return_value = {"test": 1}
        mock_session.run.return_value = mock_result
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=False)
        mock_driver.session.return_value = mock_session
        mock_graph_db.driver.return_value = mock_driver

        client = Neo4jClient()
        result = client.test_connection()

        assert result is True
        mock_session.run.assert_called_once_with("RETURN 1 as test")

    @patch("app.utils.neo4j_client.GraphDatabase")
    @patch("app.utils.neo4j_client.Config")
    def test_test_connection_failure(self, mock_config, mock_graph_db):
        """Test test_connection returns False on connection failure."""
        mock_config.NEO4J_URL = "bolt://localhost:7687"
        mock_config.NEO4J_USERNAME = "neo4j"
        mock_config.NEO4J_PASSWORD = "password"

        mock_driver = Mock()
        mock_driver.session.side_effect = Exception("Connection failed")
        mock_graph_db.driver.return_value = mock_driver

        client = Neo4jClient()
        result = client.test_connection()

        assert result is False

    @patch("app.utils.neo4j_client.GraphDatabase")
    @patch("app.utils.neo4j_client.Config")
    def test_close_connection(self, mock_config, mock_graph_db):
        """Test close() properly closes the driver."""
        mock_config.NEO4J_URL = "bolt://localhost:7687"
        mock_config.NEO4J_USERNAME = "neo4j"
        mock_config.NEO4J_PASSWORD = "password"

        mock_driver = Mock()
        mock_graph_db.driver.return_value = mock_driver

        client = Neo4jClient()
        client.close()

        mock_driver.close.assert_called_once()

    @patch("app.utils.neo4j_client.GraphDatabase")
    @patch("app.utils.neo4j_client.Config")
    def test_reset_instance(self, mock_config, mock_graph_db):
        """Test reset_instance clears the singleton."""
        mock_config.NEO4J_URL = "bolt://localhost:7687"
        mock_config.NEO4J_USERNAME = "neo4j"
        mock_config.NEO4J_PASSWORD = "password"

        mock_driver = Mock()
        mock_graph_db.driver.return_value = mock_driver

        client1 = Neo4jClient()
        Neo4jClient.reset_instance()

        # After reset, a new instance should be created
        mock_graph_db.driver.reset_mock()
        client2 = Neo4jClient()

        # Driver should be created again after reset
        assert mock_graph_db.driver.call_count == 1

    @patch("app.utils.neo4j_client.GraphDatabase")
    @patch("app.utils.neo4j_client.Config")
    def test_get_neo4j_client_function(self, mock_config, mock_graph_db):
        """Test get_neo4j_client returns singleton instance."""
        mock_config.NEO4J_URL = "bolt://localhost:7687"
        mock_config.NEO4J_USERNAME = "neo4j"
        mock_config.NEO4J_PASSWORD = "password"

        mock_driver = Mock()
        mock_graph_db.driver.return_value = mock_driver

        client1 = get_neo4j_client()
        client2 = get_neo4j_client()

        assert client1 is client2


class TestMem0Client:
    """Test mem0_client module with mocked Memory."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_memory_instance()

    def teardown_method(self):
        """Reset singleton after each test."""
        reset_memory_instance()

    @patch("app.adapters.mem0_client._check_embedding_service", return_value=True)
    @patch("app.adapters.mem0_client.Memory")
    @patch("app.adapters.mem0_client.Config")
    def test_get_memory_instance_creates_instance(
        self, mock_config, mock_memory, mock_check
    ):
        """Test get_memory_instance creates Memory instance with correct config."""
        mock_config.USE_MEM0 = True
        mock_config.LLM_API_KEY = "test-api-key"
        mock_config.NEO4J_PASSWORD = "test-password"
        mock_config.NEO4J_URL = "bolt://localhost:7687"
        mock_config.NEO4J_USERNAME = "neo4j"
        mock_config.LLM_MODEL_NAME = "gpt-4"
        mock_config.LLM_BASE_URL = "https://api.openai.com/v1"
        mock_config.EMBEDDING_MODEL = "text-embedding-ada-002"
        mock_config.EMBEDDING_BASE_URL = "https://api.openai.com/v1"
        mock_config.EMBEDDING_API_KEY = "test-embedding-key"

        mock_memory_instance = Mock()
        mock_memory.from_config.return_value = mock_memory_instance

        result = get_memory_instance()

        assert result is mock_memory_instance
        mock_memory.from_config.assert_called_once()

        # Verify config structure passed to Memory.from_config
        call_args = mock_memory.from_config.call_args
        config = call_args[0][0]

        assert config["graph_store"]["provider"] == "neo4j"
        assert config["graph_store"]["config"]["url"] == "bolt://localhost:7687"
        assert config["graph_store"]["config"]["username"] == "neo4j"
        assert config["graph_store"]["config"]["password"] == "test-password"
        assert config["llm"]["provider"] == "openai"
        assert config["llm"]["config"]["model"] == "gpt-4"
        assert config["embedder"]["provider"] == "openai"
        assert config["embedder"]["config"]["api_key"] == "test-embedding-key"
        assert config["vector_store"]["provider"] == "qdrant"

    @patch("app.adapters.mem0_client._check_embedding_service", return_value=True)
    @patch("app.adapters.mem0_client.Memory")
    @patch("app.adapters.mem0_client.Config")
    def test_get_memory_instance_singleton(self, mock_config, mock_memory, mock_check):
        """Test get_memory_instance returns same instance (singleton)."""
        mock_config.USE_MEM0 = True
        mock_config.LLM_API_KEY = "test-api-key"
        mock_config.NEO4J_PASSWORD = "test-password"
        mock_config.NEO4J_URL = "bolt://localhost:7687"
        mock_config.NEO4J_USERNAME = "neo4j"
        mock_config.LLM_MODEL_NAME = "gpt-4"
        mock_config.LLM_BASE_URL = "https://api.openai.com/v1"
        mock_config.EMBEDDING_MODEL = "text-embedding-ada-002"
        mock_config.EMBEDDING_BASE_URL = "https://api.openai.com/v1"
        mock_config.EMBEDDING_API_KEY = "test-embedding-key"

        mock_memory_instance = Mock()
        mock_memory.from_config.return_value = mock_memory_instance

        instance1 = get_memory_instance()
        instance2 = get_memory_instance()

        assert instance1 is instance2
        # Memory.from_config should only be called once
        mock_memory.from_config.assert_called_once()

    @patch("app.adapters.mem0_client.Memory")
    @patch("app.adapters.mem0_client.Config")
    def test_get_memory_instance_raises_when_disabled(self, mock_config, mock_memory):
        """Test get_memory_instance raises RuntimeError when USE_MEM0 is False."""
        mock_config.USE_MEM0 = False

        with pytest.raises(RuntimeError, match="USE_MEM0 未启用"):
            get_memory_instance()

    @patch("app.adapters.mem0_client.Memory")
    @patch("app.adapters.mem0_client.Config")
    def test_get_memory_instance_raises_without_api_key(self, mock_config, mock_memory):
        """Test get_memory_instance raises RuntimeError when LLM_API_KEY is missing."""
        mock_config.USE_MEM0 = True
        mock_config.LLM_API_KEY = None
        mock_config.NEO4J_PASSWORD = "test-password"

        with pytest.raises(RuntimeError, match="LLM_API_KEY 未配置"):
            get_memory_instance()

    @patch("app.adapters.mem0_client.Memory")
    @patch("app.adapters.mem0_client.Config")
    def test_get_memory_instance_raises_without_neo4j_password(
        self, mock_config, mock_memory
    ):
        """Test get_memory_instance raises RuntimeError when NEO4J_PASSWORD is missing."""
        mock_config.USE_MEM0 = True
        mock_config.LLM_API_KEY = "test-api-key"
        mock_config.NEO4J_PASSWORD = None

        with pytest.raises(RuntimeError, match="NEO4J_PASSWORD 未配置"):
            get_memory_instance()

    @patch("app.adapters.mem0_client._check_embedding_service", return_value=True)
    @patch("app.adapters.mem0_client.Memory")
    @patch("app.adapters.mem0_client.Config")
    def test_reset_memory_instance(self, mock_config, mock_memory, mock_check):
        """Test reset_memory_instance clears the singleton."""
        mock_config.USE_MEM0 = True
        mock_config.LLM_API_KEY = "test-api-key"
        mock_config.NEO4J_PASSWORD = "test-password"
        mock_config.NEO4J_URL = "bolt://localhost:7687"
        mock_config.NEO4J_USERNAME = "neo4j"
        mock_config.LLM_MODEL_NAME = "gpt-4"
        mock_config.LLM_BASE_URL = "https://api.openai.com/v1"
        mock_config.EMBEDDING_MODEL = "text-embedding-ada-002"
        mock_config.EMBEDDING_BASE_URL = "https://api.openai.com/v1"
        mock_config.EMBEDDING_API_KEY = "test-embedding-key"

        mock_memory_instance = Mock()
        mock_memory.from_config.return_value = mock_memory_instance

        # Create instance
        get_memory_instance()

        # Reset
        reset_memory_instance()

        # Create again - should create new instance
        mock_memory.from_config.reset_mock()
        get_memory_instance()

        mock_memory.from_config.assert_called_once()


class TestEmbeddingServiceValidation:
    """测试 Embedding 服务可用性验证。"""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_memory_instance()

    def teardown_method(self):
        """Reset singleton after each test."""
        reset_memory_instance()

    @patch("app.adapters.mem0_client.requests.get")
    def test_check_embedding_service_success(self, mock_get):
        """测试 Embedding 服务可用时返回 True。"""
        from app.adapters.mem0_client import _check_embedding_service

        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        result = _check_embedding_service("http://127.0.0.1:8001/v1")

        assert result is True
        mock_get.assert_called_once_with("http://127.0.0.1:8001/v1/models", timeout=5)

    @patch("app.adapters.mem0_client.requests.get")
    def test_check_embedding_service_failure_status(self, mock_get):
        """测试 Embedding 服务返回非 200 状态码时返回 False。"""
        from app.adapters.mem0_client import _check_embedding_service

        mock_response = Mock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        result = _check_embedding_service("http://127.0.0.1:8001/v1")

        assert result is False

    @patch("app.adapters.mem0_client.requests.get")
    def test_check_embedding_service_connection_error(self, mock_get):
        """测试 Embedding 服务连接失败时返回 False。"""
        from app.adapters.mem0_client import _check_embedding_service

        mock_get.side_effect = Exception("Connection refused")

        result = _check_embedding_service("http://127.0.0.1:8001/v1")

        assert result is False

    @patch("app.adapters.mem0_client.requests.get")
    def test_check_embedding_service_timeout(self, mock_get):
        """测试 Embedding 服务超时时返回 False。"""
        from app.adapters.mem0_client import _check_embedding_service

        import requests

        mock_get.side_effect = requests.Timeout("Connection timed out")

        result = _check_embedding_service("http://127.0.0.1:8001/v1")

        assert result is False

    @patch("app.adapters.mem0_client._check_embedding_service", return_value=False)
    @patch("app.adapters.mem0_client.Memory")
    @patch("app.adapters.mem0_client.Config")
    def test_get_memory_instance_raises_when_embedding_unavailable(
        self, mock_config, mock_memory, mock_check
    ):
        """测试 Embedding 服务不可用时抛出清晰的错误信息。"""
        mock_config.USE_MEM0 = True
        mock_config.LLM_API_KEY = "test-api-key"
        mock_config.NEO4J_PASSWORD = "test-password"
        mock_config.EMBEDDING_BASE_URL = "http://127.0.0.1:8001/v1"

        with pytest.raises(RuntimeError, match="Embedding 服务不可用"):
            get_memory_instance()

    @patch("app.adapters.mem0_client._check_embedding_service", return_value=False)
    @patch("app.adapters.mem0_client.Memory")
    @patch("app.adapters.mem0_client.Config")
    def test_get_memory_instance_embedding_error_contains_help(
        self, mock_config, mock_memory, mock_check
    ):
        """测试 Embedding 服务不可用时错误信息包含帮助提示。"""
        mock_config.USE_MEM0 = True
        mock_config.LLM_API_KEY = "test-api-key"
        mock_config.NEO4J_PASSWORD = "test-password"
        mock_config.EMBEDDING_BASE_URL = "http://127.0.0.1:8001/v1"

        with pytest.raises(RuntimeError) as exc_info:
            get_memory_instance()

        error_message = str(exc_info.value)
        # 错误信息应包含帮助提示
        assert "vllm" in error_message.lower() or "embedding" in error_message.lower()
        assert "启动" in error_message or "start" in error_message.lower()
