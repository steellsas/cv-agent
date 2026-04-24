import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from memory.vector_store import VectorStore, ProfileBlock


class TestProfileBlock:
    """Tests for ProfileBlock Pydantic model"""

    def test_basic_creation(self):
        block = ProfileBlock(
            category="work_experience",
            text="Python Developer at DeVoro",
            source=["linkedin"],
            confidence="high"
        )
        assert block.category == "work_experience"
        assert block.text == "Python Developer at DeVoro"
        assert block.source == ["linkedin"]
        assert block.confidence == "high"
        assert block.user_confirmed == False

    def test_default_values(self):
        block = ProfileBlock(category="project", text="Some project")
        assert block.source == []
        assert block.confidence == "medium"
        assert block.metadata == {}
        assert block.user_confirmed == False

    def test_multiple_sources(self):
        block = ProfileBlock(
            category="tech_skill",
            text="Python programming",
            source=["linkedin", "github", "conversation"]
        )
        assert len(block.source) == 3
        assert "github" in block.source

    def test_user_confirmed(self):
        block = ProfileBlock(
            category="education",
            text="BSc at KTU",
            source=["linkedin"],
            user_confirmed=True
        )
        assert block.user_confirmed == True

    def test_metadata(self):
        block = ProfileBlock(
            category="work_experience",
            text="Developer at Company",
            source=["linkedin"],
            metadata={"company": "TestCorp", "role": "Developer"}
        )
        assert block.metadata["company"] == "TestCorp"
        assert block.metadata["role"] == "Developer"

    @pytest.mark.parametrize("confidence", ["low", "medium", "high"])
    def test_confidence_levels(self, confidence):
        block = ProfileBlock(
            category="project",
            text="Test project",
            confidence=confidence
        )
        assert block.confidence == confidence

    @pytest.mark.parametrize("category", [
        "work_experience", "project", "education",
        "tech_skill", "soft_skill", "personality", "other"
    ])
    def test_all_categories(self, category):
        block = ProfileBlock(category=category, text="Test")
        assert block.category == category


class TestVectorStore:
    """Tests for VectorStore with mocked Qdrant and embeddings"""

    @pytest.fixture
    def mock_qdrant(self):
        with patch("memory.vector_store.QdrantClient") as mock:
            client = MagicMock()
            mock.return_value = client

            # Mock collections
            collection = MagicMock()
            collection.name = "other_collection"
            client.get_collections.return_value = MagicMock(collections=[collection])
            yield client

    @pytest.fixture
    def mock_embeddings(self):
        with patch("memory.vector_store.OpenAIEmbeddings") as mock:
            embedder = MagicMock()
            embedder.embed_query.return_value = [0.1] * 1536
            mock.return_value = embedder
            yield embedder

    @pytest.fixture
    def store(self, config, mock_qdrant, mock_embeddings):
        return VectorStore(config)

    def test_creates_collection_if_missing(self, store, mock_qdrant):
        mock_qdrant.create_collection.assert_called_once()

    def test_skips_creation_if_exists(self, config, mock_embeddings):
        with patch("memory.vector_store.QdrantClient") as mock:
            client = MagicMock()
            mock.return_value = client
            collection = MagicMock()
            collection.name = "test_user_profile"
            client.get_collections.return_value = MagicMock(collections=[collection])
            store = VectorStore(config)
            client.create_collection.assert_not_called()

    def test_save_block(self, store, mock_qdrant, mock_embeddings):
        block = ProfileBlock(
            category="work_experience",
            text="Python Developer at DeVoro",
            source=["linkedin"],
            confidence="high"
        )
        store.save_block(block)
        mock_qdrant.upsert.assert_called_once()
        call_args = mock_qdrant.upsert.call_args
        point = call_args[1]["points"][0]
        assert point.payload["category"] == "work_experience"
        assert point.payload["text"] == "Python Developer at DeVoro"
        assert point.payload["source"] == ["linkedin"]

    def test_save_block_generates_uuid(self, store, mock_qdrant, mock_embeddings):
        block = ProfileBlock(category="project", text="Test project", source=["github"])
        store.save_block(block)
        point = mock_qdrant.upsert.call_args[1]["points"][0]
        assert len(point.id) == 36  # UUID format

    def test_save_block_uses_embedding(self, store, mock_qdrant, mock_embeddings):
        block = ProfileBlock(category="tech_skill", text="Python FastAPI")
        store.save_block(block)
        mock_embeddings.embed_query.assert_called_with("Python FastAPI")

    def test_search_calls_query_points(self, store, mock_qdrant, mock_embeddings):
        mock_qdrant.query_points.return_value = MagicMock(points=[])
        results = store.search("Python developer experience")
        mock_qdrant.query_points.assert_called_once()
        assert results == []

    def test_search_with_category_filter(self, store, mock_qdrant, mock_embeddings):
        mock_qdrant.query_points.return_value = MagicMock(points=[])
        store.search("Python", category="tech_skill")
        call_args = mock_qdrant.query_points.call_args
        assert call_args[1]["query_filter"] is not None

    def test_search_without_category_no_filter(self, store, mock_qdrant, mock_embeddings):
        mock_qdrant.query_points.return_value = MagicMock(points=[])
        store.search("Python")
        call_args = mock_qdrant.query_points.call_args
        assert call_args[1]["query_filter"] is None

    def test_get_all_returns_list(self, store, mock_qdrant, mock_embeddings):
        mock_qdrant.scroll.return_value = ([], None)
        result = store.get_all()
        assert isinstance(result, list)

    def test_get_profile_summary_all_categories(self, store, mock_qdrant, mock_embeddings):
        mock_qdrant.scroll.return_value = ([], None)
        summary = store.get_profile_summary()
        expected = ["work_experience", "project", "education",
                    "tech_skill", "soft_skill", "personality"]
        for cat in expected:
            assert cat in summary

    def test_clear_collection(self, store, mock_qdrant, mock_embeddings):
        store.clear_collection()
        mock_qdrant.delete_collection.assert_called_once()
        mock_qdrant.create_collection.assert_called()