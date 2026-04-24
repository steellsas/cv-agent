import pytest
import json
from unittest.mock import MagicMock, patch
from agents.profile_coordinator import ProfileCoordinator


@pytest.fixture
def coordinator(config, mock_store, mock_llm):
    with patch("agents.profile_coordinator.VectorStore", return_value=mock_store):
        with patch("agents.profile_coordinator.get_llm", return_value=mock_llm):
            c = ProfileCoordinator(config)
            c.store = mock_store
            c.llm = mock_llm
            return c


class TestProfileCoordinator:

    def test_get_master_profile_returns_cached(self, coordinator, sample_master_profile):
        coordinator.master_profile = sample_master_profile
        result = coordinator.get_master_profile()
        assert result == sample_master_profile
        coordinator.store.get_all.assert_not_called()

    def test_get_master_profile_loads_from_qdrant(self, coordinator, mock_store, sample_master_profile):
        coordinator.master_profile = None
        mock_store.get_all.return_value = [
            {"text": json.dumps(sample_master_profile), "category": "master_profile",
             "source": ["synthesized"], "confidence": "high", "user_confirmed": True}
        ]
        result = coordinator.get_master_profile()
        assert result["full_name"] == "Andrius Plienius"

    def test_get_master_profile_returns_empty_if_none(self, coordinator, mock_store):
        coordinator.master_profile = None
        mock_store.get_all.return_value = []
        result = coordinator.get_master_profile()
        assert result == {}

    def test_gather_all_data_includes_categories(self, coordinator, mock_store):
        mock_store.get_all.return_value = [
            {"text": "Python Developer at DeVoro", "category": "work_experience",
             "source": ["linkedin"], "confidence": "high", "user_confirmed": True}
        ]
        result = coordinator._gather_all_data()
        assert "WORK_EXPERIENCE" in result
        assert "Python Developer at DeVoro" in result

    def test_get_strengths_for_job(self, coordinator, mock_llm, sample_master_profile):
        coordinator.master_profile = sample_master_profile
        job_match_data = {
            "match_score": 78,
            "strong_matches": [{"requirement": "Python", "evidence": "5+ years"}],
            "partial_matches": [],
            "gaps": [],
            "unique_value": "AI + business combo",
            "emphasis_for_cv": ["Python", "AI"]
        }
        mock_llm.invoke.return_value = MagicMock(
            content=json.dumps(job_match_data)
        )
        result = coordinator.get_strengths_for_job("We need a Python developer")
        assert result["match_score"] == 78
        assert len(result["strong_matches"]) == 1

    def test_get_strengths_returns_empty_if_no_profile(self, coordinator):
        coordinator.master_profile = None
        coordinator.store.get_all.return_value = []
        result = coordinator.get_strengths_for_job("Job posting text")
        assert result == {}

    def test_build_returns_empty_if_no_data(self, coordinator, mock_store):
        mock_store.get_profile_summary.return_value = {
            cat: {"count": 0, "sources": [], "confirmed": 0}
            for cat in ["work_experience", "project", "education",
                        "tech_skill", "soft_skill", "personality"]
        }
        result = coordinator.build()
        assert result == {}