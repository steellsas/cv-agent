import pytest
from unittest.mock import MagicMock, patch, mock_open
from tools.linkedin_parser import LinkedInParser
from memory.vector_store import ProfileBlock


@pytest.fixture
def parser(config, mock_store):
    with patch("tools.linkedin_parser.VectorStore", return_value=mock_store):
        with patch("tools.linkedin_parser.get_llm") as mock_llm_fn:
            mock_llm_fn.return_value = MagicMock()
            p = LinkedInParser(config)
            p.store = mock_store
            return p


SAMPLE_LINKEDIN_DATA = {
    "full_name": "Andrius Plienius",
    "headline": "AI & ML Engineer | Python Backend Developer",
    "summary": "Experienced developer in AI and ML solutions",
    "work_experience": [
        {
            "company": "DeVoro",
            "role": "Python Developer",
            "period": "2019-2021",
            "description": "Built Django platform and API integrations"
        }
    ],
    "education": [
        {
            "institution": "KTU",
            "degree": "BSc Information Technology",
            "period": "2001-2005"
        }
    ],
    "skills": ["Python", "Django", "FastAPI", "Machine Learning"],
    "projects": [
        {
            "name": "ISP-AI-Agent",
            "description": "AI customer support bot"
        }
    ],
    "certifications": ["Python 3 Programming - University of Michigan"],
    "languages": ["Lithuanian", "English"]
}


class TestLinkedInParserSaveToStore:

    def test_saves_summary(self, parser, mock_store):
        parser._save_to_store(SAMPLE_LINKEDIN_DATA)
        calls = mock_store.save_block.call_args_list
        categories = [c[0][0].category for c in calls]
        assert "personality" in categories

    def test_saves_headline(self, parser, mock_store):
        parser._save_to_store(SAMPLE_LINKEDIN_DATA)
        calls = mock_store.save_block.call_args_list
        texts = [c[0][0].text for c in calls]
        assert any("AI & ML Engineer" in t for t in texts)

    def test_saves_work_experience(self, parser, mock_store):
        parser._save_to_store(SAMPLE_LINKEDIN_DATA)
        calls = mock_store.save_block.call_args_list
        categories = [c[0][0].category for c in calls]
        assert "work_experience" in categories

    def test_work_experience_source_is_linkedin(self, parser, mock_store):
        parser._save_to_store(SAMPLE_LINKEDIN_DATA)
        calls = mock_store.save_block.call_args_list
        exp_calls = [c for c in calls if c[0][0].category == "work_experience"]
        assert all("linkedin" in c[0][0].source for c in exp_calls)

    def test_saves_education(self, parser, mock_store):
        parser._save_to_store(SAMPLE_LINKEDIN_DATA)
        calls = mock_store.save_block.call_args_list
        categories = [c[0][0].category for c in calls]
        assert "education" in categories

    def test_saves_skills(self, parser, mock_store):
        parser._save_to_store(SAMPLE_LINKEDIN_DATA)
        calls = mock_store.save_block.call_args_list
        categories = [c[0][0].category for c in calls]
        assert "tech_skill" in categories

    def test_saves_projects(self, parser, mock_store):
        parser._save_to_store(SAMPLE_LINKEDIN_DATA)
        calls = mock_store.save_block.call_args_list
        categories = [c[0][0].category for c in calls]
        assert "project" in categories

  
    def test_confidence_is_high_for_linkedin(self, parser, mock_store):
        parser._save_to_store(SAMPLE_LINKEDIN_DATA)
        calls = mock_store.save_block.call_args_list
        for call in calls:
            block = call[0][0]
            if block.category == "project":
                assert block.confidence == "medium"  # projektai = medium
            else:
                assert block.confidence == "high"    # visa kita = high

    def test_skips_empty_fields(self, parser, mock_store):
        data = {"full_name": "Test", "headline": None, "summary": None,
                "work_experience": [], "education": [], "skills": [],
                "projects": [], "certifications": []}
        parser._save_to_store(data)
        assert mock_store.save_block.call_count == 0

    def test_returns_saved_count(self, parser, mock_store):
        count = parser._save_to_store(SAMPLE_LINKEDIN_DATA)
        assert count > 0