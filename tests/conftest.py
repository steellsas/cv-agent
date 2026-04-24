import pytest
from unittest.mock import MagicMock, patch
from memory.vector_store import VectorStore, ProfileBlock

# ── Fake config ──
@pytest.fixture
def config():
    return {
        "llm": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "temperature": 0.7
        },
        "language": {"default": "en"},
        "profile": {
            "vector_db_url": "http://localhost:6333",
            "collection_name": "test_user_profile"
        }
    }

# ── Mock LLM ──
@pytest.fixture
def mock_llm():
    llm = MagicMock()
    response = MagicMock()
    response.content = "Mocked LLM response"
    llm.invoke.return_value = response
    return llm

# ── Mock VectorStore ──
@pytest.fixture
def mock_store():
    store = MagicMock(spec=VectorStore)
    store.save_block.return_value = "fake-uuid"
    store.search.return_value = []
    store.get_all.return_value = []
    store.get_profile_summary.return_value = {
        "work_experience": {"count": 0, "sources": [], "confirmed": 0},
        "project":         {"count": 0, "sources": [], "confirmed": 0},
        "education":       {"count": 0, "sources": [], "confirmed": 0},
        "tech_skill":      {"count": 0, "sources": [], "confirmed": 0},
        "soft_skill":      {"count": 0, "sources": [], "confirmed": 0},
        "personality":     {"count": 0, "sources": [], "confirmed": 0},
    }
    return store

# ── Sample ProfileBlock ──
@pytest.fixture
def sample_block():
    return ProfileBlock(
        category="work_experience",
        text="Python Developer at DeVoro (2019-2021): Built backend APIs",
        source=["linkedin"],
        confidence="high",
        metadata={"company": "DeVoro", "role": "Python Developer"},
        user_confirmed=True
    )

# ── Sample Master Profile ──
@pytest.fixture
def sample_master_profile():
    return {
        "full_name": "Andrius Plienius",
        "headline": "AI & ML Engineer | Python Backend Developer",
        "summary": "Experienced developer specializing in AI and ML solutions.",
        "core_strengths": [
            {"strength": "AI agent development", "evidence": "Built ISP AI Agent with LangChain"}
        ],
        "work_experience": [
            {
                "company": "DeVoro",
                "role": "Python Developer",
                "period": "2019-2021",
                "key_achievements": ["Built Django platform", "Integrated 3CX with Odoo"],
                "skills_demonstrated": ["Python", "Django", "API integration"]
            }
        ],
        "projects": [
            {
                "name": "ISP-AI-Agent",
                "description": "AI customer support bot for ISPs",
                "tech_stack": ["Python", "LangChain", "Streamlit"],
                "business_value": "Reduced support response time by 50%",
                "source": "github"
            }
        ],
        "technical_skills": {
            "strong": ["Python", "FastAPI", "LangChain", "Machine Learning"],
            "familiar": ["Docker", "SQL", "Django"]
        },
        "soft_skills": ["problem solving", "communication", "proactivity"],
        "education": [
            {"institution": "KTU", "degree": "BSc Information Technology", "period": "2001-2005"}
        ],
        "personality_traits": ["analytical", "creative problem solver"],
        "authentic_phrases": ["think outside the box", "problem solver"],
        "profile_gaps": []
    }

# ── Sample Job Match ──
@pytest.fixture
def sample_job_match():
    return {
        "match_score": 78,
        "strong_matches": [
            {"requirement": "Python", "evidence": "5+ years Python development"},
            {"requirement": "AI/ML", "evidence": "Multiple ML projects on GitHub"}
        ],
        "partial_matches": [
            {"requirement": "Team leadership", "evidence": "Led freelance projects"}
        ],
        "gaps": ["formal team lead experience"],
        "unique_value": "Combines technical AI skills with business communication background",
        "emphasis_for_cv": ["AI agent development", "Python backend", "problem solving"]
    }

