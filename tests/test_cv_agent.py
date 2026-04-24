import pytest
import json
from unittest.mock import MagicMock, patch
from agents.cv_agent import CVAgent


@pytest.fixture
def cv_agent(config, mock_store, mock_llm):
    with patch("agents.cv_agent.VectorStore", return_value=mock_store):
        with patch("agents.cv_agent.get_llm", return_value=mock_llm):
            agent = CVAgent(config)
            agent.store = mock_store
            agent.llm = mock_llm
            return agent


class TestCVAgent:

    def test_generate_summary(self, cv_agent, mock_llm, sample_master_profile, sample_job_match):
        mock_llm.invoke.return_value = MagicMock(
            content="Experienced AI engineer with strong Python background."
        )
        result = cv_agent._generate_section("summary", sample_master_profile, sample_job_match)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_experience_returns_list(self, cv_agent, mock_llm,
                                               sample_master_profile, sample_job_match):
        experience_data = [
            {"company": "DeVoro", "role": "Python Developer",
             "period": "2019-2021", "bullets": ["Built Django platform"]}
        ]
        mock_llm.invoke.return_value = MagicMock(content=json.dumps(experience_data))
        result = cv_agent._generate_section("experience", sample_master_profile, sample_job_match)
        assert isinstance(result, list)
        assert len(result) > 0
        assert "company" in result[0]

    def test_generate_projects_returns_list(self, cv_agent, mock_llm,
                                             sample_master_profile, sample_job_match):
        projects_data = [
            {"name": "ISP-AI-Agent", "description": "AI bot",
             "tech_stack": ["Python"], "highlight": "50% faster support"}
        ]
        mock_llm.invoke.return_value = MagicMock(content=json.dumps(projects_data))
        result = cv_agent._generate_section("projects", sample_master_profile, sample_job_match)
        assert isinstance(result, list)
        assert result[0]["name"] == "ISP-AI-Agent"

    def test_generate_skills_returns_dict(self, cv_agent, mock_llm,
                                           sample_master_profile, sample_job_match):
        skills_data = {
            "technical": ["Python", "FastAPI"],
            "tools": ["Docker", "Git"],
            "soft": ["communication", "problem solving"]
        }
        mock_llm.invoke.return_value = MagicMock(content=json.dumps(skills_data))
        result = cv_agent._generate_section("skills", sample_master_profile, sample_job_match)
        assert isinstance(result, dict)
        assert "technical" in result
        assert "tools" in result

    def test_confidence_score_no_experience(self, cv_agent, sample_job_match):
        score = cv_agent._confidence_score("experience", [], sample_job_match)
        assert "0%" in score

    # def test_confidence_score_with_experience(self, cv_agent, sample_job_match):
    #     data = [{"company": "A"}, {"company": "B"}]
    #     score = cv_agent._confidence_score("experience", data, sample_job_match)
    #     assert "0%" not in score

    def test_confidence_score_with_experience(self, cv_agent, sample_job_match):
        data = [{"company": "A"}, {"company": "B"}]
        score = cv_agent._confidence_score("experience", data, sample_job_match)
        assert "░░░░░ 0%" not in score  # ← tikslesnė patikra

    def test_confidence_score_skills_shows_matches(self, cv_agent, sample_job_match):
        score = cv_agent._confidence_score("skills", {}, sample_job_match)
        assert "strong matches" in score or "%" in score

    def test_parse_json_valid(self, cv_agent):
        result = cv_agent._parse_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_parse_json_invalid_returns_none(self, cv_agent):
        result = cv_agent._parse_json("not json at all {{")
        assert result is None

    def test_call_llm_strips_markdown(self, cv_agent, mock_llm):
        mock_llm.invoke.return_value = MagicMock(
            content="```json\n{\"key\": \"value\"}\n```"
        )
        result = cv_agent._call_llm("test prompt")
        assert not result.startswith("```")
        assert "{" in result