import json
from agents.llm_factory import get_llm
from memory.vector_store import VectorStore
from langchain_core.messages import HumanMessage
from agents.state import AgentState

MAX_BLOCK_ITERATIONS = 3
MAX_HOLISTIC_ITERATIONS = 2

GENERATE_SUMMARY_PROMPT = """You are a professional CV writer.
Write a compelling personal summary based ONLY on real candidate data.

Job match analysis:
{job_match}

Candidate Master Profile:
{master_profile}

Guidelines:
- 3-4 sentences maximum
- Use candidate's authentic phrases where possible
- Highlight strongest matches to this job
- Do NOT fabricate — only use provided information
- Write in {language}

Return ONLY the summary text."""


GENERATE_EXPERIENCE_PROMPT = """You are a professional CV writer.
Write work experience descriptions based ONLY on real candidate data.

Job match analysis:
{job_match}

Work experience from Master Profile:
{experience}

Guidelines:
- Focus on achievements relevant to job match
- Use action verbs
- Quantify where data supports it — never invent numbers
- Write in {language}

Return ONLY a JSON array:
[
    {{
        "company": "company name",
        "role": "job title",
        "period": "dates",
        "bullets": ["achievement 1", "achievement 2"]
    }}
]"""


GENERATE_PROJECTS_PROMPT = """You are a professional CV writer.
Select and describe the most relevant projects for this CV.

Job match analysis:
{job_match}

Projects from Master Profile:
{projects}

Guidelines:
- Select 2-3 most relevant projects
- Describe in business value language, not just technical
- Write in {language}

Return ONLY a JSON array:
[
    {{
        "name": "project name",
        "description": "2-3 sentences — what it does and why it matters",
        "tech_stack": ["tech1", "tech2"],
        "highlight": "key achievement or metric"
    }}
]"""


GENERATE_SKILLS_PROMPT = """You are a professional CV writer.
Select and organize skills most relevant to this job.

Job match analysis:
{job_match}

Skills from Master Profile:
{skills}

Return ONLY a JSON object:
{{
    "technical": ["most relevant tech skills"],
    "tools": ["relevant tools and frameworks"],
    "soft": ["relevant soft skills"]
}}"""


HOLISTIC_REVIEW_PROMPT = """You are a senior CV reviewer doing a holistic review.

Complete CV sections:
{cv_sections}

Job match analysis:
{job_match}

Evaluate:
1. Do all sections tell ONE clear story?
2. Is there a consistent professional identity?
3. Would a recruiter understand in 30 seconds who this person is?
4. Are there contradictions between sections?

Return ONLY a JSON object:
{{
    "is_coherent": true/false,
    "main_message": "the CV's core message in one sentence",
    "issues": ["issue 1", "issue 2"],
    "suggestions": ["suggestion 1", "suggestion 2"],
    "ready_to_proceed": true/false
}}"""


class CVAgent:
    def __init__(self, config: dict):
        self.llm = get_llm(config)
        self.store = VectorStore(config)
        self.language = config["language"]["default"]

    def _call_llm(self, prompt: str) -> str:
        response = self.llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return raw.strip()

    def _parse_json(self, text: str):
        try:
            return json.loads(text)
        except Exception as e:
            print(f"⚠️  Parse error: {e}")
            return None

    def _confidence_score(self, section: str, data, job_match: dict) -> str:
        """Returns confidence score based on real data richness"""
        strong = job_match.get("strong_matches", [])
        partial = job_match.get("partial_matches", [])

        if section == "experience":
            if not data:
                return "░░░░░ 0% — no data"
            score = min(len(data) * 25, 100)
            bar = "█" * (score // 20) + "░" * (5 - score // 20)
            return f"{bar} {score}%"

        elif section == "projects":
            if not data:
                return "░░░░░ 0% — no data"
            score = min(len(data) * 30, 100)
            bar = "█" * (score // 20) + "░" * (5 - score // 20)
            return f"{bar} {score}%"

        elif section == "skills":
            strong_count = len(strong)
            score = min(strong_count * 20, 100)
            bar = "█" * (score // 20) + "░" * (5 - score // 20)
            return f"{bar} {score}% — {strong_count} strong matches"

        elif section == "summary":
            score = 70 if strong else 40
            bar = "█" * (score // 20) + "░" * (5 - score // 20)
            return f"{bar} {score}%"

        return "░░░░░"

    def _generate_section(self, section: str, master_profile: dict, job_match: dict):
        """Generates a single CV section"""
        job_match_str = json.dumps(job_match, indent=2)
        profile_str = json.dumps(master_profile, indent=2)

        if section == "summary":
            prompt = GENERATE_SUMMARY_PROMPT.format(
                job_match=job_match_str,
                master_profile=profile_str,
                language=self.language
            )
            return self._call_llm(prompt)

        elif section == "experience":
            experience = json.dumps(master_profile.get("work_experience", []))
            prompt = GENERATE_EXPERIENCE_PROMPT.format(
                job_match=job_match_str,
                experience=experience,
                language=self.language
            )
            return self._parse_json(self._call_llm(prompt)) or []

        elif section == "projects":
            projects = json.dumps(master_profile.get("projects", []))
            prompt = GENERATE_PROJECTS_PROMPT.format(
                job_match=job_match_str,
                projects=projects,
                language=self.language
            )
            return self._parse_json(self._call_llm(prompt)) or []

        elif section == "skills":
            skills = json.dumps(master_profile.get("technical_skills", {}))
            soft = json.dumps(master_profile.get("soft_skills", []))
            prompt = GENERATE_SKILLS_PROMPT.format(
                job_match=job_match_str,
                skills=f"Technical: {skills}\nSoft: {soft}"
            )
            return self._parse_json(self._call_llm(prompt)) or {}

    def _display_section(self, section: str, data, score: str):
        """Displays a single CV section"""
        labels = {
            "summary": "📝 PERSONAL SUMMARY",
            "experience": "💼 WORK EXPERIENCE",
            "projects": "🚀 PROJECTS",
            "skills": "🛠️  SKILLS"
        }
        print(f"\n{'─'*50}")
        print(f"{labels.get(section, section)}")
        print(f"Confidence: {score}")
        print("─"*50)

        if section == "summary":
            print(f"\n{data}\n")

        elif section == "experience":
            for exp in data or []:
                print(f"\n{exp.get('role')} @ {exp.get('company')} | {exp.get('period')}")
                for b in exp.get("bullets", []):
                    print(f"  • {b}")

        elif section == "projects":
            for proj in data or []:
                print(f"\n{proj.get('name')}")
                print(f"  {proj.get('description')}")
                print(f"  Stack: {', '.join(proj.get('tech_stack', []))}")
                if proj.get("highlight"):
                    print(f"  ⭐ {proj.get('highlight')}")

        elif section == "skills":
            if data:
                if data.get("technical"):
                    print(f"Technical: {', '.join(data['technical'])}")
                if data.get("tools"):
                    print(f"Tools:     {', '.join(data['tools'])}")
                if data.get("soft"):
                    print(f"Soft:      {', '.join(data['soft'])}")

    def _review_block(self, section: str, data, master_profile: dict, job_match: dict):
        """Block-level review with iteration limit"""
        iterations = 0

        while iterations < MAX_BLOCK_ITERATIONS:
            score = self._confidence_score(section, data, job_match)
            self._display_section(section, data, score)

            if iterations > 0:
                print(f"  (Iteration {iterations}/{MAX_BLOCK_ITERATIONS})")

            print("\n  Options:")
            print("  'ok'       — accept this section")
            print("  'redo'     — regenerate")
            print("  'feedback' — provide specific feedback")
            print("  'skip'     — skip for now")

            choice = input("\n  Your choice: ").strip().lower()

            if choice == "ok":
                return data

            elif choice == "skip":
                return data

            elif choice == "redo":
                iterations += 1
                print(f"\n  🔄 Regenerating {section}...")
                data = self._generate_section(section, master_profile, job_match)

            elif choice == "feedback":
                feedback = input("  Your feedback: ").strip()
                iterations += 1
                print(f"\n  🔄 Applying feedback...")

                fix_prompt = f"""Improve this CV section based on feedback.
Section type: {section}
Current content: {json.dumps(data)}
Feedback: {feedback}
Job match context: {json.dumps(job_match.get('emphasis_for_cv', []))}
Return ONLY the improved content in the same format."""

                result = self._call_llm(fix_prompt)
                if section == "summary":
                    data = result
                else:
                    parsed = self._parse_json(result)
                    if parsed:
                        data = parsed

            else:
                print("  ❓ Unknown command")

        # Max iterations reached
        print(f"\n⚠️  Maximum iterations reached for {section}.")
        print("  'keep' — keep best version | 'skip' — skip section")
        choice = input("  Your choice: ").strip().lower()
        return data if choice != "skip" else None

    def _holistic_review(self, cv_sections: dict, job_match: dict) -> bool:
        """Holistic review — checks overall coherence"""
        print("\n" + "="*50)
        print("🔍 HOLISTIC REVIEW — 30-second scan test")
        print("="*50)

        iterations = 0
        while iterations < MAX_HOLISTIC_ITERATIONS:
            prompt = HOLISTIC_REVIEW_PROMPT.format(
                cv_sections=json.dumps(cv_sections, indent=2),
                job_match=json.dumps(job_match, indent=2)
            )
            result = self._parse_json(self._call_llm(prompt))

            if not result:
                break

            print(f"\n📌 Core message: {result.get('main_message', '?')}")
            print(f"✅ Coherent: {result.get('is_coherent', False)}")

            if result.get("issues"):
                print("\n⚠️  Issues found:")
                for issue in result["issues"]:
                    print(f"  • {issue}")

            if result.get("suggestions"):
                print("\n💡 Suggestions:")
                for s in result["suggestions"]:
                    print(f"  • {s}")

            print("\nOptions:")
            print("  'ok'     — proceed to quality pipeline")
            print("  'adjust' — make adjustments")

            choice = input("\nYour choice: ").strip().lower()

            if choice == "ok":
                return True
            elif choice == "adjust":
                iterations += 1
                feedback = input("What to adjust: ").strip()
                # Apply holistic feedback
                fix_prompt = f"""Improve the CV summary to better reflect this feedback.
Current summary: {cv_sections.get('summary', '')}
Feedback: {feedback}
Core message should be: {result.get('main_message', '')}
Return ONLY the improved summary text."""
                cv_sections["summary"] = self._call_llm(fix_prompt)
            else:
                print("❓ Unknown command")

        return True

    def _display_full_cv(self, cv_sections: dict, master_profile: dict):
        """Displays complete CV preview"""
        print("\n" + "="*60)
        print("📄 FULL CV PREVIEW")
        print("="*60)

        name = master_profile.get("full_name", "")
        headline = master_profile.get("headline", "")
        if name:
            print(f"\n{name}")
        if headline:
            print(f"{headline}")

        print(f"\n{cv_sections.get('summary', '')}")

        print("\n── WORK EXPERIENCE ──")
        for exp in cv_sections.get("experience", []):
            print(f"\n{exp.get('role')} @ {exp.get('company')} | {exp.get('period')}")
            for b in exp.get("bullets", []):
                print(f"  • {b}")

        print("\n── SKILLS ──")
        skills = cv_sections.get("skills", {})
        if skills.get("technical"):
            print(f"Technical: {', '.join(skills['technical'])}")
        if skills.get("tools"):
            print(f"Tools:     {', '.join(skills['tools'])}")
        if skills.get("soft"):
            print(f"Soft:      {', '.join(skills['soft'])}")

        print("\n── PROJECTS ──")
        for proj in cv_sections.get("projects", []):
            print(f"\n{proj.get('name')}")
            print(f"  {proj.get('description')}")
            print(f"  Stack: {', '.join(proj.get('tech_stack', []))}")

        print("\n── EDUCATION ──")
        for edu in master_profile.get("education", []):
            print(f"  • {edu.get('degree')} — {edu.get('institution')} {edu.get('period', '')}")

        print("\n" + "="*60)

    def run(self, state: AgentState, master_profile: dict, job_match: dict) -> dict:
        """Main CV generation flow"""
        print("\n" + "="*50)
        print("📄 CV GENERATION")
        print("="*50)

        # Show match score
        match_score = job_match.get("match_score", 0)
        unique_value = job_match.get("unique_value", "")
        print(f"\n🎯 Match score: {match_score}/100")
        if unique_value:
            print(f"⭐ Your unique value: {unique_value}")

        gaps = job_match.get("gaps", [])
        if gaps:
            print(f"⚠️  Gaps: {', '.join(gaps)}")

        input("\nPress Enter to start generating CV sections...")

        # Generate and review each section
        sections_order = ["summary", "experience", "projects", "skills"]
        cv_sections = {}

        for section in sections_order:
            print(f"\n🤖 Generating {section}...")
            data = self._generate_section(section, master_profile, job_match)
            reviewed = self._review_block(section, data, master_profile, job_match)
            if reviewed is not None:
                cv_sections[section] = reviewed

        # Holistic review
        self._holistic_review(cv_sections, job_match)

        # Full preview
        self._display_full_cv(cv_sections, master_profile)

        print("\n✅ CV sections complete!")
        print("Next: Human Tone Polisher → CV Optimizer → PDF Export")
        input("\nPress Enter to continue...")

        return {
            "cv_sections": cv_sections,
            "user_input": "menu"
        }