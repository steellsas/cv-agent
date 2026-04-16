import json
from agents.llm_factory import get_llm
from memory.vector_store import VectorStore
from prompts.cv_prompts import (
    ANALYZE_JOB_PROMPT,
    GENERATE_SUMMARY_PROMPT,
    GENERATE_EXPERIENCE_PROMPT,
    GENERATE_SKILLS_PROMPT,
    GENERATE_PROJECTS_PROMPT
)
from langchain_core.messages import HumanMessage
from agents.state import AgentState


class CVAgent:
    def __init__(self, config: dict):
        self.llm = get_llm(config)
        self.store = VectorStore(config)
        self.config = config
        self.language = config["language"]["default"]

    def _call_llm(self, prompt: str) -> str:
        """Helper to call LLM and clean response"""
        response = self.llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return raw.strip()

    def _parse_json(self, text: str) -> dict | list:
        """Safely parse JSON response"""
        try:
            return json.loads(text)
        except Exception as e:
            print(f"⚠️ JSON parse error: {e}")
            return {}

    def _analyze_job(self, job_posting: str) -> dict:
        """Extracts key requirements from job posting"""
        print("\n🔍 Analyzing job posting...")
        prompt = ANALYZE_JOB_PROMPT.format(job_posting=job_posting)
        result = self._parse_json(self._call_llm(prompt))
        print(f"  Position: {result.get('job_title', 'Unknown')}")
        print(f"  Level: {result.get('experience_level', 'Unknown')}")
        print(f"  Required skills: {', '.join(result.get('required_skills', []))}")
        return result

    def _get_profile_info(self, job_requirements: dict) -> dict:
        """Retrieves relevant profile info from Qdrant"""
        print("\n📋 Retrieving profile information...")

        # Build search query from job requirements
        query = f"{job_requirements.get('job_title', '')} {' '.join(job_requirements.get('required_skills', []))}"

        profile = {
            "experience": self.store.search(query, category="work_experience", top_k=5),
            "projects": self.store.search(query, category="project", top_k=5),
            "tech_skills": self.store.get_all(category="tech_skill"),
            "soft_skills": self.store.get_all(category="soft_skill"),
            "personality": self.store.get_all(category="personality"),
            "education": self.store.get_all(category="education"),
        }

        print(f"  Found: {len(profile['experience'])} experience entries")
        print(f"  Found: {len(profile['projects'])} projects")
        print(f"  Found: {len(profile['tech_skills'])} skill entries")

        return profile

    def _format_for_prompt(self, items: list) -> str:
        """Formats profile items as readable text for prompts"""
        if not items:
            return "No information available"
        return "\n".join([f"- {item['text']}" for item in items])

    def _generate_summary(self, job_req: dict, profile: dict) -> str:
        """Generates personal summary section"""
        print("\n✍️  Generating summary...")
        personality = self._format_for_prompt(profile["personality"])
        experience = self._format_for_prompt(profile["experience"][:3])
        skills = self._format_for_prompt(profile["tech_skills"][:3])

        profile_info = f"Personality:\n{personality}\n\nExperience:\n{experience}\n\nSkills:\n{skills}"

        prompt = GENERATE_SUMMARY_PROMPT.format(
            job_requirements=json.dumps(job_req, indent=2),
            profile_info=profile_info,
            language=self.language
        )
        return self._call_llm(prompt)

    def _generate_experience(self, job_req: dict, profile: dict) -> list:
        """Generates work experience section"""
        print("✍️  Generating experience section...")
        experience_info = self._format_for_prompt(profile["experience"])

        prompt = GENERATE_EXPERIENCE_PROMPT.format(
            job_requirements=json.dumps(job_req, indent=2),
            experience_info=experience_info,
            language=self.language
        )
        result = self._parse_json(self._call_llm(prompt))
        return result if isinstance(result, list) else []

    def _generate_skills(self, job_req: dict, profile: dict) -> dict:
        """Generates skills section"""
        print("✍️  Generating skills section...")
        skills_info = (
            self._format_for_prompt(profile["tech_skills"]) +
            "\n" +
            self._format_for_prompt(profile["soft_skills"])
        )

        prompt = GENERATE_SKILLS_PROMPT.format(
            job_requirements=json.dumps(job_req, indent=2),
            skills_info=skills_info
        )
        result = self._parse_json(self._call_llm(prompt))
        return result if isinstance(result, dict) else {}

    def _generate_projects(self, job_req: dict, profile: dict) -> list:
        """Generates projects section"""
        print("✍️  Generating projects section...")
        projects_info = self._format_for_prompt(profile["projects"])

        prompt = GENERATE_PROJECTS_PROMPT.format(
            job_requirements=json.dumps(job_req, indent=2),
            projects_info=projects_info,
            language=self.language
        )
        result = self._parse_json(self._call_llm(prompt))
        return result if isinstance(result, list) else []

    def _display_cv(self, cv_sections: dict):
        """Displays generated CV in terminal"""
        print("\n" + "="*60)
        print("📄 GENERATED CV PREVIEW")
        print("="*60)

        print("\n--- PERSONAL SUMMARY ---")
        print(cv_sections.get("summary", ""))

        print("\n--- WORK EXPERIENCE ---")
        for exp in cv_sections.get("experience", []):
            print(f"\n{exp.get('role')} @ {exp.get('company')} | {exp.get('period')}")
            for bullet in exp.get("bullets", []):
                print(f"  • {bullet}")

        print("\n--- SKILLS ---")
        skills = cv_sections.get("skills", {})
        if skills.get("technical_skills"):
            print(f"Technical: {', '.join(skills['technical_skills'])}")
        if skills.get("tools"):
            print(f"Tools: {', '.join(skills['tools'])}")
        if skills.get("soft_skills"):
            print(f"Soft skills: {', '.join(skills['soft_skills'])}")

        print("\n--- PROJECTS ---")
        for proj in cv_sections.get("projects", []):
            print(f"\n{proj.get('name')}")
            print(f"  {proj.get('description')}")
            print(f"  Stack: {', '.join(proj.get('tech_stack', []))}")
            if proj.get("highlights"):
                print(f"  ⭐ {proj.get('highlights')}")

        print("\n--- EDUCATION ---")
        for edu in cv_sections.get("education", []):
            print(f"  • {edu.get('text', '')}")

        print("\n" + "="*60)

    def _review_loop(self, cv_sections: dict) -> dict:
        """Lets user review and request changes to CV sections"""
        while True:
            self._display_cv(cv_sections)

            print("\nOptions:")
            print("  'ok'         — accept and proceed to PDF export")
            print("  'summary'    — regenerate summary")
            print("  'experience' — regenerate experience")
            print("  'skills'     — regenerate skills")
            print("  'projects'   — regenerate projects")
            print("  'edit'       — provide custom feedback")

            choice = input("\nYour choice: ").strip().lower()

            if choice == "ok":
                return cv_sections

            elif choice == "edit":
                feedback = input("Describe what to change: ").strip()
                print(f"\n🔄 Applying feedback: {feedback}")
                # Apply feedback via LLM
                fix_prompt = f"""
You are editing a CV. Apply the following feedback to the CV sections below.
Feedback: {feedback}

Current CV:
{json.dumps(cv_sections, indent=2)}

Return the complete updated CV as JSON with the same structure.
Return ONLY JSON, no other text."""
                result = self._parse_json(self._call_llm(fix_prompt))
                if result:
                    cv_sections = result

            elif choice in ["summary", "experience", "skills", "projects"]:
                print(f"\n🔄 Regenerating {choice}...")
                # Store job_req in cv_sections for regeneration
                job_req = cv_sections.get("_job_req", {})
                profile = cv_sections.get("_profile", {})

                if choice == "summary":
                    cv_sections["summary"] = self._generate_summary(job_req, profile)
                elif choice == "experience":
                    cv_sections["experience"] = self._generate_experience(job_req, profile)
                elif choice == "skills":
                    cv_sections["skills"] = self._generate_skills(job_req, profile)
                elif choice == "projects":
                    cv_sections["projects"] = self._generate_projects(job_req, profile)
            else:
                print("❓ Unknown option")

    def run(self, state: AgentState) -> dict:
        """Main CV generation flow"""
        print("\n" + "="*60)
        print("📄 CV GENERATION")
        print("="*60)
        print("Paste the job posting below.")
        print("When done, type 'END' on a new line and press Enter.\n")

        # Collect job posting (multiline)
        lines = []
        while True:
            line = input()
            if line.strip().upper() == "END":
                break
            lines.append(line)

        job_posting = "\n".join(lines).strip()
        if not job_posting:
            print("❌ No job posting provided")
            return {"user_input": "menu"}

        # 1. Analyze job posting
        job_req = self._analyze_job(job_posting)

        # 2. Get profile info from Qdrant
        profile = self._get_profile_info(job_req)

        # 3. Generate CV sections
        print("\n🤖 Generating CV sections...")
        education_items = profile.get("education", [])

        cv_sections = {
            "summary": self._generate_summary(job_req, profile),
            "experience": self._generate_experience(job_req, profile),
            "skills": self._generate_skills(job_req, profile),
            "projects": self._generate_projects(job_req, profile),
            "education": education_items,
            "_job_req": job_req,    # for regeneration
            "_profile": profile     # for regeneration
        }

        # 4. Review loop
        final_cv = self._review_loop(cv_sections)

        # Remove internal keys
        final_cv.pop("_job_req", None)
        final_cv.pop("_profile", None)

        print("\n✅ CV approved! Proceeding to PDF export soon...")
        input("Press Enter to continue...")

        return {
            "cv_sections": final_cv,
            "user_input": "menu"
        }