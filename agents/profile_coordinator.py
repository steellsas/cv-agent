from agents.llm_factory import get_llm
from memory.vector_store import VectorStore, ProfileBlock
from langchain_core.messages import HumanMessage
import json

MASTER_PROFILE_PROMPT = """You are a senior HR specialist creating a Master Profile from verified career data.

Analyze all the information below and create a structured, comprehensive Master Profile.
Focus on REAL strengths — do not inflate or fabricate anything.
All output must be in English.

Raw profile data:
{profile_data}

Return ONLY a JSON object:
{{
    "full_name": "name if known or null",
    "headline": "professional headline — concise, authentic, max 10 words",
    "summary": "2-3 sentence authentic career summary — who this person really is",
    "core_strengths": [
        {{
            "strength": "specific strength",
            "evidence": "concrete example from profile data"
        }}
    ],
    "work_experience": [
        {{
            "company": "company name",
            "role": "job title",
            "period": "dates",
            "key_achievements": ["achievement 1", "achievement 2"],
            "skills_demonstrated": ["skill1", "skill2"]
        }}
    ],
    "projects": [
        {{
            "name": "project name",
            "description": "what it does and why it matters",
            "tech_stack": ["tech1", "tech2"],
            "business_value": "impact or outcome",
            "source": "github/linkedin/conversation"
        }}
    ],
    "technical_skills": {{
        "strong": ["skills with clear evidence"],
        "familiar": ["skills mentioned but less evidence"]
    }},
    "soft_skills": ["skill1", "skill2"],
    "education": [
        {{
            "institution": "name",
            "degree": "degree",
            "period": "dates"
        }}
    ],
    "personality_traits": ["trait1", "trait2"],
    "authentic_phrases": ["memorable phrase from user", "another phrase"],
    "profile_gaps": ["what is missing or unclear"]
}}

Return ONLY JSON."""

STRENGTHS_FOR_JOB_PROMPT = """You are matching a candidate's profile to a job posting.

Master Profile:
{master_profile}

Job posting:
{job_posting}

Identify what this candidate can genuinely offer for this position.
Do NOT fabricate — only use what exists in the profile.

Return ONLY a JSON object:
{{
    "match_score": 0-100,
    "strong_matches": [
        {{
            "requirement": "job requirement",
            "evidence": "candidate's matching experience/skill"
        }}
    ],
    "partial_matches": [
        {{
            "requirement": "job requirement",
            "evidence": "candidate has related but not exact experience"
        }}
    ],
    "gaps": ["requirement candidate does not meet"],
    "unique_value": "what makes this candidate stand out for this role",
    "emphasis_for_cv": ["what to highlight in CV for this job"]
}}

Return ONLY JSON."""


class ProfileCoordinator:
    def __init__(self, config: dict):
        self.llm = get_llm(config)
        self.store = VectorStore(config)
        self.master_profile = None

    def _gather_all_data(self) -> str:
        """Gathers all confirmed profile blocks"""
        categories = [
            "work_experience", "project", "education",
            "tech_skill", "soft_skill", "personality",
            "user_phrase", "other"
        ]

        all_data = []
        for category in categories:
            items = self.store.get_all(category=category)
            if items:
                all_data.append(f"\n=== {category.upper()} ===")
                for item in items:
                    sources = ", ".join(item["source"])
                    confirmed = "[CONFIRMED]" if item["user_confirmed"] else "[unconfirmed]"
                    all_data.append(f"{confirmed} (source: {sources}) {item['text']}")

        return "\n".join(all_data)

    def build(self) -> dict:
        """Builds Master Profile from all collected data"""
        print("\n" + "="*50)
        print("🧠 BUILDING MASTER PROFILE")
        print("="*50)

        # Check if we have enough data
        summary = self.store.get_profile_summary()
        total = sum(d["count"] for d in summary.values())

        if total == 0:
            print("❌ No profile data found.")
            print("Please import LinkedIn, GitHub or run HR conversation first.")
            return {}

        confirmed_total = sum(d["confirmed"] for d in summary.values())
        print(f"\n📊 Data available: {total} entries ({confirmed_total} confirmed)")

        if confirmed_total == 0:
            print("⚠️  No confirmed entries found.")
            proceed = input("Build Master Profile from unconfirmed data? (yes/no): ").strip().lower()
            if proceed not in ["yes", "y"]:
                print("Run 'factcheck' first to confirm your data.")
                input("\nPress Enter to continue...")
                return {}

        # Gather and synthesize
        print("\n🔄 Gathering all profile data...")
        profile_data = self._gather_all_data()

        print("🤖 Synthesizing Master Profile...")
        prompt = MASTER_PROFILE_PROMPT.format(profile_data=profile_data)
        response = self.llm.invoke([HumanMessage(content=prompt)])

        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        try:
            self.master_profile = json.loads(raw.strip())
        except Exception as e:
            print(f"❌ Error building profile: {e}")
            return {}

        # Display Master Profile
        self._display_master_profile()

        # Save to Qdrant as special block
        self.store.save_block(ProfileBlock(
            category="master_profile",
            text=json.dumps(self.master_profile),
            source=["synthesized"],
            confidence="high",
            metadata={"type": "master_profile"},
            user_confirmed=False
        ))

        # User review
        return self._review_master_profile()

    def _display_master_profile(self):
        """Displays Master Profile clearly"""
        p = self.master_profile
        if not p:
            return

        print("\n" + "="*50)
        print("📋 MASTER PROFILE PREVIEW")
        print("="*50)

        if p.get("full_name"):
            print(f"\n👤 {p['full_name']}")
        if p.get("headline"):
            print(f"💼 {p['headline']}")
        if p.get("summary"):
            print(f"\n📝 Summary:\n   {p['summary']}")

        if p.get("core_strengths"):
            print("\n⭐ Core Strengths:")
            for s in p["core_strengths"]:
                print(f"  • {s['strength']}")
                print(f"    Evidence: {s['evidence'][:80]}")

        if p.get("work_experience"):
            print("\n💼 Work Experience:")
            for exp in p["work_experience"]:
                print(f"\n  {exp.get('role')} @ {exp.get('company')} | {exp.get('period')}")
                for ach in exp.get("key_achievements", []):
                    print(f"    • {ach}")

        if p.get("technical_skills"):
            skills = p["technical_skills"]
            if skills.get("strong"):
                print(f"\n🛠️  Strong skills: {', '.join(skills['strong'])}")
            if skills.get("familiar"):
                print(f"   Familiar with: {', '.join(skills['familiar'])}")

        if p.get("projects"):
            print("\n🚀 Projects:")
            for proj in p["projects"]:
                print(f"  • {proj['name']} — {proj.get('description', '')[:70]}")

        if p.get("profile_gaps"):
            print(f"\n⚠️  Profile gaps: {', '.join(p['profile_gaps'])}")

    def _review_master_profile(self) -> dict:
        """Lets user review and confirm Master Profile"""
        print("\n" + "─"*50)
        print("Options:")
        print("  'ok'      — confirm Master Profile")
        print("  'rebuild' — rebuild with current data")
        print("  'cancel'  — cancel")

        while True:
            choice = input("\nYour choice: ").strip().lower()

            if choice == "ok":
                # Mark as confirmed
                self.store.save_block(ProfileBlock(
                    category="master_profile",
                    text=json.dumps(self.master_profile),
                    source=["synthesized"],
                    confidence="high",
                    metadata={"type": "master_profile"},
                    user_confirmed=True
                ))
                print("\n✅ Master Profile confirmed!")
                print("You can now generate a CV with 'cv' command.")
                input("\nPress Enter to continue...")
                return self.master_profile

            elif choice == "rebuild":
                print("\n🔄 Rebuilding...")
                return self.build()

            elif choice == "cancel":
                input("\nPress Enter to continue...")
                return {}

            else:
                print("❓ Unknown command")

    def get_master_profile(self) -> dict:
        """Returns cached or loads latest Master Profile from Qdrant"""
        if self.master_profile:
            return self.master_profile

        # Try to load from Qdrant
        items = self.store.get_all(category="master_profile", confirmed_only=True)
        if items:
            try:
                self.master_profile = json.loads(items[-1]["text"])
                return self.master_profile
            except:
                pass
        return {}

    def get_strengths_for_job(self, job_posting: str) -> dict:
        """Matches Master Profile to specific job posting"""
        profile = self.get_master_profile()
        if not profile:
            return {}

        print("\n🎯 Matching profile to job posting...")
        prompt = STRENGTHS_FOR_JOB_PROMPT.format(
            master_profile=json.dumps(profile, indent=2),
            job_posting=job_posting
        )
        response = self.llm.invoke([HumanMessage(content=prompt)])

        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        try:
            match = json.loads(raw.strip())
            print(f"  Match score: {match.get('match_score', '?')}/100")
            return match
        except:
            return {}