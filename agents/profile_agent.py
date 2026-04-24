from agents.state import AgentState
from agents.llm_factory import get_llm
from memory.vector_store import VectorStore, ProfileBlock
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import json

SYSTEM_PROMPT = """You are a warm and experienced HR consultant helping to build a comprehensive career profile.
You have already reviewed the candidate's available data (LinkedIn, GitHub, old CVs).
Your goal is to deepen, clarify and enrich what you already know — not repeat questions about known facts.

Guidelines:
- Be warm, conversational and encouraging
- Ask ONE question at a time
- Reference what you already know: "I can see you worked at X — tell me more about..."
- Dig deeper into achievements, challenges, impact and motivation
- Extract soft skills and personality through natural conversation
- Save the candidate's exact phrases — they reveal authentic voice
- Always respond in the same language the user is using
- Never ask about something already well covered unless clarifying"""

ANALYZE_GAPS_PROMPT = """You are analyzing a career profile to find gaps and areas to explore.

Current profile summary:
{profile_summary}

Based on this, identify:
1. What is well covered (don't ask about these)
2. What needs deepening (ask follow-up questions)
3. What is completely missing (ask directly)

Return ONLY a JSON object:
{{
    "well_covered": ["category1", "category2"],
    "needs_deepening": [
        {{"category": "work_experience", "known": "worked at Balticum TV", "ask_about": "key achievements and impact"}}
    ],
    "missing": ["soft_skills", "motivation"],
    "suggested_questions": [
        "I can see you worked at Balticum TV for 15 years — what was your biggest achievement there?",
        "Your GitHub shows strong Python skills — how did you develop these professionally?"
    ]
}}

Return ONLY JSON, no other text."""

EXTRACT_PROMPT = """Extract career information from this message and return ONLY a JSON object.
All output must be in English regardless of input language.

{{
    "work_experience": "work experience mentioned or null",
    "project": "projects mentioned or null",
    "education": "education mentioned or null",
    "tech_skill": "technical skills mentioned or null",
    "soft_skill": "soft skills or traits mentioned or null",
    "personality": "personal characteristics or null",
    "user_phrase": "memorable exact phrase that reveals authentic voice or null",
    "other": "any other relevant info or null"
}}

Message: {message}
Return ONLY JSON."""


class ProfileAgent:
    def __init__(self, config: dict):
        self.llm = get_llm(config)
        self.store = VectorStore(config)
        self.config = config

    def _analyze_gaps(self) -> dict:
        """Checks what is known and what needs exploring"""
        summary = self.store.get_profile_summary()

        summary_text = ""
        for category, data in summary.items():
            summary_text += f"\n{category}: {data['count']} entries, sources: {data['sources']}"

        if not any(data["count"] > 0 for data in summary.values()):
            return {
                "well_covered": [],
                "needs_deepening": [],
                "missing": ["everything"],
                "suggested_questions": [
                    "Let's start from the beginning — could you tell me about yourself and your career journey?"
                ]
            }

        prompt = ANALYZE_GAPS_PROMPT.format(profile_summary=summary_text)
        response = self.llm.invoke([HumanMessage(content=prompt)])

        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        try:
            return json.loads(raw.strip())
        except:
            return {
                "well_covered": [],
                "needs_deepening": [],
                "missing": [],
                "suggested_questions": ["Tell me more about your experience and skills."]
            }

    def _get_opening(self, gaps: dict) -> str:
        """Generates opening message based on what is already known"""
        well_covered = gaps.get("well_covered", [])
        missing = gaps.get("missing", [])

        if "everything" in missing:
            return "Hi! I'm here to help build your career profile. Let's start from scratch — could you tell me about yourself and your career journey so far?"

        covered_text = ", ".join(well_covered) if well_covered else "some information"
        return (
            f"Hi! I've already reviewed your LinkedIn and GitHub data — "
            f"I can see {covered_text} is well covered. "
            f"I'd like to dig a bit deeper and fill in some gaps. "
            f"I'll ask you a few targeted questions."
        )

    def _extract_and_save(self, user_message: str):
        """Extracts info from message and saves structured blocks"""
        prompt = EXTRACT_PROMPT.format(message=user_message)
        response = self.llm.invoke([HumanMessage(content=prompt)])

        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        try:
            data = json.loads(raw.strip())
        except:
            return

        category_map = {
            "work_experience": "work_experience",
            "project": "project",
            "education": "education",
            "tech_skill": "tech_skill",
            "soft_skill": "soft_skill",
            "personality": "personality",
            "user_phrase": "user_phrase",
            "other": "other"
        }

        saved = 0
        for field, category in category_map.items():
            if data.get(field):
                self.store.save_block(ProfileBlock(
                    category=category,
                    text=data[field],
                    source=["conversation"],
                    confidence="medium",
                    metadata={"type": field}
                ))
                saved += 1

        if saved > 0:
            print(f"\n  💾 Saved {saved} new piece(s) to profile")

    def _get_next_question(self, messages: list, gaps: dict) -> str:
        """Gets next question from LLM based on gaps and conversation"""
        suggested = gaps.get("suggested_questions", [])
        needs_deepening = gaps.get("needs_deepening", [])

        context = ""
        if suggested:
            context = f"\nSuggested areas to explore: {json.dumps(suggested[:2])}"
        if needs_deepening:
            context += f"\nAreas needing depth: {json.dumps(needs_deepening[:2])}"

        system = SYSTEM_PROMPT + context
        all_messages = [SystemMessage(content=system)] + messages[-8:]

        response = self.llm.invoke(all_messages)
        return response.content

    def run(self, state: AgentState) -> dict:
        """Main smart HR conversation loop"""
        print("\n" + "="*50)
        print("📋 PROFILE BUILDING — Smart HR")
        print("="*50)
        print("Type 'done' when finished, 'menu' to go back\n")

        # Analyze what we already know
        print("🔍 Analyzing existing profile data...")
        gaps = self._analyze_gaps()

        well_covered = gaps.get("well_covered", [])
        missing = gaps.get("missing", [])

        if well_covered:
            print(f"  ✅ Well covered: {', '.join(well_covered)}")
        if missing and "everything" not in missing:
            print(f"  ❓ Needs attention: {', '.join(missing)}")

        messages = list(state.messages)

        # Opening based on what we know
        opening = self._get_opening(gaps)
        print(f"\n🤖 Agent: {opening}\n")
        messages.append(AIMessage(content=opening))

        while True:
            # Get next question
            agent_message = self._get_next_question(messages, gaps)
            print(f"🤖 Agent: {agent_message}\n")
            messages.append(AIMessage(content=agent_message))

            # User input
            user_input = input("You: ").strip()

            if user_input.lower() == "done":
                print("\n✅ Profile session complete!")

                # Show updated summary
                summary = self.store.get_profile_summary()
                print("\n📊 Profile completeness:")
                for cat, data in summary.items():
                    if data["count"] > 0:
                        bar = "█" * min(data["count"], 10)
                        print(f"  {cat:<20} {bar} ({data['count']} entries, sources: {', '.join(data['sources'])})")

                return {
                    "messages": messages,
                    "user_input": "menu",
                    "profile_complete": True
                }

            if user_input.lower() == "menu":
                return {"messages": messages, "user_input": "menu"}

            messages.append(HumanMessage(content=user_input))

            # Extract and save new info
            self._extract_and_save(user_input)

            # Re-analyze gaps after new info
            gaps = self._analyze_gaps()