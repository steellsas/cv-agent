from agents.llm_factory import get_llm
from memory.profile_store import ProfileStore, UserProfile
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import json

MAX_QUESTIONS = 8

SYSTEM_PROMPT = """You are a warm and experienced HR consultant building a comprehensive career profile.
You have already reviewed the candidate's CV data.
Your goal is to deepen, clarify and enrich what you already know.

Guidelines:
- Be warm, conversational and encouraging
- Ask ONE question at a time
- Always reference what you already know
- Dig deeper into achievements, challenges, impact and motivation
- Extract soft skills through natural conversation
- Preserve candidate's exact phrases — they reveal authentic voice
- Never ask about something already well covered
- Respond in the same language the user is using"""

ANALYZE_PROFILE_PROMPT = """You are analyzing a career profile to find gaps and areas to explore.

Current profile:
{profile_summary}

Identify gaps and return ONLY JSON:
{{
    "well_covered": ["what is already known well"],
    "needs_deepening": [
        {{
            "area": "work experience / projects / skills",
            "known": "what we know",
            "missing": "what would add value"
        }}
    ],
    "completely_missing": ["personality", "motivation"],
    "priority_questions": [
        "specific question referencing known data"
    ]
}}

Return ONLY JSON."""

GENERATE_QUESTION_PROMPT = """You are an HR consultant continuing a profile building conversation.

Known profile data:
{profile_summary}

Conversation so far:
{conversation}

Gaps still to fill:
{gaps}

Questions asked so far: {questions_asked}/{max_questions}

Generate the NEXT single question that:
- References something already known if relevant
- Fills the most important gap
- Feels natural in conversation flow
- Is warm and encouraging

If enough info collected or max questions reached, respond with: DONE

Return ONLY the question text or DONE."""

EXTRACT_FROM_ANSWER_PROMPT = """Extract career profile information from this conversation answer.
All output must be in English regardless of input language.

Question asked: {question}
Answer: {answer}

Return ONLY JSON:
{{
    "work_experience_update": {{
        "company": "if updating specific company or null",
        "achievements": ["new achievement mentioned"],
        "impact": "business impact mentioned or null",
        "key_phrases": ["authentic phrase used"]
    }},
    "personality": {{
        "traits": ["trait mentioned"],
        "work_style": "work style described or null",
        "motivation": "motivation mentioned or null"
    }},
    "soft_skills": ["soft skill mentioned"],
    "technical_skills": ["tech skill mentioned"],
    "authentic_phrases": ["exact memorable phrase"],
    "other": "any other relevant info or null"
}}

Return ONLY JSON."""


class ProfileBuilder:
    def __init__(self, config: dict):
        self.llm = get_llm(config)
        self.store = ProfileStore()
        self.config = config

    def _get_profile_summary_text(self) -> str:
        """Returns readable profile summary for LLM"""
        p = self.store.profile
        parts = []

        if p.personal.name:
            parts.append(f"Name: {p.personal.name}")
        if p.personal.headline:
            parts.append(f"Headline: {p.personal.headline}")
        if p.summary:
            parts.append(f"Summary: {p.summary[:300]}")

        if p.experience:
            parts.append("\nWork Experience:")
            for exp in p.experience:
                parts.append(f"  - {exp.role} @ {exp.company} ({exp.period})")
                if exp.achievements:
                    parts.append(f"    Achievements: {'; '.join(exp.achievements[:2])}")
                if exp.impact:
                    parts.append(f"    Impact: {exp.impact}")

        if p.projects:
            parts.append("\nProjects:")
            for proj in p.projects:
                parts.append(f"  - {proj.name}: {proj.description[:80]}")

        if p.skills.technical:
            parts.append(f"\nTechnical: {', '.join(p.skills.technical[:10])}")
        if p.skills.soft:
            parts.append(f"Soft skills: {', '.join(p.skills.soft)}")

        if p.personality.work_style:
            parts.append(f"\nWork style: {p.personality.work_style}")
        if p.personality.motivation:
            parts.append(f"Motivation: {p.personality.motivation}")
        if p.personality.traits:
            parts.append(f"Traits: {', '.join(p.personality.traits)}")

        if p.education:
            parts.append("\nEducation:")
            for edu in p.education:
                parts.append(f"  - {edu.degree} @ {edu.institution}")

        return "\n".join(parts)

    def _analyze_gaps(self) -> dict:
        """Analyzes profile and finds what needs to be explored"""
        profile_summary = self._get_profile_summary_text()

        if not profile_summary.strip():
            return {
                "well_covered": [],
                "needs_deepening": [],
                "completely_missing": ["everything"],
                "priority_questions": [
                    "Let's start from the beginning — tell me about yourself and your career journey."
                ]
            }

        prompt = ANALYZE_PROFILE_PROMPT.format(profile_summary=profile_summary)
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
                "completely_missing": [],
                "priority_questions": ["Tell me more about your experience and achievements."]
            }

    def _get_next_question(self, messages: list, gaps: dict, questions_asked: int) -> str:
        """Generates next question based on gaps and conversation"""
        profile_summary = self._get_profile_summary_text()
        conversation = "\n".join([
            f"{'Agent' if isinstance(m, AIMessage) else 'Candidate'}: {m.content[:100]}"
            for m in messages[-6:]
        ])

        prompt = GENERATE_QUESTION_PROMPT.format(
            profile_summary=profile_summary,
            conversation=conversation,
            gaps=json.dumps(gaps, indent=2),
            questions_asked=questions_asked,
            max_questions=MAX_QUESTIONS
        )

        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()

    def _extract_and_save(self, question: str, answer: str):
        """Extracts info from answer and saves to ProfileStore"""
        prompt = EXTRACT_FROM_ANSWER_PROMPT.format(
            question=question,
            answer=answer
        )
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

        saved = 0

        # Update work experience
        exp_update = data.get("work_experience_update", {})
        if exp_update and exp_update.get("company"):
            self.store.update_experience(
                company=exp_update["company"],
                role="",
                updates={
                    k: v for k, v in exp_update.items()
                    if k != "company" and v
                }
            )
            saved += 1

        # Update personality
        personality = data.get("personality", {})
        if any(personality.values()):
            self.store.update_personality(
                traits=personality.get("traits", []),
                work_style=personality.get("work_style", ""),
                motivation=personality.get("motivation", ""),
                phrases=data.get("authentic_phrases", [])
            )
            saved += 1

        # Add soft skills
        soft = data.get("soft_skills", [])
        if soft:
            self.store.add_skills(soft=soft)
            saved += 1

        # Add technical skills
        tech = data.get("technical_skills", [])
        if tech:
            self.store.add_skills(technical=tech)
            saved += 1

        if saved > 0:
            self.store.save()
            print(f"\n  💾 Profile updated with {saved} new piece(s)")

    def _show_progress(self, questions_asked: int, gaps: dict):
        """Shows conversation progress"""
        remaining = MAX_QUESTIONS - questions_asked
        well_covered = gaps.get("well_covered", [])
        missing = gaps.get("completely_missing", [])

        print(f"\n  📊 Progress: {questions_asked}/{MAX_QUESTIONS} questions")
        if well_covered:
            print(f"  ✅ Covered: {', '.join(well_covered[:3])}")
        if missing:
            print(f"  ❓ Still needed: {', '.join(missing[:3])}")
        if remaining <= 2:
            print(f"  ⚠️  {remaining} question(s) remaining")

    def run(self) -> bool:
        """Main profile building conversation"""
        print("\n" + "="*50)
        print("📋 PROFILE BUILDER — Smart HR")
        print("="*50)

        # Check existing profile
        is_empty = self.store.is_empty()

        if not is_empty:
            self.store.display_summary()
            print("\n  Profile data found!")
            print("  I'll ask targeted questions to fill gaps and deepen your profile.")
        else:
            print("\n  No profile data found.")
            print("  I'll guide you through building your profile from scratch.")
            print("  Tip: Upload your CV first with 'upload' for a faster experience!")

        print("\nType 'done' to finish | 'skip' to skip a question | 'menu' to go back\n")
        input("Press Enter to start...")

        # Analyze gaps
        print("\n🔍 Analyzing your profile...")
        gaps = self._analyze_gaps()

        well_covered = gaps.get("well_covered", [])
        missing = gaps.get("completely_missing", [])

        if well_covered:
            print(f"✅ Already well covered: {', '.join(well_covered)}")
        if missing and "everything" not in missing:
            print(f"❓ Areas to explore: {', '.join(missing)}")

        messages = []
        questions_asked = 0
        last_question = ""

        # Opening message
        if is_empty:
            opening = "Hi! I'm here to help build your career profile. Let's start — could you tell me about yourself and your career journey so far?"
        else:
            name = self.store.profile.personal.name
            greeting = f"Hi {name}!" if name else "Hi!"
            opening = f"{greeting} I've reviewed your CV data. I'd like to ask a few targeted questions to strengthen your profile. Let's dig a bit deeper!"

        print(f"\n🤖 Agent: {opening}\n")
        messages.append(AIMessage(content=opening))

        while questions_asked < MAX_QUESTIONS:
            # Generate next question
            next_q = self._get_next_question(messages, gaps, questions_asked)

            # Check if agent says DONE
            if "DONE" in next_q.upper() and len(next_q) < 20:
                print("\n✅ Agent: Great — I have everything I need!")
                break

            print(f"🤖 Agent: {next_q}\n")
            messages.append(AIMessage(content=next_q))
            last_question = next_q
            questions_asked += 1

            # User input
            user_input = input("You: ").strip()

            if user_input.lower() == "done":
                break
            if user_input.lower() == "menu":
                return False
            if user_input.lower() == "skip":
                print("  ⏭️  Skipped\n")
                messages.append(HumanMessage(content="[skipped]"))
                continue

            messages.append(HumanMessage(content=user_input))

            # Extract and save
            self._extract_and_save(last_question, user_input)

            # Re-analyze gaps
            gaps = self._analyze_gaps()

            # Show progress every 3 questions
            if questions_asked % 3 == 0:
                self._show_progress(questions_asked, gaps)

        # Final summary
        print("\n" + "="*50)
        print("✅ PROFILE SESSION COMPLETE")
        print("="*50)
        self.store.display_summary()

        return True