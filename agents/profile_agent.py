from agents.state import AgentState
from agents.llm_factory import get_llm
from memory.vector_store import VectorStore
from prompts.profile_prompts import SYSTEM_PROMPT, HR_QUESTIONS, EXTRACT_PROMPT
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import json

class ProfileAgent:
    def __init__(self, config: dict):
        self.llm = get_llm(config)
        self.store = VectorStore(config)
        self.config = config
        self.question_index = 0

    def _extract_and_save(self, user_message: str):
        """Extracts structured info from user message and saves to Qdrant"""
        prompt = EXTRACT_PROMPT.format(user_message=user_message)
        response = self.llm.invoke([HumanMessage(content=prompt)])

        try:
            # Clean response and parse JSON
            raw = response.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            data = json.loads(raw.strip())

            # Save each non-null field to Qdrant
            category_map = {
                "work_experience": "work_experience",
                "projects": "project",
                "education": "education",
                "tech_skills": "tech_skill",
                "soft_skills": "soft_skill",
                "personality": "personality",
                "other": "other"
            }

            saved_count = 0
            for field, category in category_map.items():
                if data.get(field):
                    self.store.save(
                        text=data[field],
                        category=category,
                        metadata={"source": "conversation"}
                    )
                    saved_count += 1

            if saved_count > 0:
                print(f"\n  💾 Saved {saved_count} info piece(s) to your profile")

        except Exception as e:
            print(f"\n  ⚠️ Could not extract info: {e}")

    def _get_followup(self, messages: list) -> str:
        """Gets a natural follow-up response from LLM"""
        all_messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

        # Add next question if available
        if self.question_index < len(HR_QUESTIONS):
            next_q = HR_QUESTIONS[self.question_index]
            all_messages.append(
                HumanMessage(content=f"[Continue the conversation and then ask: {next_q}]")
            )
            self.question_index += 1

        response = self.llm.invoke(all_messages)
        return response.content

    def run(self, state: AgentState) -> dict:
        """Main profile building conversation loop"""
        print("\n" + "="*50)
        print("📋 PROFILE BUILDING")
        print("="*50)
        print("Let's build your career profile!")
        print("Type 'done' when finished, 'menu' to go back\n")

        messages = list(state.messages)

        # Opening message
        opening = "Hi! I'm here to help build your career profile. I'll ask you a few questions to get to know your background better."
        print(f"\n🤖 Agent: {opening}\n")
        messages.append(AIMessage(content=opening))

        while True:
            # Get follow-up question from LLM
            agent_message = self._get_followup(messages)
            print(f"🤖 Agent: {agent_message}\n")
            messages.append(AIMessage(content=agent_message))

            # Get user input
            user_input = input("You: ").strip()

            if user_input.lower() == "done":
                print("\n✅ Profile session complete! Your information has been saved.")
                return {
                    "messages": messages,
                    "user_input": "menu",
                    "profile_complete": True
                }

            if user_input.lower() == "menu":
                return {
                    "messages": messages,
                    "user_input": "menu"
                }

            # Save to message history
            messages.append(HumanMessage(content=user_input))

            # Extract and save info to Qdrant
            self._extract_and_save(user_input)