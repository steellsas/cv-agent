from langgraph.graph import StateGraph, END
from agents.state import AgentState
from agents.llm_factory import get_llm

def create_orchestrator(config: dict):
    llm = get_llm(config)
    graph = StateGraph(AgentState)

    # --- Nodes ---

    def greet_user(state: AgentState) -> dict:
        print("\n👋 Welcome to CV Agent!")
        print("Type 'profile' — to build your profile")
        print("Type 'cv'      — to generate a CV")
        print("Type 'quit'    — to exit")
        user_input = input("\nYou: ").strip().lower()
        return {"user_input": user_input}  # grąžinam tik pakeitimus

    def route_decision(state: AgentState) -> str:
        user_input = state.user_input  # Pydantic: state.field, ne state["field"]
        if user_input == "quit":
            return "end"
        elif user_input == "profile":
            return "profile"
        elif user_input == "cv":
            return "cv"
        else:
            return "unknown"

    def handle_unknown(state: AgentState) -> dict:
        print("\n❓ Unknown command. Please type 'profile', 'cv', or 'quit'.")
        user_input = input("\nYou: ").strip().lower()
        return {"user_input": user_input}

    def handle_profile(state: AgentState) -> dict:
        print("\n📋 Profile building — coming soon!")
        user_input = input("\nYou: ").strip().lower()
        return {"user_input": user_input}

    def handle_cv(state: AgentState) -> dict:
        print("\n📄 CV generation — coming soon!")
        user_input = input("\nYou: ").strip().lower()
        return {"user_input": user_input}

    # --- Add nodes ---
    graph.add_node("greet", greet_user)
    graph.add_node("unknown", handle_unknown)
    graph.add_node("profile", handle_profile)
    graph.add_node("cv", handle_cv)

    # --- Entry point ---
    graph.set_entry_point("greet")

    # --- Edges ---
    graph.add_conditional_edges("greet", route_decision, {
        "profile": "profile",
        "cv": "cv",
        "unknown": "unknown",
        "end": END
    })
    graph.add_conditional_edges("unknown", route_decision, {
        "profile": "profile",
        "cv": "cv",
        "unknown": "unknown",
        "end": END
    })
    graph.add_conditional_edges("profile", route_decision, {
        "profile": "profile",
        "cv": "cv",
        "unknown": "unknown",
        "end": END
    })
    graph.add_conditional_edges("cv", route_decision, {
        "profile": "profile",
        "cv": "cv",
        "unknown": "unknown",
        "end": END
    })

    return graph.compile()