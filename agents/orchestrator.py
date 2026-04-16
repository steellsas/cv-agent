from langgraph.graph import StateGraph, END
from agents.state import AgentState
from agents.llm_factory import get_llm
from agents.profile_agent import ProfileAgent

from tools.linkedin_parser import LinkedInParser
from tools.github_scraper import GitHubScraper
from agents.cv_agent import CVAgent

def create_orchestrator(config: dict):
    llm = get_llm(config)
    profile_agent = ProfileAgent(config)
    linkedin_parser = LinkedInParser(config)
    github_scraper = GitHubScraper(config)
    cv_agent = CVAgent(config)
    graph = StateGraph(AgentState)

    def greet_user(state: AgentState) -> dict:
        print("\n" + "="*50)
        print("👋 Welcome to CV Agent!")
        print("="*50)
        print("Type 'profile' — to build your profile")
        print("Type 'linkedin' — to exract form linkedin data")
        print("Type 'github'    — to import GitHub projects")
        print("Type 'cv'      — to generate a CV")
        print("Type 'quit'    — to exit")
        print("Type 'reset'     — to clear and restart profile")
        user_input = input("\nYou: ").strip().lower()
        return {"user_input": user_input}

    def route_decision(state: AgentState) -> str:
        user_input = state.user_input
        if user_input == "quit":
            return "end"
        elif user_input == "profile":
            return "profile"
        # Route decision papildyk:
        elif user_input == "linkedin":
            return "linkedin"
        elif user_input == "github":
            return "github"
        elif user_input == "cv":
            return "cv"
        elif user_input == "reset":
            return "reset"
        else:
            return "unknown"

    def handle_unknown(state: AgentState) -> dict:
        print("\n❓ Unknown command. Please type 'profile', 'cv', or 'quit'.")
        user_input = input("\nYou: ").strip().lower()
        return {"user_input": user_input}

    def handle_profile(state: AgentState) -> dict:
        return profile_agent.run(state)

    def handle_cv(state: AgentState) -> dict:
        print("\n📄 CV generation — coming soon!")
        user_input = input("\nYou: ").strip().lower()
        return {"user_input": user_input}

    def handle_linkedin(state: AgentState) -> dict:
        folder = input("\n📂 Enter LinkedIn data folder path (default: data/linkedin): ").strip()
        if not folder:
            folder = "data/linkedin"
        linkedin_parser.parse(folder)
        input("\nPress Enter to continue...")
        return {"user_input": "menu"}
    
    def handle_github(state: AgentState) -> dict:
        github_scraper.scrape()
        input("\nPress Enter to continue...")
        return {"user_input": "menu"}
    
    def handle_reset(state: AgentState) -> dict:
        confirm = input("\n⚠️  This will delete ALL profile data. Type 'yes' to confirm: ").strip()
        if confirm == "yes":
            from memory.vector_store import VectorStore
            store = VectorStore(config)
            store.clear_collection()
            print("✅ Profile cleared! Please re-import LinkedIn and GitHub data.")
        return {"user_input": "menu"}
    
    def handle_cv(state: AgentState) -> dict:
        return cv_agent.run(state)
    


    

    graph.add_node("reset", handle_reset)
    graph.add_node("github", handle_github)
    graph.add_node("linkedin", handle_linkedin)

    graph.add_node("greet", greet_user)
    graph.add_node("unknown", handle_unknown)
    graph.add_node("profile", handle_profile)
    graph.add_node("cv", handle_cv)

    graph.set_entry_point("greet")

    edges = {
        "profile": "profile",
        "linkedin": "linkedin",
        "github": "github",
        "reset": "reset", 
        "cv": "cv",
        "unknown": "unknown",
        "end": END
    }

    graph.add_conditional_edges("greet", route_decision, edges)
    graph.add_conditional_edges("unknown", route_decision, edges)
    graph.add_conditional_edges("profile", route_decision, edges)
    graph.add_conditional_edges("cv", route_decision, edges)

    return graph.compile()