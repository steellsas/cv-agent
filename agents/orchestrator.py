from langgraph.graph import StateGraph, END
from agents.state import AgentState
from agents.llm_factory import get_llm


from tools.linkedin_parser import LinkedInParser
from tools.github_scraper import GitHubScraper

from agents.cv_agent import CVAgent
from agents.profile_coordinator import ProfileCoordinator
from tools.cv_parser import CVParser
from agents.profile_builder import ProfileBuilder

def create_orchestrator(config: dict):
    llm = get_llm(config)
    # profile_agent = ProfileAgent(config)
    linkedin_parser = LinkedInParser(config)
    github_scraper = GitHubScraper(config)
    # fact_checker = FactChecker(config)
    cv_agent = CVAgent(config)
    profile_coordinator = ProfileCoordinator(config)
    profile_builder = ProfileBuilder(config)
    cv_parser = CVParser(config)
    graph = StateGraph(AgentState)


    def greet_user(state: AgentState) -> dict:
        print("\n" + "="*50)
        print("👋 Welcome to CV Agent!")
        print("="*50)
        print("Type 'profile' — to build your profile")
        print("Type 'linkedin' — to exract form linkedin data")
        print("Type 'profile'    — to build/update your profile")
        print("Type 'github'    — to import GitHub projects")
        print("Type 'cv'      — to generate a CV")
        print("Type 'quit'    — to exit")
        print("Type 'reset'     — to clear and restart profile")
        print("Type 'factcheck'  — to review and confirm your profile")
        print("Type 'masterprofile' — to build Master Profile")
        # greet_user:
        print("Type 'upload'     — to upload your CV PDF")
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
        elif user_input == "factcheck":
            return "factcheck"
        elif user_input == "reset":
            return "reset"
        elif user_input == "masterprofile":
            return "masterprofile"
        elif user_input == "upload":
            return "upload"
        else:
            return "unknown"

    def handle_unknown(state: AgentState) -> dict:
        print("\n❓ Unknown command. Please type 'profile', 'cv', or 'quit'.")
        user_input = input("\nYou: ").strip().lower()
        return {"user_input": user_input}

    def handle_profile(state: AgentState) -> dict:
        return profile_agent.run(state)
    
    def handle_upload_cv(state: AgentState) -> dict:
        print("\n📄 CV PDF Upload")
        print("Enter path to your CV PDF file:")
        pdf_path = input("Path: ").strip()
        if not pdf_path:
            pdf_path = "data/cv.pdf"

        success = cv_parser.parse(pdf_path)
        if success:
            cv_parser.store.display_summary()
        input("\nPress Enter to continue...")
        return {"user_input": "menu"}

    # def handle_cv(state: AgentState) -> dict:
    #     print("\n📄 CV generation — coming soon!")
    #     user_input = input("\nYou: ").strip().lower()
    #     return {"user_input": user_input}

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
    
    def handle_factcheck(state: AgentState) -> dict:
        fact_checker.run()
        return {"user_input": "menu"}

    def handle_masterprofile(state: AgentState) -> dict:
        profile_coordinator.build()
        return {"user_input": "menu"}
    
    def handle_cv(state: AgentState) -> dict:
        print("\n📋 Loading Master Profile...")
        master_profile = profile_coordinator.get_master_profile()

        if not master_profile:
            print("❌ No Master Profile found!")
            print("Please run 'masterprofile' first.")
            input("\nPress Enter to continue...")
            return {"user_input": "menu"}

        print("\n📄 Paste job posting below.")
        print("Type 'END' on a new line when done.\n")

        lines = []
        while True:
            line = input()
            if line.strip().upper() == "END":
                break
            lines.append(line)

        job_posting = "\n".join(lines).strip()
        if not job_posting:
            print("❌ No job posting provided.")
            return {"user_input": "menu"}

        # B variantas — Coordinator gauna match prieš CV agentą
        job_match = profile_coordinator.get_strengths_for_job(job_posting)
        if not job_match:
            print("❌ Could not analyze job match.")
            return {"user_input": "menu"}

        return cv_agent.run(state, master_profile, job_match)
    
    def handle_profile(state: AgentState) -> dict:
        profile_builder.run()
        return {"user_input": "menu"}
    
    graph.add_node("profile", handle_profile)
    graph.add_node("masterprofile", handle_masterprofile)
    graph.add_node("upload", handle_upload_cv)
    
    graph.add_node("factcheck", handle_factcheck)
    graph.add_node("reset", handle_reset)
    graph.add_node("github", handle_github)
    graph.add_node("linkedin", handle_linkedin)

    graph.add_node("greet", greet_user)
    graph.add_node("unknown", handle_unknown)
    graph.add_node("cv", handle_cv)

    graph.set_entry_point("greet")

    edges = {
        "profile": "profile",
        "linkedin": "linkedin",
        "github": "github",
        "factcheck": "factcheck",
        "reset": "reset", 
        "masterprofile": "masterprofile",
        "cv": "cv",
        "unknown": "unknown",
        "upload": "upload",
        "end": END
    }

    graph.add_conditional_edges("greet", route_decision, edges)
    graph.add_conditional_edges("unknown", route_decision, edges)
    graph.add_conditional_edges("profile", route_decision, edges)
    graph.add_conditional_edges("cv", route_decision, edges)

    return graph.compile()