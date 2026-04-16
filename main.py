from dotenv import load_dotenv
import yaml
import os
from agents.llm_factory import get_llm

from agents.orchestrator import create_orchestrator
from agents.state import AgentState

load_dotenv()

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():


    config = load_config()
    print(f"LLM: {config['llm']['provider']} — {config['llm']['model']}")

    orchestrator = create_orchestrator(config)
    initial_state = AgentState()
    orchestrator.invoke(initial_state)

if __name__ == "__main__":
    main()