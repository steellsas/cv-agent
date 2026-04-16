from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama


def get_llm(config: dict):
    """
    Returns LLM instance based on config.yaml settings.
    Switch provider by changing config.yaml — no code changes needed.
    """
    provider = config["llm"]["provider"]
    model = config["llm"]["model"]
    temperature = config["llm"]["temperature"]

    if provider == "openai":
        return ChatOpenAI(model=model, temperature=temperature)
    
    elif provider == "ollama":
        return ChatOllama(model=model, temperature=temperature)
    
    elif provider == "anthropic":
        return ChatAnthropic(model=model, temperature=temperature)
    
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")