# intentql/llm_integration.py
from abc import ABC, abstractmethod

class BaseLLMClient(ABC):
    @abstractmethod
    def generate_dsl(self, system_prompt: str, question: str) -> dict:
        """Generates the DSL JSON by calling the LLM."""
        pass

    @abstractmethod
    def interpret_results(self, question: str, results: list) -> str:
        """Generates a natural language response from query results."""
        pass

class OpenAIClient(BaseLLMClient):
    """(Mock) Client for interacting with OpenAI models."""
    def __init__(self, api_key: str, model: str = "gpt-4-turbo"):
        self.api_key = api_key
        self.model = model
        print("Initialized OpenAI client (mock).")

    def generate_dsl(self, system_prompt: str, question: str) -> dict:
        print("--- LLM generating DSL (mock) ---")
        # Mocked response for "How many records are there?"
        return {
            "dataset": "records",
            "metrics": [{"agg": "COUNT", "field": "*", "alias": "record_count"}],
            "dimensions": [],
            "filters": [],
            "limit": 1
        }

    def interpret_results(self, question: str, results: list) -> str:
        print("--- LLM interpreting results (mock) ---")
        return f"Based on your question '{question}', the answer is: {results}"


def get_llm_client(config: dict) -> BaseLLMClient:
    """Factory function to get an LLM client based on config."""
    provider = config.get("provider")
    if provider == "openai":
        return OpenAIClient(api_key=config.get("api_key"))
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
