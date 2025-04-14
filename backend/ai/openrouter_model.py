from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from backend.config import get_settings

def create_ai_agent() -> Agent:
    """
    Create a Pydantic AI agent using OpenRouter as the provider.
    """
    settings = get_settings()
    
    model = OpenAIModel(
        settings.ai_model,
        provider=OpenAIProvider(
            base_url='https://openrouter.ai/api/v1',
            api_key=settings.openrouter_api_key,
        ),
    )
    
    return Agent(model) 