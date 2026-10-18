"""llm_comp — шлюз языковых моделей контейнера agent_service."""

from vibecoder.llm_comp.factory import DeploymentMode, StrategyFactory
from vibecoder.llm_comp.gateway import LLMGateway
from vibecoder.llm_comp.models import (
    ChatMessage,
    Completion,
    GenerationParams,
    LLMUnavailableError,
)
from vibecoder.llm_comp.ports import ChatClient
from vibecoder.llm_comp.strategies import (
    CloudLLMStrategy,
    FallbackStrategy,
    LLMStrategy,
    LocalLLMStrategy,
)

__all__ = [
    "ChatClient",
    "ChatMessage",
    "CloudLLMStrategy",
    "Completion",
    "DeploymentMode",
    "FallbackStrategy",
    "GenerationParams",
    "LLMGateway",
    "LLMStrategy",
    "LLMUnavailableError",
    "LocalLLMStrategy",
    "StrategyFactory",
]
