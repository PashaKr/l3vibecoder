"""Порт к языковым моделям.

В сервисе порт реализует OpenAI-совместимый HTTP-клиент: к llm_server
(связь L2 №45) и к llm_provider (связь L2 №46). В прототипе — FakeChatClient.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

from vibecoder.llm_comp.models import ChatMessage, GenerationParams


class ChatClient(Protocol):
    """Клиент OpenAI-совместимого API чата.

    При недоступности провайдера выбрасывает LLMUnavailableError.
    """

    @property
    def provider(self) -> str:
        """Имя провайдера, например vllm или gigachat."""
        ...

    def complete(
        self, model: str, messages: list[ChatMessage], params: GenerationParams
    ) -> str:
        """Возвращает ответ целиком."""
        ...

    def stream(
        self, model: str, messages: list[ChatMessage], params: GenerationParams
    ) -> Iterator[str]:
        """Возвращает ответ потоком токенов."""
        ...
