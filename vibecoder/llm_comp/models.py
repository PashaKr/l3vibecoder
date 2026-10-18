"""DTO и исключение компонента llm_comp.

Это внутренние модели контейнера agent_service: по сети между
контейнерами они не передаются, поэтому в shared/dto.py их нет.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """Одна реплика в запросе к языковой модели."""

    role: Role
    content: str


@dataclass(frozen=True, slots=True)
class GenerationParams:
    """Параметры генерации."""

    temperature: float = 0.2
    max_tokens: int = 1024


@dataclass(frozen=True, slots=True)
class Completion:
    """Готовый ответ модели с указанием, кто его сгенерировал."""

    text: str
    model: str
    provider: str
    latency_ms: int


class LLMUnavailableError(Exception):
    """Провайдер модели недоступен: сеть, таймаут, отказ сервиса."""

    def __init__(self, provider: str, reason: str) -> None:
        super().__init__(f"{provider}: {reason}")
        self.provider: str = provider
