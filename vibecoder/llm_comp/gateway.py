"""Контекст стратегии: шлюз языковых моделей для orchestrator_comp."""

from __future__ import annotations

from collections.abc import Iterator

from vibecoder.llm_comp.models import ChatMessage, Completion, GenerationParams
from vibecoder.llm_comp.strategies import LLMStrategy


class LLMGateway:
    """Единая точка вызова модели; стратегию можно сменить на лету (S-05)."""

    def __init__(self, strategy: LLMStrategy, max_tokens_limit: int = 4096) -> None:
        if max_tokens_limit <= 0:
            raise ValueError("Лимит токенов должен быть положительным")
        self._strategy: LLMStrategy = strategy
        self._max_tokens_limit: int = max_tokens_limit

    @property
    def strategy_name(self) -> str:
        """Имя текущей стратегии."""
        return self._strategy.name

    def set_strategy(self, strategy: LLMStrategy) -> None:
        """Подменяет стратегию без пересоздания шлюза и оркестратора."""
        self._strategy = strategy

    def generate(
        self, messages: list[ChatMessage], params: GenerationParams
    ) -> Completion:
        """Ответ целиком (эндпоинт /complete для generation_service)."""
        self._validate(messages, params)
        return self._strategy.generate(messages, params)

    def stream(
        self, messages: list[ChatMessage], params: GenerationParams
    ) -> Iterator[str]:
        """Поток токенов (чат через WebSocket). Проверка — сразу, до потока."""
        self._validate(messages, params)
        return self._strategy.stream(messages, params)

    def _validate(self, messages: list[ChatMessage], params: GenerationParams) -> None:
        if not messages:
            raise ValueError("Список сообщений пуст")
        if not 0 < params.max_tokens <= self._max_tokens_limit:
            raise ValueError(
                f"max_tokens должен быть в диапазоне 1..{self._max_tokens_limit}"
            )
        if not 0.0 <= params.temperature <= 2.0:
            raise ValueError("temperature должна быть в диапазоне 0..2")
