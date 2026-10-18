"""Адаптеры прототипа: имитация языковых моделей без сети."""

from __future__ import annotations

from collections.abc import Iterator

from vibecoder.llm_comp.models import (
    ChatMessage,
    GenerationParams,
    LLMUnavailableError,
)


class FakeChatClient:
    """Имитация llm_server или llm_provider; реализует порт ChatClient.

    available=False — сервис «упал»; fail_after_tokens=N — поток
    обрывается после N токенов (сбой посреди ответа).
    """

    def __init__(
        self,
        provider: str,
        available: bool = True,
        fail_after_tokens: int | None = None,
    ) -> None:
        self._provider: str = provider
        self._available: bool = available
        self._fail_after_tokens: int | None = fail_after_tokens

    @property
    def provider(self) -> str:
        return self._provider

    def set_available(self, available: bool) -> None:
        """Включает или «роняет» сервис."""
        self._available = available

    def complete(
        self, model: str, messages: list[ChatMessage], params: GenerationParams
    ) -> str:
        if not self._available:
            raise LLMUnavailableError(self._provider, "сервис недоступен")
        return self._reply(model, messages, params)

    def stream(
        self, model: str, messages: list[ChatMessage], params: GenerationParams
    ) -> Iterator[str]:
        if not self._available:
            raise LLMUnavailableError(self._provider, "сервис недоступен")
        for index, word in enumerate(self._reply(model, messages, params).split(" ")):
            if self._fail_after_tokens is not None and index >= self._fail_after_tokens:
                raise LLMUnavailableError(self._provider, "соединение оборвано")
            yield word + " "

    def _reply(
        self, model: str, messages: list[ChatMessage], params: GenerationParams
    ) -> str:
        question = next(
            (m.content for m in reversed(messages) if m.role == "user"), ""
        )
        words = f"[{self._provider}/{model}] Ответ на вопрос: {question}".split(" ")
        return " ".join(words[: params.max_tokens])
