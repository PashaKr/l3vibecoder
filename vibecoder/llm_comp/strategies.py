"""Стратегии выбора языковой модели (паттерн «Стратегия»)."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator

from vibecoder.llm_comp.models import (
    ChatMessage,
    Completion,
    GenerationParams,
    LLMUnavailableError,
)
from vibecoder.llm_comp.ports import ChatClient

logger = logging.getLogger(__name__)


class LLMStrategy(ABC):
    """Интерфейс стратегии: как и какой моделью сгенерировать ответ."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Короткое имя стратегии для логов и метрик."""

    @abstractmethod
    def generate(
        self, messages: list[ChatMessage], params: GenerationParams
    ) -> Completion:
        """Генерирует ответ целиком."""

    @abstractmethod
    def stream(
        self, messages: list[ChatMessage], params: GenerationParams
    ) -> Iterator[str]:
        """Генерирует ответ потоком токенов."""


class LocalLLMStrategy(LLMStrategy):
    """Локальная модель на llm_server (vLLM) внутри контура заказчика."""

    def __init__(self, client: ChatClient, model: str) -> None:
        self._client: ChatClient = client
        self._model: str = model

    @property
    def name(self) -> str:
        return "local"

    def generate(
        self, messages: list[ChatMessage], params: GenerationParams
    ) -> Completion:
        started = time.perf_counter()
        text = self._client.complete(self._model, messages, params)
        latency_ms = int((time.perf_counter() - started) * 1000)
        return Completion(
            text=text,
            model=self._model,
            provider=self._client.provider,
            latency_ms=latency_ms,
        )

    def stream(
        self, messages: list[ChatMessage], params: GenerationParams
    ) -> Iterator[str]:
        return self._client.stream(self._model, messages, params)


class CloudLLMStrategy(LLMStrategy):
    """Облачная модель llm_provider (GigaChat / YandexGPT), режим SaaS."""

    def __init__(self, client: ChatClient, model: str) -> None:
        self._client: ChatClient = client
        self._model: str = model

    @property
    def name(self) -> str:
        return "cloud"

    def generate(
        self, messages: list[ChatMessage], params: GenerationParams
    ) -> Completion:
        started = time.perf_counter()
        text = self._client.complete(self._model, messages, params)
        latency_ms = int((time.perf_counter() - started) * 1000)
        return Completion(
            text=text,
            model=self._model,
            provider=self._client.provider,
            latency_ms=latency_ms,
        )

    def stream(
        self, messages: list[ChatMessage], params: GenerationParams
    ) -> Iterator[str]:
        return self._client.stream(self._model, messages, params)


class FallbackStrategy(LLMStrategy):
    """Стратегия из двух стратегий: основная и резервная (R-04).

    Сама реализует LLMStrategy, поэтому шлюз не знает о резервировании.
    """

    def __init__(self, primary: LLMStrategy, secondary: LLMStrategy) -> None:
        self._primary: LLMStrategy = primary
        self._secondary: LLMStrategy = secondary

    @property
    def name(self) -> str:
        return f"fallback({self._primary.name} -> {self._secondary.name})"

    def generate(
        self, messages: list[ChatMessage], params: GenerationParams
    ) -> Completion:
        try:
            return self._primary.generate(messages, params)
        except LLMUnavailableError as error:
            logger.warning(
                "Основная модель недоступна (%s), переключаюсь на %s",
                error,
                self._secondary.name,
            )
            return self._secondary.generate(messages, params)

    def stream(
        self, messages: list[ChatMessage], params: GenerationParams
    ) -> Iterator[str]:
        """Переключается на резерв, только если не отдан ни один токен.

        Если сбой случился посреди ответа, ошибка пробрасывается:
        уже показанный пользователю текст нельзя «заменить» ответом
        другой модели, оркестратор сообщит клиенту об обрыве.
        """
        emitted = False
        try:
            for token in self._primary.stream(messages, params):
                emitted = True
                yield token
            return
        except LLMUnavailableError as error:
            if emitted:
                raise
            logger.warning(
                "Основная модель недоступна до первого токена (%s), переключаюсь на %s",
                error,
                self._secondary.name,
            )
        yield from self._secondary.stream(messages, params)
