"""Выбор стратегии по режиму поставки платформы."""

from __future__ import annotations

from enum import Enum

from vibecoder.llm_comp.ports import ChatClient
from vibecoder.llm_comp.strategies import (
    CloudLLMStrategy,
    FallbackStrategy,
    LLMStrategy,
    LocalLLMStrategy,
)


class DeploymentMode(Enum):
    """Режим поставки платформы."""

    CLOSED_CONTOUR = "closed_contour"
    SAAS = "saas"


class StrategyFactory:
    """Собирает стратегию из конфигурации при старте сервиса."""

    @staticmethod
    def for_mode(
        mode: DeploymentMode,
        local_client: ChatClient,
        cloud_client: ChatClient | None = None,
        local_model: str = "qwen2.5-7b-instruct",
        cloud_model: str = "GigaChat-Pro",
    ) -> LLMStrategy:
        """Закрытый контур — только локальная модель (C-01);
        SaaS — локальная модель с облачным резервом (R-04)."""
        local = LocalLLMStrategy(local_client, local_model)
        if mode is DeploymentMode.CLOSED_CONTOUR:
            return local
        if cloud_client is None:
            raise ValueError("Для режима SaaS нужен клиент облачной модели")
        return FallbackStrategy(local, CloudLLMStrategy(cloud_client, cloud_model))
