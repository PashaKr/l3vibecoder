"""Демонстрация паттерна «Стратегия» в llm_comp: подмена моделей на лету.

Запуск из папки code/:  python -m vibecoder.demo
"""

from __future__ import annotations

import logging

from vibecoder.llm_comp import (
    ChatMessage,
    DeploymentMode,
    GenerationParams,
    LLMGateway,
    LLMUnavailableError,
    LocalLLMStrategy,
    StrategyFactory,
)
from vibecoder.llm_comp.prototype_adapters import FakeChatClient


def show_stream(gateway: LLMGateway, messages: list[ChatMessage], params: GenerationParams) -> None:
    """Печатает поток токенов; при обрыве сообщает об ошибке."""
    tokens = gateway.stream(messages, params)
    try:
        first = next(tokens, "")  # переключение на резерв происходит здесь
        print(f"  поток: {first}", end="", flush=True)
        for token in tokens:
            print(token, end="", flush=True)
        print()
    except LLMUnavailableError as error:
        print(f"\n  ! поток оборван: {error}")


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="  [журнал] %(message)s")

    vllm = FakeChatClient("vllm")
    gigachat = FakeChatClient("gigachat")
    messages = [
        ChatMessage("system", "Ты маркетинговый ассистент компании."),
        ChatMessage("user", "Какие у нас кейсы в нефтегазе?"),
    ]
    params = GenerationParams(temperature=0.2, max_tokens=64)

    print("1. SaaS: FallbackStrategy(local -> cloud), локальная модель работает")
    gateway = LLMGateway(StrategyFactory.for_mode(DeploymentMode.SAAS, vllm, gigachat))
    print(f"  стратегия: {gateway.strategy_name}")
    completion = gateway.generate(messages, params)
    print(f"  провайдер: {completion.provider}, модель: {completion.model}")
    print(f"  ответ: {completion.text}")

    print("\n2. Локальный llm_server упал — резерв срабатывает без изменений в шлюзе")
    vllm.set_available(False)
    completion = gateway.generate(messages, params)
    print(f"  провайдер: {completion.provider}, модель: {completion.model}")
    show_stream(gateway, messages, params)

    print("\n3. Закрытый контур: подменяем стратегию на лету, облако не используется")
    vllm.set_available(True)
    gateway.set_strategy(StrategyFactory.for_mode(DeploymentMode.CLOSED_CONTOUR, vllm))
    print(f"  стратегия: {gateway.strategy_name}")
    show_stream(gateway, messages, params)

    print("\n4. Закрытый контур, llm_server недоступен — резерва нет, ошибка видна сразу")
    vllm.set_available(False)
    try:
        gateway.generate(messages, params)
    except LLMUnavailableError as error:
        print(f"  ! {error}")

    print("\n5. Сбой посреди потока: резерв не подменяет уже показанный текст")
    flaky = FakeChatClient("vllm", fail_after_tokens=3)
    gateway.set_strategy(
        StrategyFactory.for_mode(DeploymentMode.SAAS, flaky, gigachat)
    )
    show_stream(gateway, messages, params)

    print("\n6. Проверка запроса в шлюзе: пустой список сообщений")
    gateway.set_strategy(LocalLLMStrategy(FakeChatClient("vllm"), "qwen2.5-7b-instruct"))
    try:
        gateway.generate([], params)
    except ValueError as error:
        print(f"  ! {error}")


if __name__ == "__main__":
    main()
