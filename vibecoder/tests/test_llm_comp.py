"""Тесты компонента llm_comp.

Запуск из папки code/:  python -m unittest discover -s vibecoder/tests -t .
"""

from __future__ import annotations

import logging
import unittest
from collections.abc import Iterator

from vibecoder.llm_comp import (
    ChatMessage,
    CloudLLMStrategy,
    Completion,
    DeploymentMode,
    FallbackStrategy,
    GenerationParams,
    LLMGateway,
    LLMStrategy,
    LLMUnavailableError,
    LocalLLMStrategy,
    StrategyFactory,
)
from vibecoder.llm_comp.prototype_adapters import FakeChatClient

MESSAGES = [ChatMessage("user", "Привет")]
PARAMS = GenerationParams()


class StubStrategy(LLMStrategy):
    """Тестовая стратегия: возвращает заранее заданный текст."""

    def __init__(self, label: str) -> None:
        self._label = label
        self.calls = 0

    @property
    def name(self) -> str:
        return self._label

    def generate(self, messages: list[ChatMessage], params: GenerationParams) -> Completion:
        self.calls += 1
        return Completion(text=self._label, model="stub", provider=self._label, latency_ms=0)

    def stream(self, messages: list[ChatMessage], params: GenerationParams) -> Iterator[str]:
        self.calls += 1
        return iter([self._label])


class GatewayTests(unittest.TestCase):
    def test_delegates_to_strategy(self) -> None:
        strategy = StubStrategy("a")
        gateway = LLMGateway(strategy)
        self.assertEqual(gateway.generate(MESSAGES, PARAMS).text, "a")
        self.assertEqual(list(gateway.stream(MESSAGES, PARAMS)), ["a"])
        self.assertEqual(strategy.calls, 2)

    def test_set_strategy_on_the_fly(self) -> None:
        gateway = LLMGateway(StubStrategy("a"))
        gateway.set_strategy(StubStrategy("b"))
        self.assertEqual(gateway.strategy_name, "b")
        self.assertEqual(gateway.generate(MESSAGES, PARAMS).text, "b")

    def test_rejects_empty_messages(self) -> None:
        gateway = LLMGateway(StubStrategy("a"))
        with self.assertRaises(ValueError):
            gateway.generate([], PARAMS)

    def test_rejects_token_limit(self) -> None:
        gateway = LLMGateway(StubStrategy("a"), max_tokens_limit=100)
        with self.assertRaises(ValueError):
            gateway.generate(MESSAGES, GenerationParams(max_tokens=101))
        with self.assertRaises(ValueError):
            gateway.generate(MESSAGES, GenerationParams(max_tokens=0))

    def test_rejects_temperature(self) -> None:
        gateway = LLMGateway(StubStrategy("a"))
        with self.assertRaises(ValueError):
            gateway.generate(MESSAGES, GenerationParams(temperature=3.0))

    def test_stream_validates_before_streaming(self) -> None:
        strategy = StubStrategy("a")
        gateway = LLMGateway(strategy)
        with self.assertRaises(ValueError):
            gateway.stream([], PARAMS)
        self.assertEqual(strategy.calls, 0)

    def test_rejects_bad_limit(self) -> None:
        with self.assertRaises(ValueError):
            LLMGateway(StubStrategy("a"), max_tokens_limit=0)


class ClientStrategyTests(unittest.TestCase):
    def test_local_completion_fields(self) -> None:
        strategy = LocalLLMStrategy(FakeChatClient("vllm"), "qwen")
        completion = strategy.generate(MESSAGES, PARAMS)
        self.assertEqual((completion.provider, completion.model), ("vllm", "qwen"))
        self.assertIn("Привет", completion.text)
        self.assertGreaterEqual(completion.latency_ms, 0)

    def test_cloud_stream(self) -> None:
        strategy = CloudLLMStrategy(FakeChatClient("gigachat"), "GigaChat-Pro")
        text = "".join(strategy.stream(MESSAGES, PARAMS))
        self.assertIn("gigachat/GigaChat-Pro", text)

    def test_unavailable_raises(self) -> None:
        strategy = LocalLLMStrategy(FakeChatClient("vllm", available=False), "qwen")
        with self.assertRaises(LLMUnavailableError):
            strategy.generate(MESSAGES, PARAMS)
        with self.assertRaises(LLMUnavailableError):
            list(strategy.stream(MESSAGES, PARAMS))


class FallbackTests(unittest.TestCase):
    def setUp(self) -> None:
        logging.disable(logging.CRITICAL)
        self.local = FakeChatClient("vllm")
        self.cloud = FakeChatClient("gigachat")
        self.strategy = FallbackStrategy(
            LocalLLMStrategy(self.local, "qwen"), CloudLLMStrategy(self.cloud, "giga")
        )

    def tearDown(self) -> None:
        logging.disable(logging.NOTSET)

    def test_name(self) -> None:
        self.assertEqual(self.strategy.name, "fallback(local -> cloud)")

    def test_uses_primary_when_available(self) -> None:
        self.assertEqual(self.strategy.generate(MESSAGES, PARAMS).provider, "vllm")

    def test_switches_to_secondary(self) -> None:
        self.local.set_available(False)
        self.assertEqual(self.strategy.generate(MESSAGES, PARAMS).provider, "gigachat")

    def test_stream_switches_before_first_token(self) -> None:
        self.local.set_available(False)
        text = "".join(self.strategy.stream(MESSAGES, PARAMS))
        self.assertIn("gigachat", text)

    def test_stream_reraises_after_tokens(self) -> None:
        flaky = FallbackStrategy(
            LocalLLMStrategy(FakeChatClient("vllm", fail_after_tokens=2), "qwen"),
            CloudLLMStrategy(self.cloud, "giga"),
        )
        received: list[str] = []
        with self.assertRaises(LLMUnavailableError):
            for token in flaky.stream(MESSAGES, PARAMS):
                received.append(token)
        self.assertEqual(len(received), 2)

    def test_both_unavailable(self) -> None:
        self.local.set_available(False)
        self.cloud.set_available(False)
        with self.assertRaises(LLMUnavailableError):
            self.strategy.generate(MESSAGES, PARAMS)


class FactoryTests(unittest.TestCase):
    def test_closed_contour_is_local_only(self) -> None:
        strategy = StrategyFactory.for_mode(
            DeploymentMode.CLOSED_CONTOUR, FakeChatClient("vllm"), FakeChatClient("gigachat")
        )
        self.assertIsInstance(strategy, LocalLLMStrategy)

    def test_saas_is_fallback(self) -> None:
        strategy = StrategyFactory.for_mode(
            DeploymentMode.SAAS, FakeChatClient("vllm"), FakeChatClient("gigachat")
        )
        self.assertIsInstance(strategy, FallbackStrategy)

    def test_saas_requires_cloud_client(self) -> None:
        with self.assertRaises(ValueError):
            StrategyFactory.for_mode(DeploymentMode.SAAS, FakeChatClient("vllm"))


if __name__ == "__main__":
    unittest.main()
