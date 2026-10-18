# vibecoder / llm_comp — шлюз языковых моделей (agent_service)

Прототип компонента llm_comp с паттерном «Стратегия». Только стандартная библиотека Python 3.12.

Команды выполняются из папки `code/`:

    python -m vibecoder.demo                                  # подмена стратегий на лету
    python -m unittest discover -s vibecoder/tests -t .       # 19 тестов
    pip install -r requirements-dev.txt && mypy vibecoder     # проверка типов (strict, mypy.ini)

Файлы пакета `vibecoder/llm_comp/`:
- `models.py` — ChatMessage, GenerationParams, Completion, LLMUnavailableError
- `ports.py` — порт ChatClient (Protocol) к llm_server (№45) и llm_provider (№46)
- `strategies.py` — LLMStrategy (ABC), LocalLLMStrategy, CloudLLMStrategy, FallbackStrategy
- `gateway.py` — контекст LLMGateway
- `factory.py` — DeploymentMode и StrategyFactory (закрытый контур / SaaS)
- `prototype_adapters.py` — FakeChatClient (имитация модели без сети)
