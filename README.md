# Call Processing Subsystem

Локальная микросервисная подсистема для обработки телефонных обращений:

1. прием аудиофайла;
2. транскрибация и диаризация;
3. антиспам и маршрутизация;
4. извлечение сущностей;
5. создание тикета;
6. передача результата во внешнюю тикет-систему.

## Основные части

- `services/transcription` - распознавание речи
- `services/router` - классификация, антиспам, маршрутизация
- `services/entity_extraction` - NER
- `services/ticket_creation` - генерация карточки и регистрация тикета
- `services/orchestrator` - точка входа и координация пайплайна

Общий gRPC-контракт: `proto/call_processing.proto`

## Быстрый запуск

- Docker: `docker compose up --build`
- Локальный запуск сервисов: `scripts/run_all.sh`

Основные env-файлы лежат в `configs/`.

## Проверка

- функциональные тесты: `tests/run_functional_tests.py`
- интеграционные тесты оркестратора: `services/orchestrator/tests/orchestrator_integration_test.go`
- сравнение с ручной классификацией: `tests/evaluate_ab_test.py`

```bash
cd services/orchestrator && go test ./...
python3 -m unittest discover -s tests -p 'test_*.py'
```

## Безопасность

- JWT-аутентификация и роли `admin` / `operator`
- аудит действий через `/api/v1/audit/events`
- локальная обработка данных без облачных LLM по умолчанию
- ограничение CORS и хранение чувствительных настроек в env

## Документация по развертыванию

- Linux: `deploy/linux/DEPLOY.md`
- Docker: `deploy/docker/DEPLOY.md`
