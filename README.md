# Flight Delay MLOps — ВКР

Магистерская ВКР: «Разработка модели искусственного интеллекта для прогнозирования задержек авиарейсов и классификации их причин».

Главный результат — **выстроенный ML-процесс** (data → DVC → features → training → MLflow → FastAPI → monitoring → feedback loop), а не отдельная модель. Подробный контекст и правила работы — в [CLAUDE.md](./CLAUDE.md).

---

## Стек

| Слой | Инструмент |
|---|---|
| Версионирование кода | git |
| Версионирование данных | DVC |
| Трекинг экспериментов | MLflow |
| Инференс | FastAPI |
| Контейнеризация | Docker + docker-compose |
| Мониторинг | Prometheus + Grafana |
| ML-библиотеки | scikit-learn, XGBoost, LightGBM, CatBoost, Optuna, SHAP |
| Качество кода | ruff, mypy, pytest, pre-commit |

## Данные

Синтетический датасет `Flight Delays RU 2023–2025` (220 000 рейсов, 22 аэропорта РФ, 11 авиакомпаний). Полностью воспроизводим из `generate_dataset.py` (seed=42).

См. [DATASET_CARD.md](./DATASET_CARD.md), [DATA_DICTIONARY.md](./DATA_DICTIONARY.md), [DATA_QUALITY_REPORT.md](./DATA_QUALITY_REPORT.md).

## Структура

```
src/data/        # ingest, annotate, split
src/features/    # feature engineering, feature sets
src/models/      # train, tune, evaluate, ensemble, registry
src/api/         # FastAPI inference
src/monitoring/  # logger, metrics, feedback loop
notebooks/       # EDA и главы 1-2 ВКР
tests/           # pytest
docker/          # Dockerfiles
reports/         # фигуры и сводка экспериментов
```

## Quick start

> Окружение и Docker-сервисы будут описаны по мере прохождения этапов проекта.

```bash
# clone
git clone https://github.com/GeorgeSard/VKR2.git
cd VKR2

# (далее — после bootstrap зависимостей)
```

## Этапы

См. план в `CLAUDE.md`, раздел 7. Текущий — **Этап 0: bootstrap**.
