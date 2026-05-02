# Session Handoff — flight-delay-mlops

> Файл для быстрого входа в контекст следующей сессии Claude.
> Источник правды для статуса проекта между сессиями.
> При старте новой сессии скажи Claude: «прочитай HANDOFF.md» — этого
> вместе с CLAUDE.md и memory будет достаточно, чтобы продолжить с
> того же места без `/compact`.

**Последнее обновление:** 2026-05-02 (после Run #7)
**HEAD git:** `94174e5` (`origin/main`, всё запушено)
**Текущая ветка:** `main`

---

## Где мы по плану CLAUDE.md §7

| Этап | Статус | Что сделано |
|---|---|---|
| 0. Bootstrap | ✅ | `8c4be93` — структура, pyproject, .gitignore, pre-commit |
| 1. Глава 1 | ⛔ skip | вне scope (см. memory `feedback_scope_programmatic_only_vkr2`) |
| 2. Глава 2 | ⛔ skip | вне scope |
| 3. Данные + DVC | ✅ | `0da22bc` — dvc init, локальный remote, ingest+split pipeline, params.yaml |
| 4. Feature engineering | ✅ (для 1й задачи) | в `6f28a3b` — 3 feature sets: basic / extended / with_weather |
| 5. DVC pipeline | ⏳ partial | есть стадии ingest+split в dvc.yaml; train/evaluate как DVC stages — НЕ добавлены, train запускается напрямую |
| 6. MLflow + baseline | ✅ | `6f28a3b` — train.py с MLflow tracking, file backend в `mlruns/` |
| 7. Научные эксперименты | 🔄 в процессе | 7 runs готовы (см. ниже), серия по `delay_cause` ещё не начата |
| 8. FastAPI + Docker | ❌ не начато | — |
| 9. Мониторинг + feedback loop | ❌ не начато | — |
| 10. Демонстрация end-to-end | ❌ не начато | — |

---

## Серия экспериментов — что есть в MLflow

`mlruns/` (file backend), эксперимент `flight-delay-mlops`. Все 7 runs
имеют теги `git_commit`, `dvc_data_hash`, `feature_set`, `model`,
`task=delay_binary`, `params_version=v1`. Полное описание дельт
с интерпретацией — в `reports/experiments_log.md`.

| # | run_id | git_commit | конфиг | f1 | roc_auc |
|---|---|---|---|---|---|
| 1 | `a19cc81f` | `6f28a3b` | basic + logreg | 0.450 | 0.665 |
| 2 | `41b18e98` | `6f28a3b` | extended + logreg | 0.543 | 0.766 |
| 3 | `cb56b5bc` | `6f28a3b` | with_weather + logreg | 0.606 | 0.825 |
| 4 | `1b00226a` | `45834a0` | with_weather + xgboost (default) | 0.572 | 0.839 |
| 5 | `a7a0ecd5` | `341e954` | with_weather + xgboost (manual tune) | 0.619 | 0.830 |
| 6 | `4227cd7b` | `b3dc9c8` | with_weather + xgboost (optuna 30 trials) | **0.630** | **0.839** |
| 7 | `d60d3feb` | `c419a05` | with_weather + lightgbm (default) | 0.573 | 0.839 |

**Текущий лидер по F1:** Run #6 (xgboost optuna), F1=0.630, ROC-AUC=0.839.
**Run #7 — научный смысл:** copy-paste результата Run #4 (lgbm default ≈ xgb default
на ±0.001 по всем метрикам). Подтверждает: смена boosting library без балансировки
классов не двигает метрики — узкое место в `scale_pos_weight`/`is_unbalance`, не в
выборе библиотеки. См. Δ6 в `experiments_log.md` (важный сюжет для главы 3).
**Лучшие гиперпараметры XGBoost:** `models/best_xgboost_params.yaml` (gitignored, локальный артефакт; копия — артефакт MLflow run #6).

---

## Текущее состояние `params.yaml`

После Run #7. Активная конфигурация:

```yaml
features.active_set: with_weather
train.active_model: lightgbm    # ← переключено в commit c419a05 (Run #7)
train.task: delay_binary
train.lightgbm:                 # дефолты params.yaml — конфиг Run #7
  n_estimators: 400
  num_leaves: 63
  learning_rate: 0.05
  subsample: 0.9
  colsample_bytree: 0.9
  # БЕЗ class_weight / is_unbalance — это и есть причина почему Run #7
  # дублирует Run #4 по метрикам. Run #8 = добавить балансировку + Optuna.
train.xgboost:                  # без изменений с Run #5; не активен
  n_estimators: 800
  max_depth: 8
  learning_rate: 0.05
  subsample: 0.9
  colsample_bytree: 0.9
  scale_pos_weight: 3.17        # Optuna нашла лучший в Run #6 → models/best_xgboost_params.yaml
```

**Решение по промоушу Optuna best params в `params.yaml`** — всё ещё отложено.
Если нужно вернуть лидера (Run #6) одной командой:
1. `params.yaml → train.active_model: lightgbm → xgboost`
2. Скопировать содержимое `models/best_xgboost_params.yaml` в `params.yaml → train.xgboost`
3. Коммит `experiment: promote optuna best params to xgboost defaults` (или просто переключить active_model назад на xgboost — сами параметры Run #5 уже там).

---

## Где продолжать серию экспериментов

Пользователь явно просил «продолжить серию экспериментов». На столе три
направления, выбор за пользователем:

### Вариант A — расширить серию по `delay_binary` (рекомендую как первый)

- **Run #8 — LightGBM с балансировкой классов + Optuna.** Это честный
  head-to-head с Run #6 и логичный ответ на вывод из Δ6 (см.
  `experiments_log.md`): без `is_unbalance=True` / `class_weight='balanced'`
  LightGBM просто копирует Run #4. Гипотеза: F1 ≈ 0.62–0.64, паритет
  с XGBoost. **15–25 минут** (Optuna 30 trials на LightGBM ~ как было
  для XGBoost, поправить `tune.py` под новую модель).
  - В `tune.py` добавить ветку для `lightgbm` (или сделать `--model lightgbm`)
  - Запустить `python -m src.models.tune --model lightgbm --n-trials 30`
  - Сохранить best в `models/best_lightgbm_params.yaml`
  - update `experiments_log.md` + commit

- **Run #9 — CatBoost** на `with_weather` + `auto_class_weights=Balanced`.
  Часто лучший на табличке с категориями (у нас их много —
  airline_code, airport ICAO, route, fleet_type). **5–10 минут** для
  defaults; +Optuna ~ ещё 15–25 минут.

- **Run #10 — Stacking/Voting ensemble** — лучшего logreg + xgboost (Run #6) +
  lightgbm (Run #8 после тюнинга). Нужно дописать `src/models/ensemble.py`
  (создать его). **20–30 мин** кода + коммит. Обычно ещё +1–2 пункта F1.

### Вариант B — вторая голова: серия по `delay_cause` (multi-class)

Повторить ту же цепочку (data → model → tuning), но на multi-class
таргете `probable_delay_cause`. Это:
1. Расширить `train.py` для multi-class — `evaluate.py` уже надо дополнить
   функцией `multiclass_classification_metrics` (macro_f1, weighted_f1,
   per-class). NB! `train.py` сейчас падает на task != "delay_binary"
   (`raise NotImplementedError`).
2. Прогнать `basic → extended → with_weather → +xgboost → +tune` ещё 5–6 runs.
3. Соответственно расширить `experiments_log.md`.

Это **большое расширение по объёму работы** — закроет вторую заявленную
в CLAUDE.md задачу полностью. **Полдня работы.**

### Вариант C — перейти к Этапу 8 (FastAPI + Docker)

Серию экспериментов считать достаточной (7 runs, 6 дельт), идти дальше
по жизненному циклу. План §7.8:
- Регистрация финальной модели (Run #6) в MLflow Model Registry.
- `src/api/main.py`: FastAPI с `/predict/delay`, `/predict/cause`,
  `/health`, `/model/info`.
- `docker/api.Dockerfile`, `docker/mlflow.Dockerfile`,
  `docker-compose.yml` (mlflow tracking + api).
- Переключить `params.yaml → mlflow.tracking_uri` на `http://mlflow:5000`.

**Важная зависимость:** для Этапа 8 docker нужен — на хосте сейчас
docker НЕ установлен (проверял в первой сессии). Если идём в B → попросить
пользователя поставить Docker Desktop, либо использовать локальный
запуск без compose.

### Моя рекомендация на момент handoff

Спросить пользователя в начале сессии:
1. «Хочешь ещё runs по `delay_binary` (Run #8+), вторую серию по `delay_cause`,
   или переходим к API/Docker?»
2. Если ответ «как ты решишь» — идти **Вариант A → Run #8 (LightGBM
   balanced + Optuna)**: закроет вопрос «справедливо ли мы сравнили
   XGBoost vs LightGBM?», который явно поднят в Δ6 после Run #7.
   После Run #8 — Run #9 (CatBoost), потом стек (Run #10), и серия
   по `delay_binary` будет закрыта 4 моделями × 2 конфига каждая.

---

## Tooling state

| Инструмент | Где | Версия | Статус |
|---|---|---|---|
| Python | `.venv/bin/python` (локальный venv через uv) | 3.11.15 | ✅ |
| uv | `~/.local/bin/uv` (export PATH) | 0.11.8 | ✅ |
| git | системный | 2.50.1 | ✅, remote `origin` = github.com/GeorgeSard/VKR2 |
| dvc | `.venv/bin/dvc` | 3.x | ✅, remote `local-storage` = `~/.dvc-storage/vkr2-flight-delays` |
| mlflow | `.venv/bin/mlflow` | 2.x | ✅, file backend `file:./mlruns` |
| brew libomp | `/opt/homebrew/opt/libomp` | — | ✅ (нужно для xgboost на macOS) |
| docker | — | — | ❌ не установлен (нужен для Этапа 8) |
| MLflow UI | background-таск из этой сессии | — | ⚠️ умрёт после закрытия сессии — поднимать заново (см. ниже) |

### Команды для рестарта окружения завтра

```bash
cd /Users/georgij/Projects/ВКР2

# 1. Активировать venv (или использовать .venv/bin/python напрямую)
source .venv/bin/activate

# 2. Проверить что данные на месте (если data/raw/ пуст после git clone)
dvc pull

# 3. Поднять MLflow UI на http://127.0.0.1:5000
mlflow ui --backend-store-uri "file:$PWD/mlruns" --host 127.0.0.1 --port 5000

# 4. Запустить train (с текущими params.yaml)
python -m src.models.train

# 5. Запустить optuna (если нужно)
python -m src.models.tune --n-trials 30
```

---

## Файловая структура на момент handoff

Изменения относительно скелета из CLAUDE.md §6 (что уже создано):

```
├── HANDOFF.md                              ← ЭТО ОН
├── reports/experiments_log.md              ← основной артефакт серии
├── params.yaml                             ← конфиг всех экспериментов
├── dvc.yaml                                ← stages: ingest, split
├── dvc.lock
├── src/
│   ├── config.py                           ← load_params()
│   ├── data/
│   │   ├── generate.py                     ← синтетический генератор
│   │   ├── ingest.py                       ← raw → interim
│   │   └── split.py                        ← time-based train/val/test
│   ├── features/
│   │   ├── feature_sets.py                 ← BASIC / EXTENDED / WITH_WEATHER + LEAKAGE_COLUMNS
│   │   └── build_features.py               ← ColumnTransformer + make_xy
│   └── models/
│       ├── train.py                        ← entry-point + dispatch для logreg/rf/xgb/lgbm/catboost
│       ├── tune.py                         ← Optuna study (single MLflow run)
│       └── evaluate.py                     ← binary_classification_metrics
├── data/raw/{flight_delays_ru.parquet, sample.csv}.dvc   ← в git, данные в DVC remote
├── models/best_xgboost_params.yaml         ← gitignored, артефакт Run #6
└── mlruns/                                 ← gitignored, 7 runs внутри
```

Чего ещё НЕТ (упомянуто в плане, но не создано):
- `src/data/annotate.py` — вместо него минимальная валидация в `ingest.py`
- `src/models/ensemble.py` — для Run #9 stacking
- `src/models/registry.py` — обёртка над MLflow Model Registry, нужна для Этапа 8
- `src/api/`, `src/monitoring/` — пусто (только `__init__.py`)
- `docker/` — пусто
- `tests/` — пусто (тестов ещё не писали)
- `notebooks/03_eda.ipynb` — НЕ создан, EDA отложили в пользу быстрого baseline

---

## История коммитов на main (для git log сверки)

```
94174e5 docs(experiments): log Run #7 (LightGBM defaults) — boosting library is not the lever
c419a05 experiment: switch active model xgboost → lightgbm on with_weather
a1b976e docs: add HANDOFF.md for cross-session continuity
84d0892 experiment: optuna xgboost — Run #6, f1 0.619 → 0.630 (+1.8%)
b3dc9c8 feat(tuning): add Optuna XGBoost tuner with single-run MLflow logging
40e2e32 docs(experiments): log first MLflow series (5 runs, 4 deltas)
341e954 experiment: tune xgboost — depth 6→8, n_est 400→800, scale_pos_weight 3.17
45834a0 experiment: switch active model logreg → xgboost on with_weather features
b6fd9a3 feat(training): wire up RandomForest, XGBoost, LightGBM, CatBoost
129d431 experiment: data axis — feature progression on logreg
6f28a3b feat(training): baseline LogReg + MLflow tracking
0da22bc feat(data): version dataset with DVC and add ingest+split pipeline
8c4be93 chore: bootstrap project structure and tooling
```

---

## Действующие договорённости (из memory + переписки)

1. **Scope:** только программная часть (Этапы 0, 3–10), главы 1–2 ВКР — сам.
2. **Git autonomy:** коммитить и пушить без переспросов, формат Conventional Commits.
3. **Демонстрация ML-процесса:** каждый эксперимент = одна git-коммитная единица + один MLflow run + одна строка в `experiments_log.md`. Менять только **одну ось** между соседними runs.
4. **Headroom для отчёта:** не вылизывать модель/данные слишком рано — оставлять место для следующих скринов прогресса.
