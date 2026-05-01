# Session Handoff — flight-delay-mlops

> Файл для быстрого входа в контекст следующей сессии Claude.
> Источник правды для статуса проекта между сессиями.
> При старте новой сессии скажи Claude: «прочитай HANDOFF.md» — этого
> вместе с CLAUDE.md и memory будет достаточно, чтобы продолжить с
> того же места без `/compact`.

**Последнее обновление:** 2026-05-02
**HEAD git:** `84d0892` (`origin/main`, всё запушено)
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
| 7. Научные эксперименты | 🔄 в процессе | 6 runs готовы (см. ниже), серия по `delay_cause` ещё не начата |
| 8. FastAPI + Docker | ❌ не начато | — |
| 9. Мониторинг + feedback loop | ❌ не начато | — |
| 10. Демонстрация end-to-end | ❌ не начато | — |

---

## Серия экспериментов — что есть в MLflow

`mlruns/` (file backend), эксперимент `flight-delay-mlops`. Все 6 runs
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

**Текущий лидер по F1:** Run #6, F1=0.630, ROC-AUC=0.839.
**Лучшие гиперпараметры XGBoost:** `models/best_xgboost_params.yaml` (gitignored, локальный артефакт; копия — артефакт MLflow run #6).

---

## Текущее состояние `params.yaml`

После Run #6 не менял. Активная конфигурация:

```yaml
features.active_set: with_weather
train.active_model: xgboost
train.task: delay_binary
train.xgboost:
  n_estimators: 800
  max_depth: 8
  learning_rate: 0.05
  subsample: 0.9
  colsample_bytree: 0.9
  scale_pos_weight: 3.17     # это конфиг Run #5; Optuna нашла лучший в #6 но в params.yaml не промоутил
```

**Решение по промоушу Optuna best params в `params.yaml`** — отложено.
Если завтра пользователь хочет, чтобы дефолтный `python -m src.models.train`
давал результат Run #6 — скопировать содержимое `models/best_xgboost_params.yaml`
в `params.yaml → train.xgboost`, закоммитить отдельно как
`experiment: promote optuna best params to xgboost defaults`.

---

## Где продолжать серию экспериментов

Пользователь явно просил «продолжить серию экспериментов». На столе три
направления, выбор за пользователем:

### Вариант A — расширить серию по `delay_binary` (рекомендую как первый)

- **Run #7 — LightGBM** на `with_weather`. Параллельный бустинг для
  сравнения с XGBoost; обычно очень близок по метрикам, но другая модель
  даёт ещё одну дельту по «модельной оси». **5–10 минут.**
  - Edit `params.yaml`: `train.active_model: xgboost → lightgbm`
  - `git commit "experiment: lightgbm on with_weather"`
  - `python -m src.models.train`
  - update `experiments_log.md` + commit

- **Run #8 — CatBoost** аналогично, для полного pool бустингов. Часто
  лучший на табличных данных с категориями. **5–10 минут.**

- **Run #9 — Stacking ensemble** — voting/stacking лучшего logreg + xgboost +
  lightgbm. Нужно дописать `src/models/ensemble.py` (создать его). **20–30 мин**
  кода + коммит.

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

Серию экспериментов считать достаточной (6 runs, 5 дельт), идти дальше
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
1. «Хочешь ещё runs по `delay_binary` (Run #7+), вторую серию по `delay_cause`,
   или переходим к API/Docker?»
2. Если ответ «как ты решишь» — идти **Вариант A → Run #7 (LightGBM)**:
   быстрый win, очередной скрин для отчёта, не блокирует ничего.

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
└── mlruns/                                 ← gitignored, 6 runs внутри
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
