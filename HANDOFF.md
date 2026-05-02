# Session Handoff — flight-delay-mlops

> Файл для быстрого входа в контекст следующей сессии Claude.
> Источник правды для статуса проекта между сессиями.
> При старте новой сессии скажи Claude: «прочитай HANDOFF.md» — этого
> вместе с CLAUDE.md и memory будет достаточно, чтобы продолжить с
> того же места без `/compact`.

**Последнее обновление:** 2026-05-02 (после Cause series C1-C4)
**HEAD git:** см. `git log -1 --format=%h` (последние коммиты —
infra `9ab4f8a`, серия `0cbbf8a` + `4ad72c5`, лог `<this>`)
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
| 7. Научные эксперименты | 🔄 в процессе | binary серия 8 runs (plateau F1 0.63); cause серия 4 runs (C1-C4, macro_f1 0.137 → 0.359) |
| 8. FastAPI + Docker | ❌ не начато | — |
| 9. Мониторинг + feedback loop | ❌ не начато | — |
| 10. Демонстрация end-to-end | ❌ не начато | — |

---

## Серия экспериментов — что есть в MLflow

`mlruns/` (file backend), эксперимент `flight-delay-mlops`. Все 8 runs
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
| 8 | `716664c0` | `3ce36c8` | with_weather + lightgbm (optuna 30 trials) | **0.629** | **0.839** |

**Текущий лидер по F1:** формально Run #6 (xgb optuna, F1=0.630), но Run #8
(lgbm optuna, F1=0.629) — паритет в пределах шума. ROC-AUC=0.839 у обоих.
**Сюжет серии для главы 3:** XGBoost vs LightGBM при одинаковой методике
(балансировка + Optuna 30 trials, тот же `seed=42`, тот же search space там
где параметры эквивалентны) сошлись к одной точке — F1 ≈ 0.63, ROC-AUC ≈ 0.839.
Optuna независимо нашла похожие конфиги: low LR ~ 0.014, scale_pos_weight ~ 2.5
(оба мягче моего ручного 3.17), high n_estimators. Сильное доказательство, что
plateau определён данными, а не моделью.
**Лучшие гиперпараметры:** `models/best_xgboost_params.yaml` (Run #6) и
`models/best_lightgbm_params.yaml` (Run #8) — оба gitignored, локальные артефакты,
копии в MLflow runs соответственно.

---

## Текущее состояние `params.yaml`

После Cause Run C4. Активная конфигурация:

```yaml
features.active_set: with_weather
train.active_model: xgboost     # ← переключено в коммите 4ad72c5 (Cause C4)
train.task: delay_cause         # ← переключено для cause series
train.lightgbm:                 # дефолты params.yaml — конфиг Run #7 (default lgbm)
  n_estimators: 400
  num_leaves: 63
  learning_rate: 0.05
  subsample: 0.9
  colsample_bytree: 0.9
  # БЕЗ class_weight / is_unbalance в params.yaml. Optuna нашла лучший
  # конфиг в Run #8 → models/best_lightgbm_params.yaml (см. ниже).
train.xgboost:                  # без изменений с Run #5; не активен
  n_estimators: 800
  max_depth: 8
  learning_rate: 0.05
  subsample: 0.9
  colsample_bytree: 0.9
  scale_pos_weight: 3.17        # Optuna нашла лучший в Run #6 → models/best_xgboost_params.yaml
```

**Решение по промоушу Optuna best params в `params.yaml`** — всё ещё отложено
для обеих моделей. Best-params артефакты живут отдельно:

| Run | Файл | Что внутри |
|---|---|---|
| #6 | `models/best_xgboost_params.yaml` | best XGBoost (n_est=1000, lr=0.0138, scale_pos_weight=2.371, …) |
| #8 | `models/best_lightgbm_params.yaml` | best LightGBM (n_est=700, num_leaves=105, lr=0.0141, scale_pos_weight=2.665, …) |

Если нужно дефолтным `python -m src.models.train` получить Run #6 или #8 — скопировать
содержимое соответствующего файла в `params.yaml → train.<model>` и коммит
`experiment: promote optuna best params to <model> defaults`. Сейчас `train.py`
с `active_model: lightgbm` даст результат Run #7 (default lgbm).

---

## Где продолжать серию экспериментов

Пользователь явно просил «продолжить серию экспериментов». На столе три
направления, выбор за пользователем:

### Вариант A — закрыть серию `delay_binary` оставшимися моделями/осями

Серия `delay_binary` достигла **plateau ≈ F1 0.63 / ROC-AUC 0.839** на
текущем наборе фичей (Run #6 и Run #8 в паритете). Дальнейшие приросты
ждать не от ещё одного бустинга, а от смены оси.

- **Run #9 — CatBoost** на `with_weather` + `auto_class_weights=Balanced`,
  далее Optuna через расширение `tune.py` (добавить ветку `--model catboost`,
  как сделано для lightgbm в commit `3ce36c8`). Маловероятно перебьёт
  plateau, но даёт третью точку библиотечной оси (xgb / lgbm / cb) —
  закрывает «pool бустингов» в отчёте. **5 минут defaults + 20–30 минут Optuna.**

- **Run #10 — Stacking/Voting ensemble** — лучшего logreg (Run #3) +
  xgboost (Run #6) + lightgbm (Run #8) через новый `src/models/ensemble.py`.
  Обычно даёт ещё +1–2 пункта F1 на бустинговом плато; в отчёте закрывает
  архитектурную ось. **20–30 мин кода + 5–10 мин fit.**

- **Run #11+ — следующая ось данных (рекомендую как самый ценный
  следующий шаг для отчёта).** Plateau показывает, что ROI смены модели
  исчерпан; новая дельта ≥ +0.02 F1 реально получить только из новых
  фичей. Кандидаты: календарные/сезонные (праздники РФ, школьные
  каникулы, day_of_week × month взаимодействия), исторический average
  delay по маршруту/борту за последние N дней, индикатор пиковых часов
  по аэропорту. Это новая ось данных — аналог Δ1 (extended) и Δ2
  (with_weather). **30–60 мин фичей в `feature_sets.py` + 10 мин runs.**

### Вариант B — вторая голова: серия по `delay_cause` (multi-class) — ✅ В ПРОЦЕССЕ

**Сделано:** инфраструктура (commit `9ab4f8a` — `multiclass_classification_metrics`,
LabelEncoder в `train.py`, balanced sample_weight для бустингов, label
classes как MLflow артефакт) + 4 runs:

| Run | git | feature_set / model | macro_f1 | acc | weighted_f1 | roc_auc_ovr |
|---|---|---|---|---|---|---|
| C1 | `9ab4f8a` | basic / logreg          | 0.137 | 0.194 | 0.239 | 0.670 |
| C2 | `0cbbf8a` | extended / logreg       | 0.259 | 0.331 | 0.404 | 0.761 |
| C3 | `0cbbf8a` | with_weather / logreg   | 0.310 | 0.396 | 0.473 | 0.827 |
| C4 | `4ad72c5` | with_weather / xgboost  | **0.359** | **0.655** | **0.681** | 0.830 |

**Осталось** (опционально, с убывающим ROI):
- C5 — Optuna для xgboost на cause (нужно расширить `tune.py` под
  `--task delay_cause`; ожидаемая дельта +3-5 пунктов macro_f1).
- C6 — двухступенчатая постановка (бинарный `causal vs none` +
  мультиклассовый по причине внутри causal); часто +5-10 пунктов
  macro_f1 на задачах с доминирующим «нулевым» классом.
- C7 — LightGBM-параллель (для head-to-head как Run #6 vs #8).

### Вариант C — перейти к Этапу 8 (FastAPI + Docker)

Серию экспериментов считать достаточной (8 runs, 8 дельт, plateau
обнаружен и зафиксирован), идти дальше по жизненному циклу. План §7.8:
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
1. «Обе головы покрыты: binary plateau F1 0.63, cause macro_f1 0.359.
   Куда дальше:
   (i) Cause C5 — Optuna на xgboost (быстро, +3-5 пунктов macro_f1),
   (ii) Cause C6 — двухступенчатая постановка (потенциал +5-10 пунктов),
   (iii) собрать **scored test dataset** (deliverable из memory) с
        текущими лидерами обеих голов и зафиксировать,
   (iv) переходим к Этапу 8 — FastAPI + Docker?»
2. Если ответ «как ты решишь» — идти (iii) **scored test dataset**:
   обе головы уже имеют осмысленных лидеров (binary Run #6, cause C4),
   эта работа закрывает явное пользовательское требование из последней
   сессии и даёт демонстрируемый артефакт для ВКР. После можно
   возвращаться в C5/C6 или сразу в Этап 8.

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
| MLflow UI | background-таск (`bnikdp4p0` в текущей сессии) | — | 🟢 живой на http://127.0.0.1:5000 пока сессия открыта; после рестарта — команда ниже |

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

# 5. Запустить optuna (если нужно). По умолчанию — XGBoost, как в Run #6.
python -m src.models.tune --n-trials 30
# Для LightGBM (Run #8):
python -m src.models.tune --model lightgbm --n-trials 30
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
├── models/best_lightgbm_params.yaml        ← gitignored, артефакт Run #8
└── mlruns/                                 ← gitignored, 8 runs внутри
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
d8beb23 experiment: optuna lightgbm — Run #8, f1 0.573 → 0.629 (+9.8 %), parity with xgboost
3ce36c8 feat(tuning): generalize Optuna tuner to support LightGBM
81cc60f docs: refresh HANDOFF.md after Run #7 (LightGBM)
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
