# Session Handoff — flight-delay-mlops

> Файл для быстрого входа в контекст следующей сессии Claude.
> Источник правды для статуса проекта между сессиями.
> При старте новой сессии скажи Claude: «прочитай HANDOFF.md» — этого
> вместе с CLAUDE.md и memory будет достаточно, чтобы продолжить с
> того же места без `/compact`.

**Последнее обновление:** 2026-05-02 (после Cause C5 + scored deliverable
+ scored README + закрытия Этапа 7)
**HEAD git:** `549095b` (`origin/main`, всё запушено)
**Текущая ветка:** `main`

---

## ⚡ Резюме для следующей сессии — что делать сразу

1. **Этап 7 (научные эксперименты) ЗАКРЫТ**: 13 runs всего, обе головы
   достигли plateau, scored test dataset собран и задокументирован.
2. **Решение пользователя для следующей сессии:** идём на **Этап 8 —
   FastAPI + Docker**. Docker на хосте НЕ установлен, **пользователь
   попросил Claude самого установить Docker Desktop** в новой сессии
   (через `brew install --cask docker` + первый запуск `open -a Docker`
   для принятия лицензии — потребуется подтверждение пользователя).
3. **Стартовый план Этапа 8** (по плану CLAUDE.md §7.8):
   - `brew install --cask docker` + первый запуск + ждать пока daemon
     поднимется (`docker info` зелёный)
   - `src/models/registry.py` — обёртка над MLflow Model Registry
     для регистрации Run #6 (binary leader) и C5 (cause leader) под
     именами `flight-delay-binary` / `flight-delay-cause`
   - `src/api/schemas.py` — Pydantic-схемы input/output для обеих ручек
   - `src/api/inference.py` — загрузка моделей из MLflow registry
     (с in-memory кешем — модель грузится один раз при старте app)
   - `src/api/main.py` — FastAPI app с ручками:
     - `POST /predict/delay` — прогноз задержки + probability
     - `POST /predict/cause` — классификация причины + per-class proba
     - `GET /health` — health check (модели загружены?)
     - `GET /model/info` — версии моделей (binary/cause), git_commit,
       dvc_data_hash из MLflow тегов
   - `docker/api.Dockerfile` — multi-stage build на python:3.11-slim
   - `docker/mlflow.Dockerfile` — простой MLflow tracking server
   - `docker-compose.yml` — `mlflow` + `api` сервисы
   - Переключить `params.yaml → mlflow.tracking_uri` на
     `http://mlflow:5000` для контейнерного режима (с fallback на
     локальный file:./mlruns для dev)
   - Smoke: `docker compose up` → `curl POST /predict/delay` с примером
4. **Альтернатива Этапу 8:** C6 — двухступенчатая cause-постановка
   (единственный способ пробить plateau головы B без новых фичей).
   Пользователь от этого отказался в пользу деплоя.

---

## Где мы по плану CLAUDE.md §7

| Этап | Статус | Что сделано |
|---|---|---|
| 0. Bootstrap | ✅ | `8c4be93` — структура, pyproject, .gitignore, pre-commit |
| 1. Глава 1 | ⛔ skip | вне scope (см. memory `feedback_scope_programmatic_only_vkr2`) |
| 2. Глава 2 | ⛔ skip | вне scope |
| 3. Данные + DVC | ✅ | `0da22bc` — dvc init, локальный remote, ingest+split pipeline, params.yaml |
| 4. Feature engineering | ✅ | `6f28a3b` — 3 feature sets: basic / extended / with_weather |
| 5. DVC pipeline | ⏳ partial | есть стадии ingest+split; train/evaluate как DVC stages — НЕ добавлены, train запускается напрямую (план: добить вместе с Этапом 8) |
| 6. MLflow + baseline | ✅ | `6f28a3b` — train.py с MLflow tracking, file backend в `mlruns/` |
| 7. Научные эксперименты | ✅ ЗАКРЫТА | binary 8 runs (plateau F1 0.63); cause 5 runs (C1-C5, plateau macro_f1 0.36); scored test dataset + README собраны |
| **8. FastAPI + Docker** | ⏭️ **СЛЕДУЮЩИЙ** | docker сам ставлю; план в верхушке файла |
| 9. Мониторинг + feedback loop | ❌ не начато | — |
| 10. Демонстрация end-to-end | ❌ не начато | — |

---

## Серия экспериментов — что есть в MLflow

`mlruns/` (file backend), эксперимент `flight-delay-mlops`. Все 13 runs
имеют теги `git_commit`, `dvc_data_hash`, `feature_set`, `model`,
`task`, `params_version=v1`. Полное описание дельт с интерпретацией —
в `reports/experiments_log.md`.

### Binary head (`task=delay_binary`) — 8 runs

| # | run_id | git_commit | конфиг | f1 | roc_auc |
|---|---|---|---|---|---|
| 1 | `a19cc81f` | `6f28a3b` | basic + logreg | 0.450 | 0.665 |
| 2 | `41b18e98` | `6f28a3b` | extended + logreg | 0.543 | 0.766 |
| 3 | `cb56b5bc` | `6f28a3b` | with_weather + logreg | 0.606 | 0.825 |
| 4 | `1b00226a` | `45834a0` | with_weather + xgboost (default) | 0.572 | 0.839 |
| 5 | `a7a0ecd5` | `341e954` | with_weather + xgboost (manual tune) | 0.619 | 0.830 |
| **6** | `4227cd7b` | `b3dc9c8` | with_weather + xgboost (optuna 30 trials) | **0.630** | **0.839** |
| 7 | `d60d3feb` | `c419a05` | with_weather + lightgbm (default) | 0.573 | 0.839 |
| 8 | `716664c0` | `3ce36c8` | with_weather + lightgbm (optuna 30 trials) | **0.629** | **0.839** |

**Лидер binary:** Run #6 (xgb optuna). Best params в
`models/best_xgboost_params.yaml`.

### Cause head (`task=delay_cause`) — 5 runs

| # | run_id | git_commit | конфиг | macro_f1 | acc | weighted_f1 | roc_auc_ovr |
|---|---|---|---|---|---|---|---|
| C1 | `faae4868` | `9ab4f8a` | basic + logreg + class_weight | 0.137 | 0.194 | 0.239 | 0.670 |
| C2 | `fd12e644` | `0cbbf8a` | extended + logreg + class_weight | 0.259 | 0.331 | 0.404 | 0.761 |
| C3 | `6b948e3a` | `0cbbf8a` | with_weather + logreg + class_weight | 0.310 | 0.396 | 0.473 | 0.827 |
| C4 | `5520b46a` | `4ad72c5` | with_weather + xgboost defaults + balanced sw | 0.359 | 0.655 | 0.681 | 0.830 |
| **C5** | `400b6695` | `a249735` | with_weather + xgboost optuna + balanced sw | **0.361** | **0.656** | **0.682** | 0.830 |

**Лидер cause:** C5 (xgb optuna). Best params в
`models/best_xgboost_delay_cause_params.yaml`.

### Главные сюжеты для главы 3 (доказаны 13 экспериментами)

1. **Plateau на обеих головах при одинаковом наборе фичей.** binary
   plateau F1 0.63 / ROC-AUC 0.839 достигнут двумя независимыми
   путями (xgb Run #6 ↔ lgbm Run #8). cause plateau macro_f1 0.36
   достигнут аналогично (C4 defaults ↔ C5 optuna).
2. **«Перебор гиперпараметров не двигает метрики при тех же фичах»** —
   тезис руководителя проверен и подтверждён дважды (на двух головах).
   Прямое следствие: ROI смены модели исчерпан, headroom только в данных.
3. **Параметрическая параллель Optuna между library/task** — TPE
   независимо сходится к похожим режимам регуляризации (low LR, мягкая
   class balance), что усиливает интерпретируемость как «реальный
   оптимум, не артефакт сэмплера».

---

## Финальный артефакт — scored test dataset

`reports/scored_test_dataset.{parquet,csv,xlsx}` + русскоязычный
`reports/SCORED_DATASET_README.md`. Сгенерирован
`python -m src.models.score_dataset` за ~1 мин.

| файл | размер | назначение |
|---|---|---|
| `.xlsx` | 4.2 MB | Excel: 3 листа (full / preview 100 / cause confusion matrix) |
| `.csv` | 8.4 MB | UTF-8 BOM + `;` разделитель — открывается в Excel без кракозябров |
| `.parquet` | 860 KB | для pandas / downstream tooling |
| `SCORED_DATASET_README.md` | 11 KB | объяснение каждой колонки + готовые сценарии для Excel |

21 колонка на каждый из 36 983 рейсов test split (Jul+ 2025): идентификаторы +
расписание + ground truth + предсказания обеих голов + вероятности +
маркеры правильности (✓/✗/н/п).

**Headline на test:**
- Голова A (binary): **accuracy 81.9 %** на оценимых рейсах
- Голова B (cause): **accuracy 69.6 %** в целом по 7 классам

Сами файлы gitignored (регенерируемые); в git только скрипт + README.

---

## Текущее состояние `params.yaml`

После Cause C5. Активная конфигурация:

```yaml
features.active_set: with_weather
train.active_model: xgboost     # переключено в коммите 4ad72c5 (Cause C4)
train.task: delay_cause         # переключено для cause series
```

`train.xgboost`, `train.lightgbm` — defaults (gitignored Optuna best
params живут отдельно, см. ниже). `train.py` с этими настройками даст
result Cause C4 (defaults для cause), не C5 (для C5 нужно подгрузить
`models/best_xgboost_delay_cause_params.yaml`).

**Optuna best params** (gitignored, в `models/`):

| Run | Файл | Что внутри |
|---|---|---|
| #6 binary | `models/best_xgboost_params.yaml` | n_est=1000, lr=0.0138, scale_pos_weight=2.371, … |
| #8 binary | `models/best_lightgbm_params.yaml` | n_est=700, num_leaves=105, lr=0.0141, scale_pos_weight=2.665, … |
| C5 cause | `models/best_xgboost_delay_cause_params.yaml` | n_est=700, max_depth=8, lr=0.0782, subsample=0.762, colsample=0.667, min_child_weight=7 |

Промоушен этих best params в `params.yaml` отложен на момент Этапа 8
(там `inference.py` будет грузить модель из MLflow registry, а не
переобучать через `train.py`).

---

## 🖥️ MLflow UI — как поднять для скриншотов

### Запуск (одна команда, выполнить из корня репо)

```bash
cd /Users/georgij/Projects/ВКР2
mlflow ui --backend-store-uri "file:$PWD/mlruns" --host 127.0.0.1 --port 5000
```

Откроется веб-UI на **http://127.0.0.1:5000**. Команда блокирующая —
держит терминал, для остановки `Ctrl+C`. Если нужно в фоне:

```bash
nohup mlflow ui --backend-store-uri "file:$PWD/mlruns" --host 127.0.0.1 --port 5000 > /tmp/mlflow_ui.log 2>&1 &
```

### Что смотреть в UI (для скриншотов в отчёт)

1. **Главная страница эксперимента** — слева выбрать
   `flight-delay-mlops`, увидишь все 13 runs списком. Скрин этого
   списка — главная иллюстрация серии экспериментов.

2. **Сравнение runs (Compare)** — выделить 2-3 runs галочками + кнопка
   `Compare` сверху. MLflow покажет parallel coordinates plot и таблицу
   метрик. Полезные пары:
   - Run #1 (basic logreg) ↔ Run #3 (with_weather logreg) → дельта от
     данных
   - Run #4 (xgb default) ↔ Run #6 (xgb optuna) → дельта от тюнинга
   - Run #6 (xgb optuna) ↔ Run #8 (lgbm optuna) → паритет библиотек
   - C4 (cause defaults) ↔ C5 (cause optuna) → cause plateau
   - C1 → C5 → весь прогресс cause-серии в одном графике

3. **Filter runs по тегу task** — в строке поиска ввести
   `tags.task = "delay_binary"` или `tags.task = "delay_cause"` чтобы
   разделить две серии для отдельных скринов.

4. **Карточка одного run** — клик по имени run → вкладки `Parameters`,
   `Metrics`, `Tags`, `Artifacts`. В `Artifacts` для tuning-runs лежит
   `best_*_params.yaml` (можно открыть прямо в UI). Для всех runs
   лежит `model/` с MLflow-сохранённым pipeline.

5. **Метрики во времени** — внутри run, вкладка `Metrics` → клик на
   метрику → график. Для одиночных runs не очень информативно (одна
   точка), полезно для cross-run сравнения через Compare.

### Если UI не открывается

- Проверить что в `mlruns/` есть данные: `ls mlruns/` должна показать
  каталоги вида `0/`, `1/`, …
- Проверить что порт 5000 свободен:
  `lsof -i :5000` (если что-то занято — убить или поменять `--port 5001`)
- Проверить `mlflow` версию: `mlflow --version` → ожидается ≥ 2.x

---

## Tooling state

| Инструмент | Где | Версия | Статус |
|---|---|---|---|
| Python | `.venv/bin/python` (локальный venv через uv) | 3.11.15 | ✅ |
| uv | `~/.local/bin/uv` (export PATH) | 0.11.8 | ✅ |
| git | системный | 2.50.1 | ✅, remote `origin` = github.com/GeorgeSard/VKR2 |
| dvc | `.venv/bin/dvc` | 3.x | ✅, remote `local-storage` = `~/.dvc-storage/vkr2-flight-delays` |
| mlflow | `.venv/bin/mlflow` | 2.x | ✅, file backend `file:./mlruns`, 13 runs |
| brew libomp | `/opt/homebrew/opt/libomp` | — | ✅ (нужно для xgboost на macOS) |
| openpyxl | `.venv/bin/python` | 3.1.5 | ✅, добавлен в `pyproject.toml` для `score_dataset.py` |
| **docker** | — | — | **❌ не установлен — Claude установит в начале Этапа 8** |

### Команды для рестарта окружения

```bash
cd /Users/georgij/Projects/ВКР2
source .venv/bin/activate

# Подтянуть данные если data/raw/ пуст после fresh clone
dvc pull

# MLflow UI на http://127.0.0.1:5000
mlflow ui --backend-store-uri "file:$PWD/mlruns" --host 127.0.0.1 --port 5000

# Обучение текущей конфигурации (params.yaml)
python -m src.models.train

# Optuna binary (Run #6/#8 reproducibility)
python -m src.models.tune --n-trials 30                       # xgboost
python -m src.models.tune --model lightgbm --n-trials 30      # lightgbm

# Optuna cause (C5)
python -m src.models.tune --task delay_cause --model xgboost --n-trials 30

# Перегенерация финального scored test dataset
python -m src.models.score_dataset
```

---

## Файловая структура на момент handoff

```
├── HANDOFF.md                              ← ЭТО ОН
├── CLAUDE.md                               ← основные правила проекта
├── reports/
│   ├── experiments_log.md                  ← основной артефакт серии (13 runs, дельты)
│   ├── SCORED_DATASET_README.md            ← русскоязычная инструкция к scored датасету
│   ├── scored_test_dataset.parquet         ← gitignored, регенерируется
│   ├── scored_test_dataset.csv             ← gitignored, регенерируется
│   └── scored_test_dataset.xlsx            ← gitignored, регенерируется
├── params.yaml                             ← конфиг всех экспериментов
├── dvc.yaml                                ← stages: ingest, split
├── dvc.lock
├── pyproject.toml                          ← + openpyxl
├── src/
│   ├── config.py                           ← load_params()
│   ├── data/
│   │   ├── generate.py                     ← синтетический генератор
│   │   ├── ingest.py                       ← raw → interim
│   │   └── split.py                        ← time-based train/val/test
│   ├── features/
│   │   ├── feature_sets.py                 ← BASIC / EXTENDED / WITH_WEATHER + LEAKAGE
│   │   └── build_features.py               ← ColumnTransformer + make_xy
│   └── models/
│       ├── train.py                        ← entry-point + binary AND multiclass dispatch
│       ├── tune.py                         ← Optuna study (xgb/lgbm × binary/cause)
│       ├── evaluate.py                     ← binary + multiclass metrics
│       └── score_dataset.py                ← финальный scored test dataset (обе головы)
├── data/raw/{flight_delays_ru.parquet, sample.csv}.dvc   ← в git, данные в DVC remote
├── models/best_xgboost_params.yaml                       ← gitignored, Run #6
├── models/best_lightgbm_params.yaml                      ← gitignored, Run #8
├── models/best_xgboost_delay_cause_params.yaml           ← gitignored, C5
├── models/label_classes_delay_cause.yaml                 ← gitignored, для inference cause-головы
└── mlruns/                                               ← gitignored, 13 runs
```

Чего ещё НЕТ (план для Этапа 8):
- `src/models/registry.py` — обёртка над MLflow Model Registry
- `src/api/main.py`, `src/api/schemas.py`, `src/api/inference.py`
- `src/monitoring/` — пусто (Этап 9)
- `docker/api.Dockerfile`, `docker/mlflow.Dockerfile`,
  `docker-compose.yml`
- `tests/` — пусто (тестов ещё не писали; добавить с Этапом 8 — pytest
  на API endpoints)
- `notebooks/03_eda.ipynb` — не создан, EDA пропустили в пользу быстрого
  baseline

---

## История коммитов на main (для git log сверки)

```
549095b docs: refresh HANDOFF.md after Cause C5 + scored deliverable
a249735 experiment: cause C5 — Optuna xgboost on delay_cause, plateau confirmed
0632ef7 feat(tuning+docs): generalize Optuna tuner to multiclass + scored-dataset README
4a7db59 feat(deliverable): scored test dataset script — both heads, human-readable
9f9f0e3 docs: refresh HANDOFF.md after Cause series C1-C4
39346b1 docs(experiments): log Cause series C1-C4 — second-head deltas on delay_cause
4ad72c5 experiment: cause C4 — xgboost on with_weather + balanced sample weights
0cbbf8a experiment: cause series C1-C3 — logreg data axis on delay_cause
9ab4f8a feat(eval+train): support delay_cause multiclass task
80ba65e docs: refresh HANDOFF.md after Run #8 (LightGBM Optuna)
d8beb23 experiment: optuna lightgbm — Run #8, f1 0.573 → 0.629 (+9.8 %)
3ce36c8 feat(tuning): generalize Optuna tuner to support LightGBM
81cc60f docs: refresh HANDOFF.md after Run #7 (LightGBM)
94174e5 docs(experiments): log Run #7 (LightGBM defaults)
c419a05 experiment: switch active model xgboost → lightgbm on with_weather
a1b976e docs: add HANDOFF.md for cross-session continuity
84d0892 experiment: optuna xgboost — Run #6, f1 0.619 → 0.630 (+1.8 %)
b3dc9c8 feat(tuning): add Optuna XGBoost tuner with single-run MLflow logging
40e2e32 docs(experiments): log first MLflow series (5 runs, 4 deltas)
341e954 experiment: tune xgboost — depth 6→8, n_est 400→800, scale_pos_weight 3.17
45834a0 experiment: switch active model logreg → xgboost on with_weather
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
5. **Финальный scored deliverable** обязателен для отчёта — это
   человекочитаемый Excel с обеими головами; обновлять каждый раз, когда
   меняется лидер любой из голов.
