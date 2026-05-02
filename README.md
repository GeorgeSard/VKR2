# Flight Delay MLOps — ВКР

Магистерская ВКР: **«Разработка модели искусственного интеллекта для прогнозирования задержек авиарейсов и классификации их причин»**.

Главный результат — **выстроенный ML-процесс** (data → DVC → features → training → MLflow → FastAPI → monitoring → feedback loop), а не отдельная модель.

> Цитата руководителя: «Если вы не построили цикл — вы ничего не сделали.»

Подробные правила работы — в [CLAUDE.md](./CLAUDE.md). Краткая инструкция по запуску — в [QUICKSTART.md](./QUICKSTART.md).

---

## 1. Что в проекте решается

Две ML-задачи на одних и тех же фичах:

| Голова | Тип | Что предсказывает |
|---|---|---|
| **A — binary** | бинарная классификация | Будет ли рейс задержан > 15 мин |
| **B — cause** | multiclass (7 классов) | Причина задержки: weather, carrier_operational, airport_congestion, reactionary, security, cancelled, none |

**Тип данных — табличные**, поэтому стек строится вокруг градиентных бустингов (XGBoost, LightGBM), а не нейросетей. Глубокое обучение и LLM — запрещены руководителем (см. CLAUDE.md §5.4).

---

## 2. Технологический стек — что для чего

| Слой | Инструмент | Зачем |
|---|---|---|
| Версионирование кода | **git** | стандарт |
| Версионирование данных | **DVC** | вместо git LFS — даёт воспроизводимый snapshot датасета по hash'у |
| Пайплайн | **DVC stages** | `dvc repro` пересобирает всё от raw до метрик одной командой |
| Трекинг экспериментов | **MLflow** | каждый run автоматически тегается git-коммитом + DVC-hash'ем данных |
| Реестр моделей | **MLflow Model Registry** | лидеры зарегистрированы как версии — API грузит «models:/flight-delay-binary/1» |
| Подбор гиперпараметров | **Optuna** | TPE сэмплер, умнее GridSearch |
| Сервис инференса | **FastAPI** | + auto-swagger, + Pydantic валидация |
| Контейнеризация | **Docker + docker-compose** | mlflow + api поднимаются одной командой |
| Мониторинг | **prometheus_client + structured JSON logs** | счётчики + гистограммы + распределение предсказаний |
| ML-библиотеки | scikit-learn, XGBoost, LightGBM, CatBoost, SHAP | классические бустинги доминируют на табличке |

### Что такое DVC простыми словами

**DVC (Data Version Control)** — это **отдельный CLI-инструмент** (`pip install dvc`), не встроенный в git. Работает поверх git:

- В git идут только маленькие текстовые файлы-указатели `*.dvc` (содержат hash и размер настоящего файла)
- Сами тяжёлые файлы (parquet датасеты, обученные модели, артефакты) живут в **DVC remote** — это просто папка `~/.dvc-storage/vkr2-flight-delays` на твоём диске (можно настроить S3, gdrive)
- Команда `dvc pull` тянет файлы из remote в `data/raw/` (как `git pull` для данных)
- Команда `dvc add data/raw/file.parquet` добавляет файл в DVC — git увидит только новый `file.parquet.dvc` ~~100 байт

**Зачем нужно:** воспроизводимость экспериментов. Любой коллега делает `git clone + dvc pull` и получает **идентичный датасет той же версии**, на котором обучалась модель. Без DVC невозможно сказать «вот эти метрики получены на этих данных» — данные могли поменяться.

Помимо версионирования, DVC даёт **пайплайн** (`dvc.yaml` + `dvc repro`) — описываешь стадии и зависимости, DVC пересобирает только то, что изменилось.

### Что такое MLflow

**MLflow** — отдельный сервис (`pip install mlflow`, поднимается как HTTP-сервер на порту 5000). Делает три вещи:
1. **Tracking** — каждый запуск обучения вызывает `mlflow.log_params() / log_metrics() / log_artifacts()`, всё пишется в file backend (`mlruns/`)
2. **Registry** — лучшие модели регистрируются как именованные версии (`flight-delay-binary` v1, v2, …), сервис грузит «по имени и версии» без знания путей
3. **UI** — веб-интерфейс на http://localhost:5001, видны все 13 runs, можно сравнивать через Compare

Аналог git, но для моделей и экспериментов.

### Что такое FastAPI и Docker

- **FastAPI** — Python-фреймворк для HTTP API. Описываешь функцию, декорируешь `@app.post(...)` — получаешь endpoint + auto-документацию в `/docs`.
- **Docker** — упаковывает приложение со всеми зависимостями в образ. `docker compose up` поднимает оба сервиса (mlflow + api) с одной команды на любой машине.

---

## 3. Жизненный цикл ML — что выстроено

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Domain analysis  →  2. Data collection                   │
│        ↑                         ↓                          │
│ 8. Feedback loop    ←   3. DVC versioning                   │
│   (POST /feedback)               ↓                          │
│        ↑                4. Feature engineering              │
│ 7. Monitoring               (3 наборов фичей)               │
│  (Prometheus +                   ↓                          │
│   JSON logs)            5. DVC pipeline (5 стадий)          │
│        ↑                         ↓                          │
│ 6. Deployment       ←   5b. Training + Optuna tuning        │
│   (FastAPI+Docker)         (MLflow tracking, 13 runs)       │
└─────────────────────────────────────────────────────────────┘
```

Каждая стрелка реализована конкретным кодом и закоммичена в git как отдельный логический шаг (см. `git log --oneline`).

---

## 4. Структура репо

```
ВКР2/
├── CLAUDE.md                     ← правила проекта (для Claude/AI-наставника)
├── HANDOFF.md                    ← статус между сессиями
├── QUICKSTART.md                 ← короткая инструкция по запуску
├── README.md                     ← этот файл
│
├── docker-compose.yml            ← стек: mlflow + api
├── docker/                       ← Dockerfile'ы
│
├── params.yaml                   ← ★ все гиперпараметры экспериментов (см. раздел 7)
├── dvc.yaml                      ← пайплайн: ingest → split → featurize → train → evaluate
├── dvc.lock                      ← снапшот hash'ей входов/выходов (как poetry.lock)
│
├── src/
│   ├── api/                      ← FastAPI сервис
│   │   ├── main.py               ← регистрация всех 6 ручек + middleware
│   │   ├── schemas.py            ← Pydantic-модели запроса/ответа
│   │   └── inference.py          ← ModelStore: грузит модели из MLflow Registry
│   ├── data/                     ← ingest, time-based split
│   ├── features/                 ← 3 feature sets (basic / extended / with_weather)
│   ├── models/
│   │   ├── train.py              ← главный trainer (MLflow tracking)
│   │   ├── tune.py               ← Optuna tuning (xgb/lgbm × binary/cause)
│   │   ├── registry.py           ← регистрация лидеров в Model Registry
│   │   ├── evaluate.py           ← метрики обеих голов
│   │   └── score_dataset.py      ← финальный scored test dataset (Excel)
│   ├── monitoring/
│   │   ├── logger.py             ← JSON-логи
│   │   ├── metrics.py            ← Prometheus
│   │   └── feedback.py           ← /feedback → parquet sink
│   └── demo/
│       └── feedback_to_training.py  ← конвертер feedback → формат raw
│
├── scripts/
│   └── demo_feedback_cycle.py    ← end-to-end демо замыкания цикла
│
├── data/                         ← gitignored, под управлением DVC
│   ├── raw/                      ← исходный parquet
│   ├── interim/                  ← после ingest (очистка)
│   ├── processed/                ← train/val/test split
│   ├── featurized/               ← X/y матрицы для обучения
│   └── feedback/                 ← parquet, куда пишет POST /feedback
│
├── models/                       ← gitignored, лучшие Optuna params
├── mlruns/                       ← gitignored, MLflow tracking + Registry
│
└── reports/
    ├── experiments_log.md                     ← журнал всех 13 runs с дельтами
    ├── DVC_SCREENSHOTS_GUIDE.md               ← инструкция: какие скрины DVC делать
    ├── MLFLOW_SCREENSHOTS_GUIDE.md            ← инструкция: какие скрины MLflow делать
    ├── SCORED_DATASET_README.md               ← описание финального scored датасета
    ├── scored_test_dataset.{csv,parquet,xlsx} ← финальный артефакт (для отчёта)
    ├── val_metrics.json, test_metrics.json    ← метрики последнего dvc repro
    └── confusion_matrix.csv                   ← cause head test confusion
```

---

## 5. Как запустить (с нуля)

### Два независимых «стека» — какой когда нужен

В проекте **два контура**, они нужны для разных вещей и **запускаются по-разному**. Не перепутай — это самая частая ошибка.

| Контур | Чем поднимается | Для чего |
|---|---|---|
| **Локальный Python (`.venv`)** | `source .venv/bin/activate` | Обучение моделей, DVC pipeline, эксперименты, запуск demo-скриптов, чтение parquet'ов |
| **Docker-стек (`docker compose`)** | `cd /tmp/vkr2-build && docker compose up -d` | Сервинг готовой модели через FastAPI + MLflow UI |

Они **независимы и могут работать одновременно**: локально ты экспериментируешь и обучаешь → когда модель хороша, регистрируешь её в MLflow Registry → Docker-API подхватит при следующем рестарте контейнера.

### Первый раз — установка

```bash
# 1. Клон
git clone https://github.com/GeorgeSard/VKR2.git
cd VKR2

# 2. Локальный venv для тренировок и DVC
uv venv && source .venv/bin/activate
uv pip install -e .

# 3. Подтянуть данные через DVC (положит parquet в data/raw/)
dvc pull

# 4. Подтянуть Docker (если ещё нет): открой Docker Desktop и дождись зелёного daemon

# 5. Поднять Docker-стек для API
#    Кириллица в пути ВКР2 ломает BuildKit, обходим через ASCII-симлинк
ln -sfn "$PWD" /tmp/vkr2-build
cd /tmp/vkr2-build
docker compose up -d

# 6. Проверка обоих контуров
curl -s http://localhost:8000/health    # API → {"status":"ok",...}
.venv/bin/python -c "import mlflow; print(mlflow.__version__)"   # локальный venv
```

### Каждая следующая сессия

```bash
# Если нужен только API (чтобы потыкать ручки или показать комиссии)
cd /tmp/vkr2-build && docker compose up -d
# Открыть в браузере: http://localhost:8000/docs (API), http://localhost:5001 (MLflow UI)

# Если будешь обучать / запускать эксперименты — параллельно активируй venv
cd /Users/georgij/Projects/ВКР2 && source .venv/bin/activate
```

---

## 6. FastAPI — все 6 ручек с примерами

После `docker compose up -d` поднимаются два контейнера: `flight-delay-api` (FastAPI на порту 8000, описан ниже) и `flight-delay-mlflow` (MLflow tracking server на 5001). Все шесть ручек ниже — в API-контейнере на http://localhost:8000.

**Интерактивная документация (Swagger UI)** — http://localhost:8000/docs, можно тыкать любую ручку прямо в браузере.

### `GET /health` — жив ли API-контейнер и обе ли модели в памяти

```bash
curl -s http://localhost:8000/health
```
```json
{"status":"ok","binary_loaded":true,"cause_loaded":true}
```

**Что показывает:** под «сервисом» здесь имеется в виду **сам FastAPI-контейнер `flight-delay-api`** (тот, что слушает 8000-й порт). Endpoint отвечает на два вопроса:
- процесс uvicorn внутри контейнера принимает HTTP — раз вернулся ответ, значит да;
- `binary_loaded` / `cause_loaded` — обе ли модели подтянулись из MLflow Registry на старте (если `false` — модель не зарегистрирована или mlflow контейнер недоступен → `/predict/*` будут отдавать 503).

Этот endpoint дёргает Docker healthcheck каждые 15 секунд (см. `HEALTHCHECK` в `docker/api.Dockerfile`). Если три проверки подряд провалились, контейнер помечается как `unhealthy` и оркестратор (compose / k8s) может его перезапустить.

---

### `GET /model/info` — какая модель сейчас обслуживает

```bash
curl -s http://localhost:8000/model/info | python3 -m json.tool
```
```json
{
  "binary": {
    "name": "flight-delay-binary",
    "version": "1",
    "run_id": "4227cd7b...",
    "git_commit": "b3dc9c8",
    "dvc_data_hash": "89f1acd4cfba",
    "feature_set": "with_weather",
    "metrics": {"f1": 0.630, "roc_auc": 0.839, ...}
  },
  "cause": {...}
}
```

**Что показывает:** полную трассируемость каждой модели в production:
- `version` — версия в MLflow Registry
- `run_id` — конкретный эксперимент в MLflow UI
- `git_commit` — на каком коммите кода обучалась
- `dvc_data_hash` — на какой версии данных обучалась
- `metrics` — метрики на validation, с которыми деплоилась

**На защите:** доказывает воспроизводимость — можешь по любой модели в проде восстановить точную версию кода + данных.

---

### `POST /predict/delay` — Голова A (бинарная)

```bash
curl -s -X POST http://localhost:8000/predict/delay \
  -H 'Content-Type: application/json' \
  -d '{
    "month":7,"day_of_week":3,"scheduled_dep_hour":14,"scheduled_dep_minute":30,
    "is_weekend":0,"is_holiday_window":0,"quarter":3,
    "distance_km":2300,"planned_block_minutes":195,"airline_fleet_avg_age":12.5,
    "origin_hub_tier":1,"destination_hub_tier":2,"inbound_delay_minutes":0,
    "origin_congestion_index":0.45,"destination_congestion_index":0.32,
    "origin_temperature_c":18,"origin_precip_mm":0,"origin_visibility_km":10,"origin_wind_mps":3.5,
    "destination_temperature_c":22,"destination_precip_mm":0,"destination_visibility_km":10,"destination_wind_mps":2.8,
    "airline_code":"SU","aircraft_family":"A320","origin_iata":"SVO","destination_iata":"LED",
    "route_group":"domestic_trunk","origin_weather_severity":"calm","destination_weather_severity":"calm"
  }'
```
```json
{
  "is_delayed": true,
  "delay_probability": 0.8836,
  "model_name": "flight-delay-binary",
  "model_version": "1"
}
```

**Как читать:**
- `is_delayed: true` — модель ожидает задержку > 15 мин
- `delay_probability: 0.88` — уверенность 88%. Threshold 0.5 → `is_delayed`
- `model_version` — какая версия в Registry дала ответ (для аудита)

В заголовке ответа: `X-Request-ID: 47506c3b...` — нужен для последующего `/feedback`.

---

### `POST /predict/cause` — Голова B (multiclass)

Принимает тот же payload, что `/predict/delay`.
```json
{
  "predicted_cause": "weather",
  "class_probabilities": {
    "airport_congestion": 0.00006,
    "cancelled": 0.0069,
    "carrier_operational": 0.1536,
    "none": 0.2564,
    "reactionary": 0.00003,
    "security": 0.0138,
    "weather": 0.5691
  },
  "model_name": "flight-delay-cause",
  "model_version": "1"
}
```

**Как читать:**
- `predicted_cause` — argmax по `class_probabilities`
- Сумма вероятностей ≈ 1
- Если `none` > 0.5 — модель считает что задержки не будет (полезно скрестить с ответом `/predict/delay`)

---

### `POST /feedback` — реальная обратная связь (замыкание цикла)

После реального вылета посылается факт. Используй `request_id` из заголовка `/predict/*`.
```bash
curl -s -X POST http://localhost:8000/feedback \
  -H 'Content-Type: application/json' \
  -d '{
    "request_id":"47506c3b...",
    "actual_is_delayed":true,
    "actual_delay_minutes":42,
    "actual_cause":"weather"
  }'
```
```json
{"stored":true,"total_records":21}
```

**Что происходит:** запись добавляется в `data/feedback/feedback.parquet` (на хосте через bind-mount, переживает рестарт). Дальше скрипт `src/demo/feedback_to_training.py` мержит её с фичами и готовит `next_round.parquet` для следующего `dvc repro` — **новые данные → новая модель**.

---

### `GET /metrics` — Prometheus

```bash
curl -s http://localhost:8000/metrics | head -30
```
```
api_requests_total{endpoint="/predict/delay",method="POST",status="200"} 1.0
api_request_duration_seconds_bucket{endpoint="/predict/delay",method="POST",le="0.05"} 1.0
api_predictions_total{model="binary",outcome="delayed"} 1.0
api_predictions_total{model="cause",outcome="weather"} 1.0
```

**Три семейства метрик:**
1. `api_requests_total` — счётчик запросов по эндпоинту/методу/статусу (включая 422 ошибки валидации)
2. `api_request_duration_seconds` — гистограмма latency по бакетам
3. `api_predictions_total` — распределение предсказаний по классам — **главная метрика для отлова дрейфа модели** (если внезапно 90% предсказаний стало `weather` — модель скорее всего деградировала)

В реальной инсталляции сюда подключается Prometheus + Grafana, на защите достаточно показать что endpoint живой и метрики растут после curl-запросов.

---

### Бонус: один скрипт прогоняет весь цикл

```bash
.venv/bin/python scripts/demo_feedback_cycle.py --n 20
```

Берёт 20 случайных рейсов из test split → шлёт через обе головы → постит обратно реальные labels → печатает batch accuracy + tail parquet. На seed=42: binary 85%, cause 60% — совпадает с тестовыми метриками главы 3.

---

## 7. Где менять параметры

**★ Главный файл — `params.yaml`** в корне репо. Там собрано **всё**, что можно крутить:

```yaml
base:
  random_seed: 42        ← воспроизводимость
  params_version: v1     ← тег для группировки runs

split:
  strategy: time_based   ← или random
  train_end_date: "2024-12-31"
  val_end_date: "2025-06-30"

features:
  active_set: with_weather   ← ★ переключатель: basic / extended / with_weather

train:
  active_model: xgboost      ← ★ переключатель: logreg / xgboost / lightgbm / random_forest / catboost
  task: delay_cause          ← ★ delay_binary или delay_cause

  xgboost:                   ← гиперпараметры конкретной модели
    n_estimators: 800
    max_depth: 8
    learning_rate: 0.05
    ...
```

**Сценарий «провести новый эксперимент»:**
```bash
# 1. Поправить нужное поле в params.yaml (например, active_model: lightgbm)
# 2. Запустить пайплайн
dvc repro
# 3. Закоммитить
git commit -am "experiment: switch to lightgbm with with_weather"
# 4. Посмотреть результат
dvc metrics show              # текущие метрики
dvc metrics diff HEAD~1 HEAD  # дельта от предыдущего коммита
```

DVC увидит, что изменился `params.yaml`, и пересоберёт **только зависящие стадии** (featurize → train → evaluate), не трогая ingest/split.

**Optuna best params** живут отдельно в `models/best_*_params.yaml` (gitignored, регенерируемые). Чтобы воспроизвести Run #6:
```bash
python -m src.models.tune --n-trials 30                       # binary xgboost
python -m src.models.tune --task delay_cause --model xgboost  # cause head
```

---

## 8. Как обучить модель (на дефолтных или своих данных)

### Сценарий A — на текущем датасете (тот, что в DVC remote)

Самый частый случай: ты ничего не меняешь в данных, только хочешь перепрогнать обучение или попробовать другую модель / набор фичей.

```bash
# 1. Активируй venv (Docker-стек тут НЕ нужен — обучение идёт в локальном Python)
cd /Users/georgij/Projects/ВКР2 && source .venv/bin/activate

# 2. Убедись что данные на месте
ls data/raw/                # должен быть flight_delays_ru.parquet
# если пусто — dvc pull

# 3. (опционально) Поправь params.yaml — например, active_model: lightgbm

# 4. Запусти полный пайплайн через DVC
dvc repro

# Эквивалент без DVC (для разовой пробы):
python -m src.models.train
```

`dvc repro` выполнит стадии: `ingest → split → featurize → train → evaluate`. Каждая логирует своё в MLflow автоматически.

### Сценарий B — на твоих собственных данных

Есть **два варианта**, в зависимости от того, насколько твой датасет похож на текущий.

#### B1. Твой parquet в той же схеме (68 колонок, см. `DATA_DICTIONARY.md`)

```bash
# 1. Положи свой файл по тому же пути (имя то же)
cp /path/to/your_flights.parquet data/raw/flight_delays_ru.parquet

# 2. Перепрогон
dvc repro

# 3. Зафиксировать новую версию данных в DVC
dvc add data/raw/flight_delays_ru.parquet
dvc push   # положит файл в local-storage, .dvc-метаданные пойдут в git
git add data/raw/flight_delays_ru.parquet.dvc params.yaml
git commit -m "data: switch to my dataset, retrain"
```

Требования к схеме (минимум):
- Колонка `flight_id` (уникальный ключ)
- Колонки расписания: `flight_date`, `scheduled_departure_local`, `scheduled_arrival_local`
- Целевые: `is_departure_delayed_15m` (0/1), `dep_delay_minutes` (число), `probable_delay_cause` (строка из 7 классов)
- Все 28 фичей, что перечислены в `src/api/schemas.py::FlightFeatures` (это те же колонки, что принимает API)

Если каких-то колонок нет — `dvc repro` упадёт на стадии `ingest` или `featurize` с понятной ошибкой про missing column.

#### B2. Твой датасет в другой схеме (произвольный CSV/parquet)

Тогда нужно либо:
- **Адаптировать данные под схему проекта** (переименовать колонки, добавить недостающие как заглушки) — самый быстрый путь;
- **Или поправить `src/data/ingest.py`** под свою схему + обновить `src/features/feature_sets.py` — это уже отдельная работа, не пять минут.

Для проверки концепции рекомендую **B1**: подгони свой файл под текущую схему.

#### B3. Сгенерировать новый синтетический датасет (для теста)

Если просто хочется убедиться что цикл работает на «свежих» данных:
```bash
python -m src.data.generate --rows 50000 --seed 100 --out data/raw/flight_delays_ru.parquet
dvc repro
```
Получишь новый снапшот данных + перепрогон всего пайплайна → новые метрики.

### Что произойдёт при `dvc repro`

```
✓ ingest      raw parquet → data/interim/flight_delays_clean.parquet
✓ split       train/val/test parquet'ы по датам
✓ featurize   X/y матрицы под выбранный feature_set + manifest
✓ train       обучение модели → models/dvc_model.pkl + reports/val_metrics.json
✓ evaluate    оценка на test → reports/test_metrics.json + confusion_matrix.csv
```

Каждая стадия запускается **только если её входы или params изменились** (DVC хранит хэши в `dvc.lock`). Если ты поправил только `train.active_model`, ingest и split не перезапустятся — экономия времени.

### Где увидеть результат обучения

После `dvc repro` метрики появляются в **трёх местах одновременно** (это намеренно — каждое для своей задачи):

| Где | Что показывает | Как открыть |
|---|---|---|
| **Файлы в `reports/`** | val + test метрики свежего прогона | `cat reports/test_metrics.json` |
| **`dvc metrics show`** | те же метрики таблицей в терминале | `dvc metrics show` |
| **`dvc metrics diff`** | дельта от предыдущего git-коммита | `dvc metrics diff HEAD~1 HEAD` |
| **MLflow UI** | детальный run: параметры, артефакты, plots | http://localhost:5001 → experiment `flight-delay-mlops` |
| **Свежий scored dataset (Excel)** | предсказания обеих голов на каждом test-рейсе с маркерами правильности | `python -m src.models.score_dataset` → `reports/scored_test_dataset.xlsx` |

**Если MLflow UI не запущен** (например, ты не поднимал Docker-стек):
```bash
mlflow ui --backend-store-uri "file:$PWD/mlruns" --host 127.0.0.1 --port 5050
# открыть http://127.0.0.1:5050
```

### Зарегистрировать новую модель в Registry, чтобы её подхватил API

После хорошего эксперимента:
```bash
# 1. Найди run_id своего лучшего прогона в MLflow UI
# 2. Поправь src/models/registry.py — там в RUN_IDS прописать новый run_id
# 3. Запусти регистрацию
python -m src.models.registry
# → создастся flight-delay-binary v2 (или v3, v4...)

# 4. Перезапусти API контейнер чтобы он подхватил свежую версию
cd /tmp/vkr2-build && docker compose restart api
curl -s http://localhost:8000/model/info    # увидишь "version": "2"
```

### Краткая шпаргалка: «полный цикл за один присест»

```bash
source .venv/bin/activate

# Поменять параметры эксперимента
vim params.yaml                  # например, active_model: lightgbm

# Перепрогнать
dvc repro

# Посмотреть метрики в терминале
dvc metrics show

# Посмотреть в MLflow UI
open http://localhost:5001       # уже поднят через docker compose

# Если результат хорош — закоммитить
git add params.yaml dvc.lock reports/*.json
git commit -m "experiment: lightgbm with with_weather, F1 0.629"

# И зарегистрировать как новую версию для API
python -m src.models.registry
docker compose -f /tmp/vkr2-build/docker-compose.yml restart api
```

---

## 9. Где смотреть результаты

| Что | Где |
|---|---|
| Все 13 экспериментов | http://localhost:5001 → `flight-delay-mlops` |
| Сравнение runs (parallel coords) | в MLflow UI выделить runs галочками → Compare |
| Метрики последнего dvc repro | `dvc metrics show` (val + test одной таблицей) |
| Дельта между коммитами | `dvc metrics diff HEAD~1 HEAD` |
| Граф пайплайна (для скрина) | `dvc dag` |
| Журнал экспериментов с интерпретацией | `reports/experiments_log.md` |
| Финальный scored Excel (3 листа) | `reports/scored_test_dataset.xlsx` |
| Confusion matrix cause head | `reports/confusion_matrix.csv` |
| Карточка датасета | `DATASET_CARD.md`, `DATA_DICTIONARY.md`, `DATA_QUALITY_REPORT.md` |

---

## 10. Главные результаты для защиты

### Доказанная гипотеза руководителя

> «Если перебор гиперпараметров ничего не даёт — проблема в данных. Если даёт — в архитектуре.»

**Доказано двумя независимыми путями:**

| Голова | Plateau | Достигнут | Подтверждено |
|---|---|---|---|
| Binary | F1 = 0.63, ROC-AUC = 0.839 | xgb (Run #6, optuna 30 trials) | lgbm (Run #8, optuna 30 trials) — те же числа |
| Cause | macro_f1 = 0.36 | xgb defaults (C4) | xgb optuna (C5) — улучшения 0.001 |

**Вывод:** ROI смены модели исчерпан. Headroom только в новых данных → именно поэтому в Stage 9 построен feedback loop.

### Что показывать комиссии (по чек-листу CLAUDE.md §9)

1. `git log --oneline` — видна история всех 10 этапов
2. `dvc dag` — граф пайплайна
3. MLflow UI — 13 runs, два registered models, очевидный лидер
4. `docker compose up` — стек поднимается за минуту
5. `curl POST /predict/delay` — живой ответ
6. `curl GET /metrics` — Prometheus метрики
7. `.venv/bin/python scripts/demo_feedback_cycle.py --n 20` — замыкание цикла на ваших глазах
8. `reports/scored_test_dataset.xlsx` — финальный артефакт (3 листа Excel)
9. `reports/experiments_log.md` — журнал экспериментов с дельтами

### Headline-метрики на test split

- **Голова A (binary):** accuracy **81.9 %** на оценимых рейсах
- **Голова B (cause):** accuracy **69.6 %** в целом по 7 классам

Полностью воспроизводимо командой `python -m src.models.score_dataset`.

---

## 11. Что осталось вне scope (намеренно)

- **Главы 1-2 ВКР** (предметная область + state of the art) — текстовая часть, не код
- **Catboost тюнинг** — два других бустинга показали plateau, дополнительная библиотека ничего не докажет
- **Grafana dashboards** — Prometheus endpoint готов, но Grafana не поднимается; в реальности это 5 минут работы DevOps
- **pytest unit-tests** — пропущены в пользу скорости. Smoke через TestClient + curl против compose покрывает критичный путь
- **CatBoost / NN** — запрещены руководителем (CLAUDE.md §5.4)

---

## Полезные ссылки внутри репо

- [QUICKSTART.md](./QUICKSTART.md) — быстрый старт, как запустить и где что
- [HANDOFF.md](./HANDOFF.md) — статус проекта между сессиями
- [CLAUDE.md](./CLAUDE.md) — философия и правила
- [reports/experiments_log.md](./reports/experiments_log.md) — детальный журнал 13 экспериментов
- [reports/SCORED_DATASET_README.md](./reports/SCORED_DATASET_README.md) — как читать финальный Excel
- [reports/DVC_SCREENSHOTS_GUIDE.md](./reports/DVC_SCREENSHOTS_GUIDE.md) — какие скрины DVC делать для отчёта
- [reports/MLFLOW_SCREENSHOTS_GUIDE.md](./reports/MLFLOW_SCREENSHOTS_GUIDE.md) — какие скрины MLflow делать для отчёта
