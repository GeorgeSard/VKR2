# QUICKSTART — как запустить и что где лежит

## 1. Запуск всего стека (mlflow + api)

Кириллица в пути `ВКР2` ломает Docker BuildKit, поэтому собираем через ASCII-симлинк:

```bash
ln -sfn /Users/georgij/Projects/ВКР2 /tmp/vkr2-build
cd /tmp/vkr2-build

docker compose up -d           # стек уже собран → поднимется за ~30 сек
docker compose ps              # оба должны быть (healthy)
```

Остановить:
```bash
cd /tmp/vkr2-build && docker compose down
```

Пересобрать после изменений в коде:
```bash
cd /tmp/vkr2-build && docker compose build api && docker compose up -d
```

## 2. Что где открывать в браузере

| URL | Что это |
|---|---|
| http://localhost:8000/docs | Swagger UI — интерактивная документация API, можно дёргать ручки прямо из браузера |
| http://localhost:8000/redoc | то же самое в стиле ReDoc |
| http://localhost:5001 | MLflow UI — все 13 экспериментов + Model Registry |

(MLflow на 5001, потому что macOS AirPlay занимает 5000.)

## 3. Ручки FastAPI — что делают

Все 6 ручек сервиса:

### `GET /health`
Жив ли сервис и обе ли модели загружены в память.
```bash
curl -s http://localhost:8000/health
# {"status":"ok","binary_loaded":true,"cause_loaded":true}
```
Используется Docker healthcheck'ом и для k8s liveness probe.

### `GET /model/info`
Метаданные обеих моделей: версия в Registry, run_id из MLflow, git-коммит, на котором обучена, hash датасета DVC, метрики на val.
```bash
curl -s http://localhost:8000/model/info | python3 -m json.tool
```
Зачем: понять «какая модель сейчас обслуживает запросы» без захода в MLflow UI.

### `POST /predict/delay` — Голова A (бинарная)
Будет ли рейс задержан > 15 минут.
```bash
curl -s -X POST http://localhost:8000/predict/delay \
  -H 'Content-Type: application/json' \
  -d @payload.json
# {"is_delayed":true,"delay_probability":0.8836,"model_name":"flight-delay-binary","model_version":"1"}
```
Пример `payload.json` — ниже.

### `POST /predict/cause` — Голова B (multiclass)
Какая причина задержки наиболее вероятна (7 классов: weather, carrier_operational, airport_congestion, reactionary, security, cancelled, none).
```bash
curl -s -X POST http://localhost:8000/predict/cause \
  -H 'Content-Type: application/json' \
  -d @payload.json
# {"predicted_cause":"weather","class_probabilities":{...},...}
```

### `POST /feedback` — обратная связь от реальной жизни
Замыкает ML-цикл: после реального вылета приходит факт (была ли задержка, какая причина), записывается в `data/feedback/feedback.parquet`. Эти записи — семя для следующей итерации DVC (новый train split → переобучение).
```bash
# request_id берём из заголовка X-Request-ID, который вернул /predict
curl -s -X POST http://localhost:8000/feedback \
  -H 'Content-Type: application/json' \
  -d '{"request_id":"abc123","actual_is_delayed":true,"actual_delay_minutes":42,"actual_cause":"weather"}'
# {"stored":true,"total_records":1}
```

### `GET /metrics` — Prometheus
Текстовый формат для Prometheus scrape. Три семейства метрик: счётчик запросов по эндпоинтам, гистограмма latency, распределение предсказаний по классам (для отлова дрейфа модели).
```bash
curl -s http://localhost:8000/metrics | head -30
```

### Пример payload (`payload.json`)
30 полей рейса. Пример: рейс Аэрофлот SU SVO→LED, A320, лето, штатные погодные условия.
```json
{
  "month":7,"day_of_week":3,"scheduled_dep_hour":14,"scheduled_dep_minute":30,
  "is_weekend":0,"is_holiday_window":0,"quarter":3,
  "distance_km":2300,"planned_block_minutes":195,"airline_fleet_avg_age":12.5,
  "origin_hub_tier":1,"destination_hub_tier":2,"inbound_delay_minutes":0,
  "origin_congestion_index":0.45,"destination_congestion_index":0.32,
  "origin_temperature_c":18,"origin_precip_mm":0,"origin_visibility_km":10,"origin_wind_mps":3.5,
  "destination_temperature_c":22,"destination_precip_mm":0,"destination_visibility_km":10,"destination_wind_mps":2.8,
  "airline_code":"SU","aircraft_family":"A320",
  "origin_iata":"SVO","destination_iata":"LED","route_group":"domestic_trunk",
  "origin_weather_severity":"calm","destination_weather_severity":"calm"
}
```

Полный список полей и их типы — на http://localhost:8000/docs (раздел Schemas → FlightFeatures).

## 4. Логи и отладка

```bash
cd /tmp/vkr2-build

docker compose logs api          # логи FastAPI (загрузка моделей + запросы)
docker compose logs mlflow       # логи MLflow tracking server
docker compose logs -f api       # follow в реальном времени
docker compose restart api       # рестарт только api без пересборки
```

## 5. Где что в репо

```
ВКР2/
├── CLAUDE.md                     ← правила проекта (для Claude)
├── HANDOFF.md                    ← статус между сессиями (для Claude)
├── QUICKSTART.md                 ← этот файл (для тебя)
├── README.md                     ← описание ВКР целиком
│
├── docker-compose.yml            ← стек: mlflow + api
├── docker/
│   ├── api.Dockerfile            ← сборка FastAPI image
│   └── mlflow.Dockerfile         ← сборка MLflow tracking server
│
├── params.yaml                   ← все гиперпараметры экспериментов
├── dvc.yaml                      ← пайплайн: ingest → split → featurize → train → evaluate
│
├── src/
│   ├── api/                      ← FastAPI сервис
│   │   ├── main.py               ← регистрация ручек
│   │   ├── schemas.py            ← Pydantic модели запроса/ответа
│   │   └── inference.py          ← загрузка моделей из MLflow Registry
│   ├── data/                     ← ingest, split
│   ├── features/                 ← feature engineering (basic/extended/with_weather)
│   ├── models/
│   │   ├── train.py              ← основной trainer (MLflow tracking)
│   │   ├── tune.py               ← Optuna hyperparam tuning
│   │   ├── registry.py           ← регистрация лидеров в Model Registry
│   │   ├── evaluate.py           ← метрики
│   │   └── score_dataset.py      ← финальный scored test dataset
│   └── monitoring/               ← Этап 9: structured logger, Prometheus, feedback sink
│       ├── logger.py             ← JSON-логи на stdout (для Loki/ELK)
│       ├── metrics.py            ← prometheus_client: requests, latency, predictions
│       └── feedback.py           ← append-only parquet sink под /feedback
│
├── data/                         ← gitignored, под управлением DVC
├── models/                       ← gitignored, лучшие Optuna params
├── mlruns/                       ← gitignored, MLflow tracking + Registry
│
├── reports/
│   ├── experiments_log.md                 ← журнал всех 13 runs
│   ├── DVC_SCREENSHOTS_GUIDE.md           ← гайд: какие скрины DVC делать
│   ├── MLFLOW_SCREENSHOTS_GUIDE.md        ← гайд: какие скрины MLflow делать
│   ├── SCORED_DATASET_README.md           ← описание финального scored датасета
│   ├── scored_test_dataset.{csv,parquet,xlsx}  ← финальный артефакт (для отчёта)
│   ├── val_metrics.json, test_metrics.json     ← последние метрики DVC stage
│   └── confusion_matrix.csv                    ← cause head test confusion
│
└── DATASET_CARD.md, DATA_DICTIONARY.md, DATA_QUALITY_REPORT.md  ← карточки данных
```

## 6. End-to-end демо замыкания цикла (Этап 10)

Один скрипт показывает живой ML-процесс: prediction → реальная разметка → seed для следующего dvc round.

```bash
# 1. Поднять стек если ещё не поднят
cd /tmp/vkr2-build && docker compose up -d

# 2. Прогнать демо: 20 случайных рейсов из test → /predict → /feedback
.venv/bin/python scripts/demo_feedback_cycle.py --n 20

# Покажет accuracy на батче (binary ~85%, cause ~60%) и tail feedback parquet.

# 3. Сконвертировать накопленный feedback в формат raw — готово к следующему dvc round
.venv/bin/python -m src.demo.feedback_to_training

# Создаёт data/feedback/next_round.parquet с теми же 68 колонками что raw,
# плюс label_source=feedback и label_received_at для аудита.
```

После этого в реальном проекте:
1. `cp data/feedback/next_round.parquet data/raw/flight_delays_feedback.parquet`
2. дополнить `src/data/ingest.py` чтобы читал оба файла
3. `dvc repro` — пересобирает featurize/train/evaluate
4. `dvc metrics diff HEAD` — видна дельта от свежей разметки

## 7. Частые операции локально (без Docker)

```bash
source .venv/bin/activate

# MLflow UI на хосте (без контейнера)
mlflow ui --backend-store-uri "file:$PWD/mlruns" --host 127.0.0.1 --port 5050

# Воспроизвести весь DVC pipeline
dvc repro                        # пересобрать только изменённые стадии
dvc metrics show                 # последние val + test метрики

# Обучить модель с текущим params.yaml
python -m src.models.train

# Optuna тюнинг (binary голова, 30 trials)
python -m src.models.tune --n-trials 30

# Перегенерировать финальный scored test dataset
python -m src.models.score_dataset
```

## 8. Если что-то сломалось

| Симптом | Что делать |
|---|---|
| `port 5000: address already in use` | macOS AirPlay; в compose host-порт mlflow уже на 5001. Если взял другой порт — System Settings → General → AirDrop & Handoff → AirPlay Receiver OFF |
| api контейнер падает на старте «No such file or directory ... mlruns» | Проверить bind-mount в `docker-compose.yml`: `./mlruns:/Users/georgij/Projects/ВКР2/mlruns` |
| `non-printable ASCII characters` при `docker compose build` | Кириллица в пути. Собирать через `/tmp/vkr2-build` симлинк (см. п. 1) |
| `/health` отдаёт `binary_loaded:false` | Модели в Registry не зарегистрированы. Запустить `python -m src.models.registry` из активированного venv на хосте |
