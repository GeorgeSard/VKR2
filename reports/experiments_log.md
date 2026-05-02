# Experiments Log

Журнал экспериментов в хронологическом порядке. Источник правды — MLflow
(`mlflow ui --backend-store-uri file:./mlruns --port 5000`,
эксперимент `flight-delay-mlops`); этот файл — человекочитаемая сводка
для отчёта.

**Постановка для всех runs ниже:** `task = delay_binary` (предсказать
`is_departure_delayed_15m`), валидация на time-based slice
**Jan–Jun 2025** (n=36 082 после удаления отменённых рейсов),
обучение на ≤ Dec 2024 (n=145 873). Random seed 42 везде.
Дисбаланс классов train: **76% вовремя / 24% задержано**.

## Сводная таблица

| # | mlflow run_id | git_commit | feature_set | model | accuracy | precision | recall | **f1** | **roc_auc** | pr_auc |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `a19cc81f` | `6f28a3b` | basic (17) | logreg | 0.599 | 0.343 | 0.653 | **0.450** | **0.665** | 0.420 |
| 2 | `41b18e98` | `6f28a3b` | extended (20) | logreg | 0.717 | 0.458 | 0.667 | **0.543** | **0.766** | 0.634 |
| 3 | `cb56b5bc` | `6f28a3b` | with_weather (28) | logreg | 0.771 | 0.534 | 0.700 | **0.606** | **0.825** | 0.707 |
| 4 | `1b00226a` | `45834a0` | with_weather (28) | xgboost (default) | 0.835 | 0.825 | 0.438 | **0.572** | **0.839** | 0.731 |
| 5 | `a7a0ecd5` | `341e954` | with_weather (28) | xgboost (tuned) | 0.790 | 0.569 | 0.677 | **0.619** | **0.830** | 0.720 |
| 6 | `4227cd7b` | `b3dc9c8` | with_weather (28) | xgboost (optuna) | 0.795 | 0.576 | 0.696 | **0.630** | **0.839** | 0.731 |
| 7 | `d60d3feb` | `c419a05` | with_weather (28) | lightgbm (default) | 0.836 | 0.825 | 0.439 | **0.573** | **0.839** | 0.731 |
| 8 | `716664c0` | `3ce36c8` | with_weather (28) | lightgbm (optuna) | 0.783 | 0.552 | 0.731 | **0.629** | **0.839** | 0.732 |

(*В feature_set указано число признаков до one-hot.*)

## Цепочка дельт — что поменяли → что получили

### Δ1.  Run #1 → Run #2 — добавили каскадные задержки и загруженность

**Что изменено в `params.yaml`:** `features.active_set: basic → extended`.
**Что добавлено в датасет:** колонки `inbound_delay_minutes`,
`origin_congestion_index`, `destination_congestion_index`. Модель,
гиперпараметры и сид — без изменений.

| метрика | Run #1 | Run #2 | Δ | Δ% |
|---|---|---|---|---|
| f1 | 0.450 | 0.543 | +0.093 | **+20.6 %** |
| roc_auc | 0.665 | 0.766 | +0.101 | +15.2 % |
| pr_auc | 0.420 | 0.634 | +0.214 | **+51.0 %** |

**Вывод для отчёта:** самый большой одноразовый прирост во всей серии
дала ровно одна правка в данных — добавление трёх колонок про
каскадную задержку и индексы загруженности аэропортов. Без них модель
видит только статическое расписание и фундаментально ограничена.

### Δ2.  Run #2 → Run #3 — добавили погодные фичи

**Что изменено в `params.yaml`:** `features.active_set: extended → with_weather`.
**Что добавлено:** температура, осадки, видимость, ветер и категория
суровости погоды для пунктов вылета и прилёта (8 числовых + 2
категориальных колонки).

| метрика | Run #2 | Run #3 | Δ | Δ% |
|---|---|---|---|---|
| f1 | 0.543 | 0.606 | +0.063 | +11.6 % |
| roc_auc | 0.766 | 0.825 | +0.060 | +7.8 % |
| pr_auc | 0.634 | 0.707 | +0.073 | +11.5 % |

**Вывод:** погода даёт второй заметный прирост, но уже в полтора-два
раза слабее, чем каскадные/congestion-фичи. Это согласуется с базовой
интуицией: погода объясняет часть задержек, но не их большинство.

### Δ3.  Run #3 → Run #4 — поменяли модель: logreg → xgboost (defaults)

**Что изменено:** `train.active_model: logreg → xgboost`. Признаки,
seed и таргет — те же. Гиперпараметры XGBoost — `params.yaml` defaults
(`n_estimators=400`, `max_depth=6`, без балансировки классов).

| метрика | Run #3 | Run #4 | Δ | комментарий |
|---|---|---|---|---|
| accuracy | 0.771 | 0.835 | +0.064 | растёт |
| precision | 0.534 | **0.825** | +0.291 | резко растёт |
| recall | 0.700 | **0.438** | **−0.263** | резко падает |
| f1 | 0.606 | 0.572 | −0.034 | слегка падает |
| roc_auc | 0.825 | 0.839 | +0.013 | растёт |
| pr_auc | 0.707 | 0.731 | +0.024 | растёт |

**Вывод:** XGBoost из коробки **лучше упорядочивает рейсы по риску**
(threshold-independent ROC-AUC и PR-AUC выросли), но при дефолтном
пороге 0.5 не борется с дисбалансом классов 76/24 — оптимизирует
точность ценой полноты, F1 даже немного падает. Полезный сюжет для
отчёта: смена модели не всегда улучшает все метрики и важно смотреть
на правильную метрику для задачи.

### Δ4.  Run #4 → Run #5 — поменяли гиперпараметры XGBoost

**Что изменено в `params.yaml` → `train.xgboost`:**
- `n_estimators`: 400 → **800**
- `max_depth`: 6 → **8**
- добавлено `scale_pos_weight: 3.17` (= (1−0.24)/0.24, нативный аналог
  `class_weight='balanced'` для XGBoost)

Признаки и сид — те же.

| метрика | Run #4 | Run #5 | Δ |
|---|---|---|---|
| accuracy | 0.835 | 0.790 | −0.045 |
| precision | 0.825 | 0.569 | −0.256 |
| **recall** | 0.438 | **0.677** | **+0.239** |
| **f1** | 0.572 | **0.619** | **+0.047** |
| roc_auc | 0.839 | 0.830 | −0.009 |
| pr_auc | 0.731 | 0.720 | −0.011 |

**Вывод:** `scale_pos_weight` вернул recall на уровень logreg, F1 стал
лучшим во всей серии (0.619). ROC-AUC и PR-AUC чуть просели, потому
что мы не «улучшаем» ранжирование — мы пересдвигаем рабочую точку
ближе к recall-приоритету. Это и есть «дельта от гиперпараметров,
изолированная от данных».

### Δ5.  Run #5 → Run #6 — Optuna-tuning XGBoost (TPE, 30 trials)

**Что изменено:** запущен `python -m src.models.tune --n-trials 30`.
Optuna (TPE-сэмплер с тем же `seed=42`) перебрал семь гиперпараметров
XGBoost: `n_estimators ∈ [200..1200]`, `max_depth ∈ [4..10]`,
`learning_rate ∈ [0.01..0.2]` (log-scale), `subsample ∈ [0.6..1.0]`,
`colsample_bytree ∈ [0.6..1.0]`, `min_child_weight ∈ [1..20]`,
`scale_pos_weight ∈ [1.0..5.0]`. Целевая функция — F1 на val.

**Найденная конфигурация (`models/best_xgboost_params.yaml`):**

```yaml
n_estimators: 1000
max_depth: 7
learning_rate: 0.0138        # ↓ от 0.05 ручного — компенсируется бо́льшим n_estimators
subsample: 0.739             # ↓ от 0.9 — сильнее регуляризация
colsample_bytree: 0.813
min_child_weight: 9          # новый, не настраивался вручную
scale_pos_weight: 2.371      # ↓ от 3.17 — менее агрессивный сдвиг рабочей точки
```

| метрика | Run #5 (manual) | Run #6 (optuna) | Δ |
|---|---|---|---|
| accuracy | 0.790 | 0.795 | +0.005 |
| precision | 0.569 | 0.576 | +0.007 |
| recall | 0.677 | 0.696 | +0.019 |
| f1 | 0.619 | **0.630** | +0.011 (+1.8 %) |
| roc_auc | 0.830 | **0.839** | +0.009 |
| pr_auc | 0.720 | 0.731 | +0.011 |

**Вывод:** Optuna нашла лучший конфиг по всем шести метрикам
одновременно — это редкий случай (обычно тюнинг сдвигает trade-off
между precision/recall). Ключевое наблюдение: алгоритм сам пришёл к
**менее агрессивному** `scale_pos_weight` (2.37 вместо моего ручного
3.17), скомпенсировав это меньшим `learning_rate` и более сильной
регуляризацией через `subsample`. Прирост скромный (+1.8 % F1),
потому что ручной конфиг Run #5 уже был близок к оптимуму поиска —
это нормальный исход и хороший сюжет: **MLOps-инструменты выжимают
последние проценты, но основную работу делают данные** (см. Δ1).

### Δ6.  Run #6 → Run #7 — поменяли boosting library: xgboost → lightgbm (defaults)

**Что изменено в `params.yaml`:** `train.active_model: xgboost → lightgbm`.
Признаки (`with_weather`, 28), сид (42), таргет (`delay_binary`) — те же.
LightGBM-параметры — `params.yaml` defaults (`n_estimators=400`,
`num_leaves=63`, `learning_rate=0.05`, `subsample=0.9`,
`colsample_bytree=0.9`, **без** балансировки классов).

| метрика | Run #6 (xgb optuna) | Run #7 (lgbm default) | Δ |
|---|---|---|---|
| accuracy | 0.795 | 0.836 | +0.041 |
| precision | 0.576 | **0.825** | +0.249 |
| recall | 0.696 | **0.439** | **−0.257** |
| f1 | **0.630** | 0.573 | −0.057 (−9.0 %) |
| roc_auc | 0.839 | 0.839 | 0.000 |
| pr_auc | 0.731 | 0.731 | 0.000 |

**Контрольное сравнение Run #4 (xgb default) vs Run #7 (lgbm default)** —
обе модели на тех же признаках без балансировки:

| метрика | Run #4 (xgb default) | Run #7 (lgbm default) | Δ |
|---|---|---|---|
| accuracy | 0.835 | 0.836 | +0.001 |
| precision | 0.825 | 0.825 | 0.000 |
| recall | 0.438 | 0.439 | +0.001 |
| f1 | 0.572 | 0.573 | +0.001 |
| roc_auc | 0.839 | 0.839 | 0.000 |
| pr_auc | 0.731 | 0.731 | 0.000 |

**Вывод:** на этих данных XGBoost и LightGBM с дефолтными
гиперпараметрами и **без** балансировки классов дают практически
идентичные результаты (разница в F1 — 0.001, ROC-AUC и PR-AUC
совпадают до третьего знака). Это **главный сюжет дельты**: смена
boosting-библиотеки сама по себе ≠ улучшение, потому что обе
библиотеки оптимизируют одно и то же логистическое правдоподобие
на одних и тех же фичах и при том же дисбалансе 76/24 «вырождаются»
в одинаковую precision-приоритетную точку. Прирост Run #5/#6 над
Run #4 (+0.047 → +0.058 F1) пришёл **не от XGBoost**, а от
`scale_pos_weight=3.17` и подбора параметров — поэтому честный
следующий шаг для LightGBM это `class_weight='balanced'` (или
`is_unbalance=True`) + Optuna, что и запланировано как Run #8.

Полезное методологическое наблюдение для главы 3: **«поменяли
модель → метрики не сдвинулись» — это не отрицательный результат,
а доказательство, что узкое место не в модели, а в обработке
дисбаланса**. Соответствует тезису руководителя: «если перебор
параметров ничего не даёт — проблема в датасете».

### Δ7.  Run #7 → Run #8 — добавили балансировку и Optuna для LightGBM

**Что изменено:** обобщён `tune.py` под `--model {xgboost, lightgbm}`
(commit `3ce36c8`), запущен `python -m src.models.tune --model lightgbm
--n-trials 30`. Optuna (TPE, тот же `seed=42`) перебрала восемь
гиперпараметров LightGBM в диапазонах, параллельных XGBoost-серии:
`n_estimators ∈ [200..1200]`, `num_leaves ∈ [15..255]` (главный
регулятор сложности дерева в LightGBM, заменяет `max_depth`),
`learning_rate ∈ [0.01..0.2]` (log), `subsample ∈ [0.6..1.0]`
(с `subsample_freq=1` — иначе LightGBM игнорирует bagging),
`colsample_bytree ∈ [0.6..1.0]`, `min_child_samples ∈ [5..100]`
(аналог `min_child_weight`), `scale_pos_weight ∈ [1.0..5.0]`.

**Найденная конфигурация (`models/best_lightgbm_params.yaml`):**

```yaml
n_estimators: 700
num_leaves: 105
learning_rate: 0.0141
subsample: 0.9659           # subsample_freq=1 проставляется отдельно
colsample_bytree: 0.7480
min_child_samples: 27
scale_pos_weight: 2.665     # < ручного 3.17, как и у XGBoost
```

| метрика | Run #7 (lgbm default) | Run #8 (lgbm optuna) | Δ |
|---|---|---|---|
| accuracy | 0.836 | 0.783 | −0.053 |
| precision | 0.825 | 0.552 | −0.273 |
| **recall** | 0.439 | **0.731** | **+0.292** |
| **f1** | 0.573 | **0.629** | **+0.056 (+9.8 %)** |
| roc_auc | 0.839 | 0.839 | 0.000 |
| pr_auc | 0.731 | 0.732 | +0.001 |

**Та же дельта, что Run #4 → Run #6 для XGBoost** (там было +0.058 F1).
Сюжет идентичен: добавление `scale_pos_weight` сдвигает рабочую
точку с precision-приоритета на recall, ROC-AUC не меняется (ранжирование
то же), F1 растёт.

### Δ8.  Финальная сверка — Run #6 (xgb optuna) vs Run #8 (lgbm optuna)

Главный head-to-head серии: обе библиотеки **с одинаковой балансировкой
и одинаковой методикой тюнинга** на тех же признаках.

| метрика | Run #6 (xgb) | Run #8 (lgbm) | Δ (lgbm − xgb) |
|---|---|---|---|
| accuracy | 0.795 | 0.783 | −0.012 |
| precision | 0.576 | 0.552 | −0.024 |
| recall | 0.696 | 0.731 | +0.035 |
| **f1** | **0.630** | 0.629 | **−0.001** |
| **roc_auc** | **0.839** | **0.839** | 0.000 |
| pr_auc | 0.731 | 0.732 | +0.001 |

**Параметрическая параллель**, что особенно красиво для отчёта:
TPE-сэмплер на двух разных библиотеках независимо сошёлся к
**похожему режиму регуляризации**:

|  | XGBoost (Run #6) | LightGBM (Run #8) |
|---|---|---|
| learning_rate | 0.0138 | 0.0141 |
| n_estimators  | 1000 | 700 |
| scale_pos_weight | 2.371 | 2.665 |
| sample_per_tree (subsample) | 0.739 | 0.966 |
| col fraction (colsample_bytree) | 0.813 | 0.748 |
| leaf-floor (min_child_*) | 9 | 27 |

В обоих случаях Optuna выбирает **низкий learning rate + умеренный
`scale_pos_weight` ~ 2.5** (мягче моего ручного 3.17). Это сильное
независимое подтверждение того, что 2.5 — реальная оптимальная точка
для этого датасета, а не артефакт конкретной библиотеки.

**Вывод для главы 3:** связка "бустинг + балансировка классов
+ Optuna" даёт plateau ≈ F1 0.63 / ROC-AUC 0.839 на этих признаках,
независимо от XGBoost vs LightGBM. Дальнейший прирост ждать **не
от смены модели в этом классе**, а от: (a) ещё одной серии фичей
(сезонность, праздники, исторические задержки маршрута), (b) другой
архитектуры — ансамбль logreg + xgb + lgbm (Run #10), (c) другой
постановки задачи — переход на регрессию минут вместо бинарного.

## Снимок прогресса по F1 и ROC-AUC

```
Run #  feature_set    model              f1     roc_auc
  1    basic          logreg            0.450    0.665      ┐
  2    extended       logreg            0.543    0.766      │  data axis
  3    with_weather   logreg            0.606    0.825      ┘
  4    with_weather   xgboost           0.572    0.839      ┐  model axis (defaults)
  7    with_weather   lightgbm          0.573    0.839      ┘  ≈ копия Run #4 — boosting
                                                              library без балансировки
                                                              классов ничего не меняет
  5    with_weather   xgboost (tuned)   0.619    0.830      ─  hyperparam axis (manual)
  6    with_weather   xgboost (optuna)  0.630    0.839      ┐  hyperparam axis (auto) —
  8    with_weather   lightgbm (optuna) 0.629    0.839      ┘  библиотеки сходятся к
                                                              одному plateau при
                                                              одинаковой методике тюнинга
```

## Что воспроизводимо и как

Каждая строка таблицы соответствует **одному git-коммиту**, который
поменял ровно один аспект `params.yaml`:

| Run | git commit (`git show <sha>`) |
|---|---|
| #1 (baseline) | `6f28a3b feat(training): baseline LogReg + MLflow tracking` |
| #2, #3 | `129d431 experiment: data axis — feature progression on logreg` |
| #4 | `45834a0 experiment: switch active model logreg → xgboost on with_weather features` |
| #5 | `341e954 experiment: tune xgboost — depth 6→8, n_est 400→800, scale_pos_weight 3.17` |
| #6 | `b3dc9c8 feat(tuning): add Optuna XGBoost tuner with single-run MLflow logging` |
| #7 | `c419a05 experiment: switch active model xgboost → lightgbm on with_weather` |
| #8 | `3ce36c8 feat(tuning): generalize Optuna tuner to support LightGBM` (study run on this HEAD) |

Чтобы воспроизвести любой run:

```bash
git checkout <sha>
dvc pull            # подтянуть данные на нужной версии
python -m src.models.train
```

В MLflow runs затегированы: `git_commit`, `dvc_data_hash` (12 символов
md5 raw-датасета), `feature_set`, `model`, `task`, `params_version`.
Эти теги — основной фильтр для сравнений в MLflow UI.

## Что дальше — задел для следующих итераций

Серия `delay_binary` достигла plateau ≈ F1 0.63 / ROC-AUC 0.839 на
текущем наборе фичей; следующие приросты можно ждать только от смены
оси (новая ось — данные/ансамбль/постановка), не от ещё одного бустинга.

- **Run #9:** CatBoost на тех же фичах (с `auto_class_weights=Balanced`)
  + Optuna через расширение `tune.py`. Маловероятно перебьёт plateau,
  но даёт третью точку библиотечной оси (xgb / lgbm / cb) — закрывает
  «pool бустингов» в отчёте.
- **Run #10:** ансамбль (stacking/voting) победителей logreg + xgboost
  (Run #6) + lightgbm (Run #8) через новый `src/models/ensemble.py`.
  Обычно даёт ещё +1–2 пункта F1 на бустинговом плато; в отчёте
  закрывает архитектурную ось.
- **Run #11+ — следующая ось данных:** добавить календарные/сезонные
  фичи (праздники РФ, школьные каникулы, день недели × месяц
  взаимодействия), исторический average delay по маршруту/борту,
  индикатор пиковых часов. Это новая дельта по data-оси, аналог
  Δ1 (extended) и Δ2 (with_weather).
- **Серия по второй задаче (`delay_cause`, multi-class):** пройти
  ровно ту же data → model → tuning цепочку, но на причине задержки.
  Требует доработки `train.py` (сейчас падает на task != "delay_binary")
  и `evaluate.py` (нужна `multiclass_classification_metrics`).
  → **запущена** ниже как Cause series (C1-C4).

---

# Cause Series — голова B (`task = delay_cause`)

Параллельная серия для второй заявленной в теме ВКР задачи —
**мультиклассовая классификация причины задержки**. Семь классов:
`none`, `weather`, `carrier_operational`, `reactionary`,
`airport_congestion`, `cancelled`, `security`. Дисбаланс train:
**75.5 % none / 8.4 % weather / 7.7 % carrier / 3.7 % reactionary /
3.7 % congestion / 0.6 % cancelled / 0.3 % security.**

Валидация — тот же time-based slice Jan–Jun 2025 (n=36 310, **в т.ч.
отменённые**, в отличие от binary-серии — для cause-головы класс
`cancelled` несёт сигнал и должен оставаться). Random seed 42 везде.
**Headline-метрика — `macro_f1`** (равный вес каждого класса, не даёт
доминирующему `none` маскировать провалы по причинам).

## Сводная таблица — Cause series

| # | mlflow run_id | git_commit | feature_set | model | accuracy | **macro_f1** | weighted_f1 | macro_prec | macro_rec | roc_auc_ovr |
|---|---|---|---|---|---|---|---|---|---|---|
| C1 | `faae4868` | `9ab4f8a` | basic (17) | logreg + class_weight | 0.194 | **0.137** | 0.239 | 0.206 | 0.314 | 0.670 |
| C2 | `fd12e644` | `0cbbf8a` | extended (20) | logreg + class_weight | 0.331 | **0.259** | 0.404 | 0.280 | 0.430 | 0.761 |
| C3 | `6b948e3a` | `0cbbf8a` | with_weather (30) | logreg + class_weight | 0.396 | **0.310** | 0.473 | 0.314 | 0.488 | 0.827 |
| C4 | `5520b46a` | `4ad72c5` | with_weather (30) | xgboost + balanced sw | **0.655** | **0.359** | **0.681** | 0.338 | 0.452 | 0.830 |

## Дельты — что меняли → что получили

### Δ-C1.  C1 → C2 — добавили каскадные/congestion-фичи

`features.active_set: basic → extended`. Модель и сид — те же.

| метрика | C1 | C2 | Δ |
|---|---|---|---|
| macro_f1 | 0.137 | **0.259** | **+0.122 (+89 %)** |
| accuracy | 0.194 | 0.331 | +0.137 |
| weighted_f1 | 0.239 | 0.404 | +0.165 |
| roc_auc_ovr | 0.670 | 0.761 | +0.091 |

**Вывод:** в cause-серии Δ1 ещё крупнее, чем в binary (89 % vs 21 % по
headline-метрике). Логично: `inbound_delay_minutes` и congestion-индексы
напрямую разделяют `reactionary` от `carrier_operational` и
`airport_congestion` — то, что logreg на голых фичах расписания
вообще не видит.

### Δ-C2.  C2 → C3 — добавили погодные фичи

`features.active_set: extended → with_weather`.

| метрика | C2 | C3 | Δ |
|---|---|---|---|
| macro_f1 | 0.259 | **0.310** | **+0.051 (+19.7 %)** |
| accuracy | 0.331 | 0.396 | +0.065 |
| weighted_f1 | 0.404 | 0.473 | +0.069 |
| roc_auc_ovr | 0.761 | 0.827 | +0.066 |

**Вывод:** погода даёт второй шаг на cause-голове ровно так же, как и
на binary — даже немного слабее. Сюжет: класс `weather` теперь стал
хоть как-то отделимым от `none`, что и дало основной вклад в дельту.

### Δ-C3.  C3 → C4 — поменяли модель: logreg → xgboost (defaults + balanced sample weights)

`train.active_model: logreg → xgboost`. Признаки и сид — те же.
`compute_sample_weight('balanced')` подаётся через
`fit_params['estimator__sample_weight']` — это multiclass-аналог
`scale_pos_weight` (последний XGBoost игнорирует на multiclass).

| метрика | C3 (logreg) | C4 (xgboost) | Δ |
|---|---|---|---|
| accuracy | 0.396 | **0.655** | **+0.259 (+65 %)** |
| weighted_f1 | 0.473 | **0.681** | **+0.208 (+44 %)** |
| macro_f1 | 0.310 | **0.359** | +0.049 (+16 %) |
| macro_precision | 0.314 | 0.338 | +0.024 |
| macro_recall | 0.488 | 0.452 | −0.036 |
| roc_auc_ovr | 0.827 | 0.830 | +0.003 (паритет) |

**Главное наблюдение:** ROC-AUC одинаковый — обе модели **одинаково
хорошо ранжируют** причины. Прирост xgboost живёт в
**decision-boundary calibration** на доминирующем классе `none`,
поэтому accuracy и weighted_f1 растут резко (+65 % / +44 %), а
macro_f1 — гораздо умереннее (+16 %). macro_recall даже слегка падает
из-за того, что модель чуть охотнее предсказывает `none`.

**Сюжет для главы 3:** на multiclass-задаче с длинным хвостом
(`security` 0.3 %, `cancelled` 0.6 %) macro-метрики живут отдельно от
weighted-метрик. Headroom для cause-головы — **именно в хвосте**:
ансамбли one-vs-rest, SMOTE по редким классам, или явная
двухступенчатая модель «есть ли причина → какая именно». Дальнейший
тюнинг xgboost (Optuna, аналог Run #6/#8) даст ≤+5 пунктов macro_f1
без вмешательства в данные — это меньшая дельта, чем смена данных
(+19 % от Δ-C2) и сильно меньшая, чем смена данных в начале серии
(+89 % от Δ-C1).

## Что дальше для cause series

- **C5 — Optuna для xgboost на cause:** аналог Run #6, должен дать
  ещё +3-5 пунктов macro_f1. `tune.py` сейчас бинарный → доработать
  под `--task delay_cause`.
- **C6 — двухступенчатая постановка:** сначала бинарный «causal vs
  none» (на всём датасете), потом мультиклассовый «какая причина»
  (только на causal). Логически аналогична human-baseline и часто
  выигрывает 5-10 пунктов macro_f1 на задачах с доминирующим
  «нулевым» классом.
- **Финальный артефакт — scored test dataset (см. memory):** после
  выбора лидеров binary и cause голов — единый parquet с колонками
  `predicted_delay`, `delay_proba`, `predicted_cause`, `cause_proba`
  на test split.
