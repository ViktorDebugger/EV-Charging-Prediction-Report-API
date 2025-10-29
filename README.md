# EV Charging Prediction API

REST API для прогнозування типу зарядної станції електромобілів на основі даних про зарядну сесію.

## Опис проекту

Цей проект використовує машинне навчання (SVC - Support Vector Classifier) для класифікації типу зарядної станції на основі параметрів зарядної сесії електромобіля. API побудований на FastAPI та SQLAlchemy.

## Встановлення

### 1. Клонування репозиторію

```bash
git clone <repository-url>
cd project
```

### 2. Встановлення залежностей

```bash
pip install -r requirements.txt
```

Основні залежності:

- FastAPI
- Uvicorn
- SQLAlchemy
- Pandas
- NumPy
- Scikit-learn

### 3. Структура проекту

```
project/
├── app/
│   ├── main.py                    # Головний файл додатку
│   ├── database/
│   │   ├── database.py            # Конфігурація БД
│   │   ├── database.db            # SQLite база даних
│   │   ├── models/                # SQLAlchemy моделі
│   │   │   ├── __init__.py        # Спільний Base
│   │   │   ├── inference_input.py # Модель вхідних даних
│   │   │   ├── processed_feature.py # Модель оброблених фіч
│   │   │   └── prediction.py      # Модель передбачень
│   │   └── crud/                  # CRUD операції
│   │       ├── inference_input.py
│   │       ├── processed_feature.py
│   │       └── prediction.py
│   ├── schemas/                   # Pydantic схеми
│   │   ├── inference_input.py
│   │   ├── processed_feature.py
│   │   └── prediction.py
│   ├── endpoints/                 # API endpoints
│   │   ├── training.py            # Endpoint тренування
│   │   ├── inference.py           # Endpoint передбачення
│   │   └── monitor.py             # Endpoints моніторингу
│   └── services/
│       ├── model.py               # SVC модель
│       └── utils/                 # Допоміжні функції
│           ├── inference_input.py # Обробка вхідних даних
│           ├── processed_feature.py # Обробка фіч
│           ├── monitor.py         # Утиліти моніторингу
│           └── logger.py          # Логування
├── models/
│   └── svc_model.pkl              # Збережена модель
├── logs/
│   ├── training.log               # Логи тренування
│   └── inference.log              # Логи передбачень
├── reports/                       # HTML звіти моніторингу
│   ├── data-drift/                # Data Drift Reports
│   ├── target-drift/              # Target Drift Reports
│   ├── classification-performance/ # Classification Performance Reports
│   └── data-quality/              # Data Quality Reports
└── README.md
```

## Запуск сервера

### Локальний запуск

```bash
python -m uvicorn app.main:app --reload
```

Сервер буде доступний за адресою: `http://127.0.0.1:8000`

### Документація API

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

## API Endpoints

### 1. Тренування моделі

**POST** `/train_model`

Тренує SVC модель на даних з таблиці `Processed-Features`.

#### Приклад запиту:

```bash
curl -X POST "http://127.0.0.1:8000/train_model"
```

#### Приклад відповіді:

```json
{
  "message": "Model trained successfully",
  "accuracy": 0.9523,
  "precision": 0.9521,
  "recall": 0.9523,
  "f1_score": 0.952,
  "train_size": 9000,
  "test_size": 1000,
  "classes": ["DC_Fast_Charger", "Level_1", "Level_2"],
  "confusion_matrix": [
    [320, 5, 8],
    [3, 315, 4],
    [6, 2, 337]
  ],
  "classification_report": {
    "DC_Fast_Charger": {
      "precision": 0.9726,
      "recall": 0.9609,
      "f1-score": 0.9667
    },
    "Level_1": {
      "precision": 0.9783,
      "recall": 0.9782,
      "f1-score": 0.9783
    },
    "Level_2": {
      "precision": 0.9656,
      "recall": 0.9799,
      "f1-score": 0.9727
    }
  }
}
```

### 2. Передбачення

**POST** `/predict`

Виконує передбачення типу зарядної станції на основі вхідних даних.

#### Приклад запиту:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "User_1",
    "vehicle_model": "BMW i3",
    "battery_capacity_kwh": 108.463007412840720,
    "vehicle_age_years": 2,
    "charging_station_id": "Station_391",
    "charging_station_location": "Houston",
    "charging_start_time": "2024-01-01 00:00:00",
    "charging_end_time": "2024-01-01 00:39:00",
    "energy_consumed_kwh": 60.712345734927770,
    "charging_duration_h": 0.591363425358500,
    "charging_rate_kw": 36.289180566988140,
    "charging_cost_usd": 13.087716791774450,
    "time_of_day": "Evening",
    "day_of_week": "Tuesday",
    "state_of_charge_start": 29.371579797140050,
    "state_of_charge_end": 86.119962444578390,
    "distance_driven_since_last_charge_km": 293.602110638327930,
    "temperature_c": 27.947593055800100,
    "user_type": "Commuter"
  }'
```

#### Приклад відповіді (успіх):

```json
{
  "status": "success",
  "predicted_class": "DC_Fast_Charger",
  "confidence": 0.8521,
  "probabilities": {
    "DC_Fast_Charger": 0.8521,
    "Level_1": 0.0789,
    "Level_2": 0.069
  },
  "processing_time": 0.0234,
  "input_summary": {
    "user_id": "User_1",
    "vehicle_model": "BMW i3",
    "charging_station_location": "Houston",
    "user_type": "Commuter"
  },
  "saved_prediction_id": 1
}
```

#### Приклад відповіді (помилка валідації):

```json
{
  "status": "error",
  "message": "Not all required fields are filled in",
  "missing_fields": ["charging_start_time", "charging_end_time"]
}
```

#### Приклад відповіді (невалідні категорії):

```json
{
  "status": "error",
  "message": "Invalid categorical values",
  "invalid_fields": ["vehicle_model", "user_type"],
  "errors": [
    "'vehicle_model' має невалідне значення 'Toyota Prius'. Допустимі значення: Tesla Model 3, Hyundai Kona, Nissan Leaf, BMW i3, Chevy Bolt",
    "'user_type' має невалідне значення 'Tourist'. Допустимі значення: Commuter, Long-Distance Traveler"
  ]
}
```

### 3. Моніторинг моделі (Evidently AI)

API надає endpoints для генерації HTML-звітів моніторингу на основі бібліотеки Evidently AI.

#### 3.1. Data Drift Report

**GET** `/data-drift`

Генерує звіт про зміни розподілу вхідних даних (Data Drift).

**Приклад запиту:**

```bash
curl -X GET "http://127.0.0.1:8000/data-drift"
```

**Відповідь:**

```json
{
  "status": "success",
  "message": "Data Drift Report completed",
  "report_path": "reports/data-drift/report_2025-10-28_20-28-37.html"
}
```

**Що показує звіт:**

- Статистичні тести дрифту для кожної ознаки
- Візуалізації розподілів (reference vs current)
- P-values та drift scores
- Гістограми та boxplots для числових ознак

#### 3.2. Target Drift Report

**GET** `/target-drift`

Генерує звіт про зміни розподілу цільової змінної (типу зарядки).

**Приклад запиту:**

```bash
curl -X GET "http://127.0.0.1:8000/target-drift"
```

**Відповідь:**

```json
{
  "status": "success",
  "message": "Target Drift Report completed",
  "report_path": "reports/target-drift/report_2025-10-28_20-56-24.html"
}
```

**Що показує звіт:**

- Зміни пропорцій класів (Level_1, Level_2, DC_Fast_Charger)
- Статистичні тести на зміну розподілу target
- Візуалізація розподілу класів до/після

#### 3.4. Data Quality Report

**GET** `/data-quality`

Генерує звіт про якість даних (пропуски, типи, унікальні значення).

**Приклад запиту:**

```bash
curl -X GET "http://127.0.0.1:8000/data-quality"
```

**Відповідь:**

```json
{
  "status": "success",
  "message": "Data Quality Report completed",
  "report_path": "reports/data-quality/report_2025-10-28_21-13-21.html"
}
```

**Що показує звіт:**

- Dataset Summary (кількість рядків, колонок, типи даних)
- Missing Values (пропущені значення)
- Статистики для числових ознак (mean, std, min, max)
- Розподіл категоріальних ознак
- Викиди (outliers)
- Кореляції між ознаками

#### 3.5. Структура моніторингу

**Reference Data (еталонні дані):**

- Витягуються з таблиці `Processed-Features`
- Дані, на яких модель тренувалася
- Використовуються як базова лінія для порівняння

**Current Data (поточні дані):**

- Витягуються з таблиці `Inference-Inputs`
- Запити до API `/predict`
- Порівнюються з reference для виявлення дрифту

**Збереження звітів:**
Всі звіти зберігаються у форматі HTML у директорії `reports/` з timestamp:

```
reports/
├── data-drift/report_2025-10-28_20-28-37.html
├── target-drift/report_2025-10-28_20-56-24.html
├── classification-performance/report_2025-10-28_21-00-15.html
└── data-quality/report_2025-10-28_21-13-21.html
```

## Структура бази даних

Проект використовує SQLite базу даних з трьома основними таблицями:

### Таблиця `Inference-Inputs`

Зберігає вхідні дані для передбачень (inference).

| Поле                                 | Тип      | Опис                                       |
| ------------------------------------ | -------- | ------------------------------------------ |
| id                                   | INTEGER  | Первинний ключ                             |
| timestamp                            | DATETIME | Час створення запису                       |
| user_id                              | STRING   | ID користувача                             |
| vehicle_model                        | STRING   | Модель електромобіля                       |
| battery_capacity_kwh                 | FLOAT    | Ємність батареї (кВт·год)                  |
| vehicle_age_years                    | FLOAT    | Вік авто (роки)                            |
| charging_station_id                  | STRING   | ID зарядної станції                        |
| charging_station_location            | STRING   | Місцезнаходження станції                   |
| charging_start_time                  | STRING   | Час початку зарядки                        |
| charging_end_time                    | STRING   | Час завершення зарядки                     |
| energy_consumed_kwh                  | FLOAT    | Спожита енергія (кВт·год)                  |
| charging_duration_h                  | FLOAT    | Тривалість зарядки (години)                |
| charging_rate_kw                     | FLOAT    | Швидкість зарядки (кВт)                    |
| charging_cost_usd                    | FLOAT    | Вартість зарядки ($)                       |
| time_of_day                          | STRING   | Час доби (Morning/Afternoon/Evening/Night) |
| day_of_week                          | STRING   | День тижня                                 |
| state_of_charge_start                | FLOAT    | Початковий заряд (%)                       |
| state_of_charge_end                  | FLOAT    | Кінцевий заряд (%)                         |
| distance_driven_since_last_charge_km | FLOAT    | Пробіг з останньої зарядки (км)            |
| temperature_c                        | FLOAT    | Температура (°C)                           |
| user_type                            | STRING   | Тип користувача                            |

### Таблиця `Processed-Features`

Зберігає оброблені та закодовані фічі для тренування моделі.

| Поле                                 | Тип      | Опис                                  |
| ------------------------------------ | -------- | ------------------------------------- |
| id                                   | INTEGER  | Первинний ключ                        |
| timestamp                            | DATETIME | Час створення запису                  |
| vehicle_model                        | FLOAT    | Закодована модель авто                |
| battery_capacity_kwh                 | FLOAT    | Ємність батареї                       |
| vehicle_age_years                    | FLOAT    | Вік авто                              |
| charging_station_id                  | FLOAT    | Закодований ID станції                |
| charging_station_location            | FLOAT    | Закодоване місцезнаходження           |
| charging_start_time                  | STRING   | Час початку                           |
| charging_end_time                    | STRING   | Час завершення                        |
| charging_duration_h                  | FLOAT    | Тривалість зарядки                    |
| charging_rate_kw                     | FLOAT    | Швидкість зарядки                     |
| energy_consumed_kwh                  | FLOAT    | Спожита енергія (стандартизована)     |
| charging_cost_usd                    | FLOAT    | Вартість зарядки                      |
| time_of_day                          | INTEGER  | Закодований час доби (0-3)            |
| day_of_week                          | INTEGER  | Закодований день тижня (0-6)          |
| distance_driven_since_last_charge_km | FLOAT    | Пробіг                                |
| temperature_c                        | FLOAT    | Температура                           |
| state_of_charge_start                | FLOAT    | Початковий заряд                      |
| state_of_charge_end                  | FLOAT    | Кінцевий заряд                        |
| charger_type_Level_1                 | INTEGER  | One-hot: Level 1 (0/1)                |
| charger_type_Level_2                 | INTEGER  | One-hot: Level 2 (0/1)                |
| user_type_Commuter                   | INTEGER  | One-hot: Commuter (0/1)               |
| user_type_Long_Distance_Traveler     | INTEGER  | One-hot: Long-Distance Traveler (0/1) |

### Таблиця `Predictions`

Зберігає результати передбачень моделі.

| Поле                  | Тип     | Опис                                        |
| --------------------- | ------- | ------------------------------------------- |
| id                    | INTEGER | Первинний ключ                              |
| inference_input_id    | INTEGER | FK → Inference-Inputs.id (для inference)    |
| processed_features_id | INTEGER | FK → Processed-Features.id (для тренування) |
| actual_class          | STRING  | Фактичний клас (для тренування)             |
| predicted_class       | STRING  | Передбачений клас                           |
| confidence            | FLOAT   | Впевненість передбачення (0-1)              |
| source                | STRING  | Джерело: "train" або "inference"            |

### Діаграма зв'язків

```
┌─────────────────────┐
│ Inference-Inputs    │
│ (вхідні дані)       │
└──────────┬──────────┘
           │ 1
           │
           │ N
┌──────────▼──────────┐
│ Predictions         │◄─────┐
│ (передбачення)      │      │
└──────────┬──────────┘      │
           │ N                │ N
           │                  │
           │ 1                │ 1
┌──────────▼──────────────────▼───┐
│ Processed-Features              │
│ (оброблені фічі для тренування) │
└─────────────────────────────────┘
```

## Приклади збережених передбачень

### Передбачення з inference (source="inference")

```json
{
  "id": 1,
  "inference_input_id": 15,
  "processed_features_id": null,
  "actual_class": null,
  "predicted_class": "DC_Fast_Charger",
  "confidence": 0.8521,
  "source": "inference"
}
```

### Передбачення з тренування (source="train")

```json
{
  "id": 2,
  "inference_input_id": null,
  "processed_features_id": 142,
  "actual_class": "Level_2",
  "predicted_class": "Level_2",
  "confidence": 0.9234,
  "source": "train"
}
```

## Валідація вхідних даних

### Обов'язкові поля

Всі поля у запиті `/predict` є обов'язковими (крім `id` та `timestamp`).

### Допустимі категоріальні значення

- **vehicle_model**: Tesla Model 3, Hyundai Kona, Nissan Leaf, BMW i3, Chevy Bolt
- **charging_station_location**: Los Angeles, San Francisco, Houston, New York, Chicago
- **time_of_day**: Morning, Afternoon, Evening, Night
- **day_of_week**: Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday
- **user_type**: Commuter, Long-Distance Traveler

## Логування

Система логує всі операції:

### Training logs (`logs/training.log`)

```json
{
  "timestamp": "2024-01-15T10:30:45.123456",
  "type": "training",
  "status": "success",
  "accuracy": 0.9523,
  "train_size": 9000,
  "test_size": 1000,
  "classes": ["DC_Fast_Charger", "Level_1", "Level_2"],
  "precision": 0.9521,
  "recall": 0.9523,
  "f1_score": 0.952
}
```

### Inference logs (`logs/inference.log`)

```json
{
  "timestamp": "2024-01-15T10:35:12.789012",
  "type": "inference",
  "status": "success",
  "inference_input_id": 15,
  "input_data": {
    "user_id": "User_1",
    "vehicle_model": "BMW i3",
    "charging_station_location": "Houston",
    "user_type": "Commuter"
  },
  "prediction": {
    "predicted_class": "DC_Fast_Charger",
    "confidence": 0.8521,
    "probabilities": {
      "DC_Fast_Charger": 0.8521,
      "Level_1": 0.0789,
      "Level_2": 0.069
    }
  },
  "processing_time": 0.0234,
  "error_message": null
}
```

## Технології

- **FastAPI** - веб-фреймворк
- **SQLAlchemy** - ORM для роботи з БД
- **Scikit-learn** - машинне навчання (SVC)
- **Pandas/NumPy** - обробка даних
- **Pydantic** - валідація даних
- **Uvicorn** - ASGI сервер
- **Evidently AI** - моніторинг моделей ML та виявлення data drift

## Моніторинг та обслуговування

### Виявлення деградації моделі

Регулярно перевіряйте звіти моніторингу для виявлення:

1. **Data Drift** - зміна розподілу вхідних даних

   - Можливі причини: зміна поведінки користувачів, нові типи автомобілів, сезонність
   - Дія: розглянути перетренування моделі на нових даних

2. **Target Drift** - зміна розподілу класів

   - Можливі причини: зміна інфраструктури зарядних станцій, нові тарифи
   - Дія: перевірити чи модель адаптується до нового розподілу

3. **Performance Degradation** - зниження метрик якості
   - Можливі причини: data drift, зміна даних, помилки в pipeline
   - Дія: терміново перетренувати модель або відкотити до попередньої версії

### Рекомендації

- Генерувати звіти моніторингу **щоденно** або **щотижня**
- Зберігати historical звіти для аналізу трендів
- Встановити алерти на критичні зміни метрик
- Збирати ground truth (`actual_class`) для оцінки реальної продуктивності
- Планувати перетренування моделі при значному дрифті (drift score > 0.5)
