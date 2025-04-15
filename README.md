# Forest Fire Prediction System

Система предсказания лесных пожаров на основе спутниковых данных и машинного обучения.

## Описание

Этот проект представляет собой веб-сервис, который анализирует спутниковые данные для предсказания вероятности лесных пожаров. Система использует данные Sentinel-2 и машинное обучение (SVM и Random Forest) для анализа различных факторов, таких как температура, влажность, местоположение и расстояние от референсной точки.

## Функциональность

- Анализ спутниковых данных Sentinel-2
- Предсказание вероятности лесных пожаров
- API для получения предсказаний
- Возможность переобучения модели на новых данных
- Контейнеризация с помощью Docker

## Технологии

- Python 3.9
- Flask (веб-сервер)
- Gunicorn (WSGI-сервер)
- scikit-learn (машинное обучение)
- pandas (обработка данных)
- Docker (контейнеризация)
- Google Earth Engine (спутниковые данные)

## Установка и запуск

### Локальная установка

1. Клонируйте репозиторий:
```bash
git clone https://github.com/yourusername/forest-fire-prediction.git
cd forest-fire-prediction
```

2. Установите зависимости:
```bash
pip install -r requirements.txt
```

3. Запустите приложение:
```bash
python app.py
```

### Запуск с Docker

1. Соберите Docker-образ:
```bash
docker build -t forest-fire-prediction .
```

2. Запустите контейнер:
```bash
docker run -p 5000:5000 forest-fire-prediction
```

## Использование API

### Обучение модели

```bash
curl -X POST http://localhost:5000/train
```

### Получение предсказания

```bash
curl -X POST http://localhost:5000/predict \
-H "Content-Type: application/json" \
-d '{
    "Date": "2023-04-01",
    "Latitude": 50.42,
    "Longitude": 80.23,
    "Humidity": 0.5,
    "Temperature": 25.0,
    "Fire": 0,
    "DistanceFromReference": 0.0
}'
```

### Формат ответа

```json
{
    "probability": 0.75,
    "prediction": 1,
    "input_data": {
        "Date": "2023-04-01",
        "Latitude": 50.42,
        "Longitude": 80.23,
        "Humidity": 0.5,
        "Temperature": 25.0,
        "Fire": 0,
        "DistanceFromReference": 0.0
    }
}
```

## Структура проекта

- `app.py` - Веб-сервер на Flask
- `agent.py` - Агент машинного обучения для предсказания пожаров
- `script.py` - Скрипт для обработки спутниковых данных
- `requirements.txt` - Зависимости проекта
- `Dockerfile` - Конфигурация Docker
- `processed_data_2023.csv` - Обработанные данные для обучения модели

## Алгоритмы машинного обучения

Проект поддерживает два алгоритма машинного обучения:

1. **Support Vector Machine (SVM)**
   - Оптимизация гиперпараметров через GridSearchCV
   - Нормализация данных через StandardScaler
   - Сбалансированные веса классов

2. **Random Forest**
   - 100 деревьев решений
   - Случайное состояние для воспроизводимости

## Вклад в проект

Если вы хотите внести свой вклад в проект, пожалуйста, создайте ветку для ваших изменений и отправьте pull request.

## Лицензия

MIT 