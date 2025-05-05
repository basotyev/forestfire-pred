import os
import numpy as np
import pandas as pd
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
import pickle
import seaborn as sns

# Пути к данным
DATA_DIR = "satellite_images"
MODEL_PATH = "fire_detection_model.pkl"
RESULTS_DIR = "results"

# Создаем директорию для результатов, если её нет
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

def load_image(file_path):
    """Загружает изображение и преобразует в массив numpy"""
    try:
        img = cv2.imread(file_path)
        if img is None:
            print(f"Ошибка чтения изображения {file_path}")
            return None
        return img
    except Exception as e:
        print(f"Ошибка при загрузке изображения {file_path}: {e}")
        return None

def extract_features(image):
    """Извлекает признаки из изображения для классификации"""
    # Преобразуем в HSV цветовое пространство
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Средние значения по каналам (BGR и HSV)
    bgr_means = np.mean(image, axis=(0,1))
    hsv_means = np.mean(hsv, axis=(0,1))

    # Стандартные отклонения по каналам
    bgr_std = np.std(image, axis=(0,1))
    hsv_std = np.std(hsv, axis=(0,1))

    # Доминирующие цвета (используем K-means)
    pixels = image.reshape(-1, 3)
    pixels = np.float32(pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 5  # количество кластеров
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    dominant_colors = centers

    # Гистограммы
    hist_b = cv2.calcHist([image], [0], None, [32], [0, 256])
    hist_g = cv2.calcHist([image], [1], None, [32], [0, 256])
    hist_r = cv2.calcHist([image], [2], None, [32], [0, 256])
    hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])

    # Процент красных пикселей (потенциальное пламя)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 + mask2

    red_pixels = np.sum(red_mask > 0)
    total_pixels = image.shape[0] * image.shape[1]
    red_percentage = red_pixels / total_pixels

    # Сегментация по яркости для обнаружения потенциальных пожаров
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    bright_percentage = np.sum(bright_mask > 0) / total_pixels

    # Объединяем все признаки в один вектор
    features = np.concatenate([
        bgr_means, bgr_std,                         # 6 признаков
        hsv_means, hsv_std,                         # 6 признаков
        dominant_colors.flatten(),                  # 15 признаков (5 кластеров x 3 канала)
        hist_b.flatten() / np.sum(hist_b),          # 32 признака (нормализованная гистограмма)
        hist_g.flatten() / np.sum(hist_g),          # 32 признака
        hist_r.flatten() / np.sum(hist_r),          # 32 признака
        hist_h.flatten() / np.sum(hist_h),          # 32 признака
        hist_s.flatten() / np.sum(hist_s),          # 32 признака
        hist_v.flatten() / np.sum(hist_v),          # 32 признака
        [red_percentage, bright_percentage]         # 2 признака
    ])

    return features, red_percentage, bright_percentage

def prepare_dataset():
    """Готовит датасет для обучения модели"""
    # Ищем все типы изображений
    all_images = []

    for year in os.listdir(DATA_DIR):
        year_dir = os.path.join(DATA_DIR, year)
        if not os.path.isdir(year_dir):
            continue

        # Поиск всех типов изображений
        for vis_type in ['fire', 'drought', 'nbr', 'natural']:
            image_files = glob.glob(os.path.join(year_dir, f"{vis_type}_*.png"))
            all_images.extend(image_files)

    print(f"Найдено {len(all_images)} изображений всего")

    # Подготовка данных
    X = []  # признаки
    y = []  # метки (1 - пожар, 0 - нет пожара)
    filenames = []  # сохраняем имена файлов для анализа

    # Создаем датасет с балансом классов
    fire_count = 0
    no_fire_count = 0

    # Ручная разметка или полуавтоматическая по имени файла
    print("Формирование датасета...")
    for img_path in tqdm(all_images):
        img = load_image(img_path)
        if img is None:
            continue

        # Извлекаем признаки из изображения
        features, red_percentage, bright_percentage = extract_features(img)

        # Классифицируем на основе порогов цвета и яркости
        # Эмпирически подобранные пороги для определения пожара
        is_fire = False

        # Для fire типа изображений - более высокая вероятность пожара
        if 'fire_' in os.path.basename(img_path):
            is_fire = red_percentage > 0.05 and bright_percentage > 0.02
        # Для nbr типа (burned area index)
        elif 'nbr_' in os.path.basename(img_path):
            is_fire = bright_percentage > 0.15
        # Для остальных типов - более строгие условия
        else:
            is_fire = red_percentage > 0.08 and bright_percentage > 0.05

        # Балансируем классы (добавляем не более 200 примеров каждого класса)
        if is_fire and fire_count < 200:
            X.append(features)
            y.append(1)
            filenames.append(img_path)
            fire_count += 1
        elif not is_fire and no_fire_count < 200:
            X.append(features)
            y.append(0)
            filenames.append(img_path)
            no_fire_count += 1

    print(f"Dataset created: {fire_count} images with forestfire, {no_fire_count} without foresfire")

    # Гарантируем что у нас есть оба класса
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        print("Adding synthetic examples...")
        if 1 not in unique_classes:  # Если нет пожаров
            # Создаем синтетический пример пожара
            x_avg = np.mean(np.array(X), axis=0)
            # Увеличиваем значения красного и яркого
            x_avg[-2] = 0.1  # red_percentage
            x_avg[-1] = 0.1  # bright_percentage
            X.append(x_avg)
            y.append(1)
            filenames.append("synthetic_fire.png")
        if 0 not in unique_classes:  # Если нет непожаров
            # Создаем синтетический пример непожара
            x_avg = np.mean(np.array(X), axis=0)
            # Уменьшаем значения красного и яркого
            x_avg[-2] = 0.01  # red_percentage
            x_avg[-1] = 0.01  # bright_percentage
            X.append(x_avg)
            y.append(0)
            filenames.append("synthetic_no_fire.png")

    # Преобразуем в numpy массивы
    X = np.array(X)
    y = np.array(y)

    # Сохраняем информацию о датасете
    dataset_info = pd.DataFrame({
        'filename': filenames,
        'label': y
    })
    dataset_info.to_csv(os.path.join(RESULTS_DIR, 'dataset_info.csv'), index=False)

    return X, y, filenames

def train_model(X, y):
    """Обучает Random Forest модель на данных"""
    # Проверяем, что есть примеры обоих классов
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        print(f"error: datasetcontains only one class {unique_classes[0]}. Impossible to train classifier")
        return None, None, None

    # Разделяем данные на тренировочные и тестовые
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    print("Training Random Forest...")
    # Инициализируем и обучаем Random Forest
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Проверяем, что модель обучилась обоим классам
    classes = model.classes_
    print(f"Model class: {classes}")

    # Сохраняем модель
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)

    # Оцениваем модель
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    print("Classification report:")
    print(classification_report(y_test, y_pred))

    # Матрица ошибок
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No fire', 'Fire'],
                yticklabels=['No fire', 'Fire'])
    plt.xlabel('Predicted values')
    plt.ylabel('True values')
    plt.title('Confusion Matrix RF')
    plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix_rf.png'))

    feature_importance = model.feature_importances_
    plt.figure(figsize=(12, 8))
    plt.bar(range(len(feature_importance)), feature_importance)
    plt.title('Feature importance')
    plt.savefig(os.path.join(RESULTS_DIR, 'feature_importance_rf.png'))

    return model, X_test, y_test

def predict_fire(image_path, model=None):
    """Предсказывает наличие пожара на изображении"""
    if model is None:
        # Загружаем модель, если не передана
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
        else:
            print("Model was not found")
            return None

    # Загружаем и обрабатываем изображение
    img = load_image(image_path)
    if img is None:
        return None

    # Извлекаем признаки
    features, red_percentage, bright_percentage = extract_features(img)

    # Проверяем классы модели
    has_multi_class = len(model.classes_) > 1

    # Прогнозируем
    prediction = model.predict([features])[0]

    # Получаем вероятность пожара безопасным способом
    probabilities = model.predict_proba([features])[0]
    fire_probability = 0.0

    # Находим вероятность класса "Пожар" (1)
    if has_multi_class:
        # Если модель знает оба класса
        for i, class_val in enumerate(model.classes_):
            if class_val == 1:
                fire_probability = probabilities[i]
                break
    else:
        # Если модель знает только один класс, используем эвристику
        if prediction == 1:
            fire_probability = 0.9  # Высокая вероятность
        else:
            # Используем признаки как запасной вариант для оценки
            fire_probability = min(red_percentage * 5, 0.9)

    # Возвращаем результат
    result = {
        'filename': image_path,
        'prediction': 'Fire' if prediction == 1 else 'No fire',
        'probability': fire_probability,
        'red_percentage': red_percentage,
        'bright_percentage': bright_percentage
    }

    return result

def visualize_prediction(image_path, result):
    img = cv2.imread(image_path)
    if img is None:
        print(f"visualize_prediction: error loading image: {image_path}")
        return None

    prediction = result.get('prediction', 'Unknown')
    probability = result.get('probability', "N/A")
    red_pct = result.get('red_pct', 0.0)
    bright_pct = result.get('bright_pct', 0.0)

    label_text = f"{prediction} (prob: {probability:.2f})" if isinstance(probability, float) else f"{prediction}"
    red_text = f"Red %: {red_pct:.3f}"
    bright_text = f"Bright %: {bright_pct:.3f}"

    color = (0, 0, 255) if prediction.lower() == 'fire' else (0, 255, 0)

    cv2.putText(img, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(img, red_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(img, bright_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    output_path = os.path.join(RESULTS_DIR, os.path.basename(image_path))
    cv2.imwrite(output_path, img)

    return output_path

def analyze_all_images():
    """Анализирует все изображения и создает отчет"""
    if not os.path.exists(MODEL_PATH):
        print("Модель не найдена. Сначала обучите модель.")
        return

    # Загружаем модель
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    results = []

    # Ищем все изображения
    for year in os.listdir(DATA_DIR):
        year_dir = os.path.join(DATA_DIR, year)
        if not os.path.isdir(year_dir):
            continue

        # Анализируем изображения каждого типа визуализации
        for vis_type in ['fire', 'drought', 'nbr', 'natural']:
            image_files = glob.glob(os.path.join(year_dir, f"{vis_type}_*.png"))

            print(f"Analysis of {len(image_files)} images of type {vis_type} for {year}")
            for img_path in tqdm(image_files):
                result = predict_fire(img_path, model)
                if result:
                    result['year'] = year
                    result['type'] = vis_type
                    results.append(result)

                    # Визуализируем если это пожар или с высокой вероятностью
                    if result['prediction'] == 'Пожар' or result['probability'] > 0.3:
                        visualize_prediction(img_path, result)

    # Создаем DataFrame с результатами
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(RESULTS_DIR, 'fire_analysis_results.csv'), index=False)

    # Статистика по годам
    if len(results) > 0:
        year_stats = results_df.groupby(['year', 'prediction']).size().unstack(fill_value=0)
        year_stats.plot(kind='bar', stacked=True, figsize=(10, 6))
        plt.title('Forestfire quantity by year')
        plt.xlabel('Year')
        plt.ylabel('Image quantity')
        plt.savefig(os.path.join(RESULTS_DIR, 'year_stats.png'))

    return results_df

def main():
    """Основная функция для запуска всего процесса"""
    print("Launching Fire Analysis on Satellite Images")

    # Если модель уже существует, спрашиваем пользователя, нужно ли переобучить
    if os.path.exists(MODEL_PATH):
        retrain = input("The model already exists. Retrain? (y/n):")
        if retrain.lower() != 'y':
            print("Skipping training.")
            analyze_all_images()
            return

    # Подготовка данных и обучение модели
    print("Preparing dataset for training...")
    X, y, filenames = prepare_dataset()

    if len(X) == 0:
        print("No data for training.")
        return

    # Обучение модели
    model, X_test, y_test = train_model(X, y)

    if model is None:
        print("Failed to train model. Please check your data.")
        return

    # Анализ всех изображений
    print("Analyzing all images...")
    results_df = analyze_all_images()

    print(f"Analysis completed. Results saved in {RESULTS_DIR}")

if __name__ == "__main__":
    main()
