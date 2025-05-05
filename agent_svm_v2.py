import os
import numpy as np
import pandas as pd
import cv2
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
import pickle
import seaborn as sns

# Пути к данным
DATA_DIR = "satellite_images"
MODEL_PATH = "fire_detection_model_svm.pkl"
RESULTS_DIR = "results_svm"

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

def load_image(file_path):
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
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    bgr_means = np.mean(image, axis=(0, 1))
    hsv_means = np.mean(hsv, axis=(0, 1))
    bgr_std = np.std(image, axis=(0, 1))
    hsv_std = np.std(hsv, axis=(0, 1))

    pixels = image.reshape(-1, 3)
    pixels = np.float32(pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 5
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    dominant_colors = centers

    hist_b = cv2.calcHist([image], [0], None, [32], [0, 256])
    hist_g = cv2.calcHist([image], [1], None, [32], [0, 256])
    hist_r = cv2.calcHist([image], [2], None, [32], [0, 256])
    hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])

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

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    bright_percentage = np.sum(bright_mask > 0) / total_pixels

    features = np.concatenate([
        bgr_means, bgr_std,
        hsv_means, hsv_std,
        dominant_colors.flatten(),
        hist_b.flatten() / np.sum(hist_b),
        hist_g.flatten() / np.sum(hist_g),
        hist_r.flatten() / np.sum(hist_r),
        hist_h.flatten() / np.sum(hist_h),
        hist_s.flatten() / np.sum(hist_s),
        hist_v.flatten() / np.sum(hist_v),
        [red_percentage, bright_percentage]
    ])
    return features, red_percentage, bright_percentage

def prepare_dataset():
    X, y, filenames = [], [], []
    all_images = glob.glob(os.path.join(DATA_DIR, "*/*.png"))
    fire_count, no_fire_count = 0, 0

    for img_path in tqdm(all_images):
        img = load_image(img_path)
        if img is None:
            continue
        features, red_percentage, bright_percentage = extract_features(img)
        is_fire = False

        if 'fire_' in os.path.basename(img_path):
            is_fire = red_percentage > 0.05 and bright_percentage > 0.02
        elif 'nbr_' in os.path.basename(img_path):
            is_fire = bright_percentage > 0.15
        else:
            is_fire = red_percentage > 0.08 and bright_percentage > 0.05

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

    print(f"Создан датасет: {fire_count} изображений с пожарами, {no_fire_count} без пожаров")

    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        print("Добавляем синтетические примеры...")
        x_avg = np.mean(np.array(X), axis=0)
        if 1 not in unique_classes:
            x_avg[-2] = 0.1
            x_avg[-1] = 0.1
            X.append(x_avg)
            y.append(1)
            filenames.append("synthetic_fire.png")
        if 0 not in unique_classes:
            x_avg[-2] = 0.01
            x_avg[-1] = 0.01
            X.append(x_avg)
            y.append(0)
            filenames.append("synthetic_no_fire.png")

    X = np.array(X)
    y = np.array(y)

    pd.DataFrame({'filename': filenames, 'label': y}).to_csv(
        os.path.join(RESULTS_DIR, 'dataset_info.csv'), index=False)
    return X, y, filenames

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

def train_model(X, y):
    if len(np.unique(y)) < 2:
        print("ОШИБКА: только один класс в данных.")
        return None, None, None

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

    print("Обучение модели LinearSVC.")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearSVC(class_weight='balanced', max_iter=10000, random_state=42)
    model.fit(X_train_scaled, y_train)

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump((model, scaler), f)

    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"Точность: {acc:.4f}")
    print("Классификационный отчет:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=['Нет пожара', 'Пожар'],
                yticklabels=['Нет пожара', 'Пожар'])
    plt.title('Матрица ошибок (LinearSVC)')
    plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix_linear_svc.png'))
    return model, X_test_scaled, y_test


def predict_fire(image_path, model=None, scaler=None):
    if model is None or scaler is None:
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, 'rb') as f:
                model, scaler = pickle.load(f)
        else:
            print("Модель не найдена.")
            return None
    else:
        scaler = StandardScaler()

    img = load_image(image_path)
    if img is None:
        return None

    features, red_percentage, bright_percentage = extract_features(img)
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)[0]
    prob = None
    label = "Пожар" if prediction == 1 else "Нет пожара"

    return {
        'path': image_path,
        'prediction': label,
        'probability': "N/A",
        'red_pct': red_percentage,
        'bright_pct': bright_percentage
    }

def analyze_all_images():
    with open(MODEL_PATH, 'rb') as f:
        model, scaler = pickle.load(f)

    results = []
    for year in os.listdir(DATA_DIR):
        year_dir = os.path.join(DATA_DIR, year)
        if not os.path.isdir(year_dir):
            continue

        for vis_type in ['fire', 'drought', 'nbr', 'natural']:
            image_files = glob.glob(os.path.join(year_dir, f"{vis_type}_*.png"))
            print(f"Анализ {len(image_files)} изображений типа {vis_type} за {year}")
            for img_path in tqdm(image_files):
                result = predict_fire(img_path, model)
                if result:
                    result['year'] = year
                    result['type'] = vis_type
                    results.append(result)

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(RESULTS_DIR, 'fire_analysis_results_svm.csv'), index=False)
    return results_df

def main():
    print("Запуск анализа пожаров (SVM)...")
    if os.path.exists(MODEL_PATH):
        retrain = input("Модель уже существует. Переобучить? (y/n): ")
        if retrain.lower() != 'y':
            print("Пропускаем обучение.")
            analyze_all_images()
            return

    X, y, filenames = prepare_dataset()
    if len(X) == 0:
        print("Нет данных для обучения.")
        return

    model, _, _ = train_model(X, y)
    if model is None:
        return

    results_df = analyze_all_images()
    print(f"Анализ завершён. Результаты сохранены в {RESULTS_DIR}")

if __name__ == "__main__":
    main()
