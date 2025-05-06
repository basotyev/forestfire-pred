
import os
import numpy as np
import pandas as pd
import cv2
import pickle
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

DATA_DIR = "satellite_images"

def load_image(file_path):
    try:
        img = cv2.imread(file_path)
        if img is None:
            print(f"error reading image {file_path}")
            return None
        return img
    except Exception as e:
        print(f"error loading image {file_path}: {e}")
        return None

def extract_features(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    bgr_means = np.mean(image, axis=(0, 1))
    hsv_means = np.mean(hsv, axis=(0, 1))
    bgr_std = np.std(image, axis=(0, 1))
    hsv_std = np.std(hsv, axis=(0, 1))
    b, g, r = cv2.split(image)
    h, s, v = cv2.split(hsv)
    histograms = np.concatenate([
        cv2.calcHist([b], [0], None, [16], [0, 256]).flatten(),
        cv2.calcHist([g], [0], None, [16], [0, 256]).flatten(),
        cv2.calcHist([r], [0], None, [16], [0, 256]).flatten(),
        cv2.calcHist([h], [0], None, [16], [0, 180]).flatten(),
        cv2.calcHist([s], [0], None, [16], [0, 256]).flatten(),
        cv2.calcHist([v], [0], None, [16], [0, 256]).flatten(),
    ])
    histograms /= np.sum(histograms)
    pixels = image.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, _, centers = cv2.kmeans(pixels, 5, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = centers.flatten()  # 5 кластеров по 3 канала = 15 признаков
    red_mask = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255)) + \
               cv2.inRange(hsv, (160, 100, 100), (180, 255, 255))
    red_percentage = np.sum(red_mask > 0) / image.size
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]
    bright_percentage = np.sum(bright_mask > 0) / image.size

    return np.concatenate([
        bgr_means, bgr_std, hsv_means, hsv_std,
        centers, histograms,
        [red_percentage, bright_percentage]
    ]), red_percentage, bright_percentage

def build_combined_dataset(years):
    X, y, paths = [], [], []
    for year in years:
        grouped = {}
        for vis_type in ['fire', 'drought', 'natural', 'nbr']:
            for path in glob.glob(os.path.join(DATA_DIR, year, f"{vis_type}_*.png")):
                filename = os.path.basename(path)
                date_part = filename.split("_")[1]
                grouped.setdefault(date_part, {})[vis_type] = path

        print(f"\n[{year}] Обработка {len(grouped)} комплектов изображений")

        for date, imgs in grouped.items():
            if len(imgs) != 4:
                continue

            all_feats = []
            skip = False
            bright_values = {}
            for vis in ['fire', 'drought', 'natural', 'nbr']:
                img = load_image(imgs[vis])
                if img is None:
                    skip = True
                    break
                feats, _, bright = extract_features(img)
                all_feats.extend(feats)
                bright_values[vis] = bright
            if skip:
                continue

            fire_b = bright_values['fire']
            nbr_b = bright_values['nbr']
            nat_b = bright_values['natural']
            drought_b = bright_values['drought']

            # Основной триггер на пожар
            if fire_b > 0.30 or nbr_b > 0.25:
                label = 1
            # Явно тёмные, невыразительные изображения
            elif nat_b < 0.03 and nbr_b < 0.2 and drought_b < 0.05:
                label = 0
            # Умеренное состояние: проверим среднее
            elif (drought_b + nat_b) / 2 > 0.3:
                label = 1
            else:
                label = 0

            print(f"[{year}] {date} → fire: {bright_values['fire']:.3f}, nbr: {bright_values['nbr']:.3f}, natural: {bright_values['natural']:.3f}, drought: {bright_values['drought']} → label: {label}")
            X.append(np.array(all_feats))
            y.append(label)
            paths.append(imgs['nbr'])

    X = np.array(X)
    y = np.array(y)
    paths = np.array(paths)

    unique, counts = np.unique(y, return_counts=True)
    print("📊 Распределение меток:", dict(zip(unique, counts)))
    return X, y, paths


def train_and_evaluate(model_name, model_class, X_train, y_train, X_test, y_test, test_paths):
    print(f"=== Model: {model_name} ===")
    results_dir = f"results_{model_name}"
    os.makedirs(results_dir, exist_ok=True)
    fire_dir = os.path.join(results_dir, "fire_predictions")
    os.makedirs(fire_dir, exist_ok=True)

    model_path = os.path.join(results_dir, "model.pkl")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = model_class()
    model.fit(X_train_scaled, y_train)
    with open(model_path, "wb") as f:
        pickle.dump((model, scaler), f)

    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No fire', 'Fire'],
                yticklabels=['No fire', 'Fire'])
    plt.title(f'Confusion Matrix ({model_name})')
    plt.savefig(os.path.join(results_dir, f'confusion_matrix_{model_name}.png'))
    plt.close()

    results = []
    for i, (features, path) in enumerate(zip(X_test, test_paths)):
        features_scaled = scaler.transform([features])
        pred = model.predict(features_scaled)[0]
        label = "Fire" if pred == 1 else "No fire"
        results.append({
            "path": path,
            "prediction": label
        })
        if label == "Fire":
            filename = os.path.basename(path)
            img = load_image(path)
            if img is not None:
                cv2.imwrite(os.path.join(fire_dir, filename), img)

    pd.DataFrame(results).to_csv(os.path.join(results_dir, f"fire_analysis_{model_name}.csv"), index=False)

def main():
    all_years = [y for y in os.listdir(DATA_DIR) if y.isdigit()]
    train_years = [y for y in all_years if y != "2023"]
    test_years = ["2023"]
    X_train, y_train, _ = build_combined_dataset(train_years)
    X_test, y_test, test_paths = build_combined_dataset(test_years)
    print("Train label distribution:", dict(zip(*np.unique(y_train, return_counts=True))))
    print("Test label distribution:", dict(zip(*np.unique(y_test, return_counts=True))))
    train_and_evaluate("rf", lambda: RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42), X_train, y_train, X_test, y_test, test_paths)
    train_and_evaluate("svm", lambda: SVC(kernel='rbf', class_weight='balanced', probability=True), X_train, y_train, X_test, y_test, test_paths)

if __name__ == "__main__":
    main()
