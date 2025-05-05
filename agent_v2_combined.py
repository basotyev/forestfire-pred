
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
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

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
    red_mask = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255)) + cv2.inRange(hsv, (160, 100, 100), (180, 255, 255))
    red_percentage = np.sum(red_mask > 0) / image.size
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]
    bright_percentage = np.sum(bright_mask > 0) / image.size
    return np.concatenate([bgr_means, bgr_std, hsv_means, hsv_std, histograms, [red_percentage, bright_percentage]]), red_percentage, bright_percentage

def build_dataset(years, return_paths=False):
    X, y, paths = [], [], []
    for year in years:
        for vis_type in ['fire', 'drought', 'nbr', 'natural']:
            image_paths = glob.glob(os.path.join(DATA_DIR, year, f"{vis_type}_*.png"))
            for path in image_paths:
                img = load_image(path)
                if img is None:
                    continue
                features, red, bright = extract_features(img)
                filename = os.path.basename(path)
                if "fire" in filename:
                    is_fire = red > 0.05 and bright > 0.02
                elif "nbr" in filename:
                    is_fire = bright > 0.15
                else:
                    is_fire = red > 0.08 and bright > 0.05
                label = 1 if is_fire else 0
                X.append(features)
                y.append(label)
                if return_paths:
                    paths.append(path)
    return (np.array(X), np.array(y), paths) if return_paths else (np.array(X), np.array(y))

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
    cm = confusion_matrix(y_test, y_pred)
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
    X_train, y_train = build_dataset(train_years)
    X_test, y_test, test_paths = build_dataset(test_years, return_paths=True)
    train_and_evaluate("rf", lambda: RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42), X_train, y_train, X_test, y_test, test_paths)
    train_and_evaluate("svm", lambda: LinearSVC(class_weight='balanced', max_iter=10000), X_train, y_train, X_test, y_test, test_paths)

if __name__ == "__main__":
    main()
