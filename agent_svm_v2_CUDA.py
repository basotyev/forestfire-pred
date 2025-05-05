import os
import cv2
import numpy as np
from cuml.svm import SVC as cuSVC
from cuml.preprocessing.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def extract_features(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    b, g, r = cv2.split(img)

    stats = [
        np.mean(r), np.std(r),
        np.mean(g), np.std(g),
        np.mean(b), np.std(b),
        np.mean(h), np.std(h),
        np.mean(s), np.std(s),
        np.mean(v), np.std(v)
    ]

    hist_s = cv2.calcHist([s], [0], None, [32], [0, 256]).flatten()
    hist_v = cv2.calcHist([v], [0], None, [32], [0, 256]).flatten()

    red_pixels = np.sum((r > 150) & (g < 80) & (b < 80)) / img.size
    bright_pixels = np.sum(v > 200) / img.size

    features = stats + hist_s.tolist() + hist_v.tolist() + [red_pixels, bright_pixels]
    return np.array(features, dtype=np.float32)

def load_dataset(data_dir):
    X, y = [], []
    for label_name in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label_name)
        if not os.path.isdir(label_path): continue
        label = 1 if "fire" in label_name.lower() else 0

        for file in os.listdir(label_path):
            img_path = os.path.join(label_path, file)
            img = cv2.imread(img_path)
            if img is None: continue
            img = cv2.resize(img, (128, 128))
            features = extract_features(img)
            X.append(features)
            y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

def train_gpu_svm(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    model = cuSVC(kernel='rbf', C=1.0, probability=True)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Classification report:")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(model, 'svm_gpu_model.pkl')
    joblib.dump(scaler, 'svm_gpu_scaler.pkl')
    return model, scaler

def predict_images(model, scaler, folder):
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        img = cv2.imread(path)
        if img is None: continue
        img_resized = cv2.resize(img, (128, 128))
        features = extract_features(img_resized)
        features_scaled = scaler.transform([features])

        prob = model.predict_proba(features_scaled)[0][1]
        prediction = model.predict(features_scaled)[0]
        label = "FIRE" if prediction == 1 else "NO FIRE"

        cv2.putText(img, f"{label} ({prob*100:.1f}%)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255) if prediction == 1 else (0, 255, 0), 2)

        cv2.imshow("Prediction", img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    data_dir = "dataset"
    X, y = load_dataset(data_dir)
    model, scaler = train_gpu_svm(X, y)
    predict_images(model, scaler, "unseen")
