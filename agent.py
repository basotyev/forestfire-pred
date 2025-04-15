import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FirePredictionAgent:
    def __init__(self, model_path=None, algorithm="rf"):
        self.model = None
        self.algorithm = algorithm
        self.scaler = None
        if model_path:
            self.model = load(model_path)

    def train(self, data_path, save_model_path=None):
        try:
            data = pd.read_csv(data_path)

            required_columns = ["Latitude", "Longitude", "Humidity", "Temperature", "DistanceFromReference", "Fire"]
            if not all(col in data.columns for col in required_columns):
                raise ValueError(f"Данные должны содержать следующие колонки: {required_columns}")

            X = data[["Latitude", "Longitude", "Humidity", "Temperature", "DistanceFromReference"]]
            y = data["Fire"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            if self.algorithm == "svm":
                logger.info("[INFO] Обучаем SVM-модель...")

                self.scaler = StandardScaler()
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)

                param_grid = {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'gamma': [0.01, 0.1, 1, 'scale']
                }

                svm_base = SVC(probability=True, class_weight='balanced', kernel='rbf', random_state=42)

                grid_search = GridSearchCV(
                    estimator=svm_base,
                    param_grid=param_grid,
                    cv=5,
                    scoring='accuracy',
                    n_jobs=-1
                )
                grid_search.fit(X_train_scaled, y_train)

                self.model = grid_search.best_estimator_

                y_pred = self.model.predict(X_test_scaled)
                logger.info(f"Accuracy: {accuracy_score(y_test, y_pred)}")
                logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")

                if not save_model_path:
                    save_model_path = "fire_model_svm.joblib"

            else:
                # Random Forest
                logger.info("[INFO] Обучаем Random Forest...")
                self.model = RandomForestClassifier(n_estimators=100, random_state=42)
                self.model.fit(X_train, y_train)

                y_pred = self.model.predict(X_test)
                logger.info(f"Accuracy: {accuracy_score(y_test, y_pred)}")
                logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")

                if not save_model_path:
                    save_model_path = "fire_model_rf.joblib"

            dump(self.model, save_model_path)
            logger.info(f"Модель сохранена в {save_model_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return False

    def predict(self, input_data):
        if self.model is None:
            raise ValueError("Модель не загружена. Сначала обучите или загрузите модель.")

        # Convert dictionary to DataFrame if needed
        if isinstance(input_data, dict):
            input_data = pd.DataFrame([input_data])
        elif isinstance(input_data, list):
            input_data = pd.DataFrame(input_data)

        required_columns = ["Latitude", "Longitude", "Humidity", "Temperature", "DistanceFromReference"]
        if not all(col in input_data.columns for col in required_columns):
            raise ValueError(f"Входные данные должны содержать следующие колонки: {required_columns}")

        X = input_data[["Latitude", "Longitude", "Humidity", "Temperature", "DistanceFromReference"]]

        if self.algorithm == "svm" and self.scaler is not None:
            X = self.scaler.transform(X)

        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]

        input_data["Fire_Probability"] = probabilities
        input_data["Fire_Prediction"] = predictions

        return input_data


if __name__ == "__main__":
    new_data = [
        {"Latitude": 49.0,
         "Longitude": 80.0,
         "Humidity": 35.0,
         "Temperature": 25.0,
         "DistanceFromReference": 120.3
         },
        {"Latitude": 49.5,
         "Longitude": 80.5,
         "Humidity": 45.0,
         "Temperature": 22.0,
         "DistanceFromReference": 150.7
         }
    ]

    # RF
    rf_agent = FirePredictionAgent(algorithm="rf")
    rf_agent.train("processed_data_2023.csv")
    rf_predictions = rf_agent.predict(new_data)
    logger.info("[RF PREDICTIONS]:")
    logger.info(rf_predictions)

    # SVM
    svm_agent = FirePredictionAgent(algorithm="svm")
    svm_agent.train("processed_data_2023.csv")
    svm_predictions = svm_agent.predict(new_data)
    logger.info("[SVM PREDICTIONS]:")
    logger.info(svm_predictions)
