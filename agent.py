import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump, load


class FirePredictionAgent:
    def __init__(self, model_path=None):
        self.model = None
        if model_path:
            self.model = load(model_path)

    def train(self, data_path, save_model_path="fire_model.joblib"):
        data = pd.read_csv(data_path)

        required_columns = ["Latitude", "Longitude", "Humidity", "Temperature", "Fire"]
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Данные должны содержать следующие колонки: {required_columns}")

        X = data[["Latitude", "Longitude", "Humidity", "Temperature"]]
        y = data["Fire"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))

        dump(self.model, save_model_path)
        print(f"Модель сохранена в {save_model_path}")

    def predict(self, input_data):
        if self.model is None:
            raise ValueError("Модель не загружена. Сначала обучите или загрузите модель.")

        if isinstance(input_data, list):
            input_data = pd.DataFrame(input_data)

        required_columns = ["Latitude", "Longitude", "Humidity", "Temperature"]
        if not all(col in input_data.columns for col in required_columns):
            raise ValueError(f"Входные данные должны содержать следующие колонки: {required_columns}")

        predictions = self.model.predict(input_data)
        probabilities = self.model.predict_proba(input_data)[:, 1]  # Вероятность пожара

        input_data["Fire_Probability"] = probabilities
        input_data["Fire_Prediction"] = predictions

        return input_data


if __name__ == "__main__":
    agent = FirePredictionAgent()

    agent.train("processed_data.csv")

    new_data = [
        {"Latitude": 49.0, "Longitude": 80.0, "Humidity": 35.0, "Temperature": 25.0},
        {"Latitude": 49.5, "Longitude": 80.5, "Humidity": 45.0, "Temperature": 22.0}
    ]
    predictions = agent.predict(new_data)
    print(predictions)
