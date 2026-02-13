import os
import time

import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from models.utils import evaluate_model
from train_test_split import load_data


class KNNModel:
	def __init__(self, n_neighbors=5):
		X_train, X_test, y_train, y_test, preprocessor = load_data()
		self.X_train = X_train
		self.X_test = X_test
		self.y_train = y_train
		self.y_test = y_test
		self.model = Pipeline(
			steps=[
				("preprocessor", preprocessor),
				(
					"classifier",
					KNeighborsClassifier(n_neighbors=n_neighbors),
				),
			]
		)

	def train(self):
		print("Training started...")
		start_time = time.time()
		self.model.fit(self.X_train, self.y_train)
		elapsed = time.time() - start_time
		print(f"Training completed in {elapsed:.2f}s.")
		return self

	def predict(self):
		return self.model.predict(self.X_test)

	def evaluate(self):
		y_pred = self.predict()
		accuracy = accuracy_score(self.y_test, y_pred)
		report = classification_report(self.y_test, y_pred)
		matrix = confusion_matrix(self.y_test, y_pred)
		return y_pred, accuracy, report, matrix

	def save_model(self, path="model_files/knn.pkl"):
		os.makedirs(os.path.dirname(path), exist_ok=True)
		joblib.dump(self.model, path)
		print(f"Model saved to: {path}")

	def save_metrics(self, metrics_df, path="results/knn_metrics.csv"):
		os.makedirs(os.path.dirname(path), exist_ok=True)
		metrics_df.to_csv(path, index=False)
		print(f"Metrics saved to: {path}")

	def run(self):
		print("KNN training run")
		self.train()
		self.save_model()
		y_pred, accuracy, report, matrix = self.evaluate()
		y_probs = self.model.predict_proba(self.X_test)[:, 1]
		metrics_df = evaluate_model("KNN", self.y_test, y_pred, y_probs)
		self.save_metrics(metrics_df)
		print(f"Accuracy: {accuracy:.4f}")
		print("\nClassification Report:\n")
		print(report)
		print("Confusion Matrix:\n")
		print(matrix)
		print("\nMetrics Summary:\n")
		print(metrics_df.to_string(index=False))


def main():
	KNNModel().run()


if __name__ == "__main__":
	main()
