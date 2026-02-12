from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

from train_test_split import load_data


class LogisticRegressionModel:
	def __init__(self):
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
					LogisticRegression(
						max_iter=1000,
						solver="liblinear",
						class_weight="balanced",
						random_state=42,
					),
				),
			]
		)

	def train(self):
		self.model.fit(self.X_train, self.y_train)
		return self

	def predict(self):
		return self.model.predict(self.X_test)

	def evaluate(self):
		y_pred = self.predict()
		accuracy = accuracy_score(self.y_test, y_pred)
		report = classification_report(self.y_test, y_pred)
		matrix = confusion_matrix(self.y_test, y_pred)
		return accuracy, report, matrix

	def run(self):
		self.train()
		accuracy, report, matrix = self.evaluate()
		print(f"Accuracy: {accuracy:.4f}")
		print("\nClassification Report:\n")
		print(report)
		print("Confusion Matrix:\n")
		print(matrix)


def main():
	LogisticRegressionModel().run()


if __name__ == "__main__":
	main()
