import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os

# Set experiment name (opsional)
mlflow.set_experiment("Loan Status Prediction")

# Load data dari path relatif terhadap lokasi script ini
csv_path = os.path.join(os.path.dirname(__file__), "loan_data_preprocessing.csv")
df = pd.read_csv(csv_path)

# Pisahkan fitur dan target
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training dan Logging ke MLflow
with mlflow.start_run() as run:
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Log parameter dan metrik
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("max_iter", 200)
    mlflow.log_metric("accuracy", acc)

    # Simpan model
    input_example = X_train[:2]
    mlflow.sklearn.log_model(model, artifact_path="model", input_example=input_example)

    print("Run ID:", run.info.run_id)
    print("Accuracy:", acc)