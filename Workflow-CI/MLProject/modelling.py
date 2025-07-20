import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Set experiment name
mlflow.set_experiment("Loan Status Prediction")

# Load data
df = pd.read_csv("Workflow-CI/MLProject/loan_data_preprocessing.csv")
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run():
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Log Parameter dan Metric
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("max_iter", 200)
    mlflow.log_metric("accuracy", acc)

    # Simpan model ke MLflow
    mlflow.sklearn.log_model(model, "model")

    print(f"Accuracy: {acc}")