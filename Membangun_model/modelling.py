import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Set experiment (manual)
mlflow.set_experiment("Loan Status Prediction")

# Load preprocessed dataset
df = pd.read_csv("loan_data_preprocessing.csv")

# Memisahkan fitur dan target
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.sklearn.autolog()

with mlflow.start_run():
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc}")