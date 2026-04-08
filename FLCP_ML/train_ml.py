import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import json

mlflow.set_experiment("ML Experiment")

# Load dataset
df = pd.read_csv("DATA/lungcancer_clean.csv")

# Fix column names
df.columns = df.columns.str.strip()
df.columns = df.columns.str.upper()
df.columns = df.columns.str.replace(" ", "_")  # 🔥 THIS FIXES IT

print("Columns:", df.columns.tolist())

# Preprocess
df = df.replace({
    "YES": 1, "NO": 0,
    "Yes": 1, "No": 0,
    "M": 1, "F": 0
})

# Split
X = df.drop("LUNG_CANCER", axis=1)
y = df["LUNG_CANCER"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train
with mlflow.start_run(run_name="ML_Model"):

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    print("ML Accuracy:", accuracy)

    # Log metric
    mlflow.log_metric("accuracy", accuracy)

    # Log model
    mlflow.sklearn.log_model(model, name="model")

# Save result
with open("FLCP_ML/ml_results.json", "w") as f:
    json.dump({"accuracy": accuracy}, f)

print("✅ ML results saved")