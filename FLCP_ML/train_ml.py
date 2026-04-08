import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

if __package__:
    from .ML.decision_tree import train_dt
    from .ML.knn import train_knn
    from .ML.random_forest import train_rf
else:
    from ML.decision_tree import train_dt
    from ML.knn import train_knn
    from ML.random_forest import train_rf


mlflow.set_experiment("ML Experiment")
BASE_DIR = Path(__file__).resolve().parent.parent
MODULE_DIR = Path(__file__).resolve().parent


df = pd.read_csv(BASE_DIR / "DATA" / "lungcancer_clean.csv")
df.columns = df.columns.str.strip().str.upper().str.replace(" ", "_")
df = df.replace({
    "YES": 1,
    "NO": 0,
    "Yes": 1,
    "No": 0,
    "M": 1,
    "F": 0,
})
df = df.apply(pd.to_numeric)

x = df.drop("LUNG_CANCER", axis=1)
y = df["LUNG_CANCER"]

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

with mlflow.start_run(run_name="ML_Model_Comparison"):
    knn = train_knn(x_train, y_train)
    dt = train_dt(x_train, y_train)
    rf = train_rf(x_train, y_train)

    knn_accuracy = accuracy_score(y_test, knn.predict(x_test))
    dt_accuracy = accuracy_score(y_test, dt.predict(x_test))
    rf_accuracy = accuracy_score(y_test, rf.predict(x_test))

    best_model_name, best_accuracy = max(
        [
            ("KNN", knn_accuracy),
            ("Decision Tree", dt_accuracy),
            ("Random Forest", rf_accuracy),
        ],
        key=lambda item: item[1],
    )

    mlflow.log_metric("knn_accuracy", knn_accuracy)
    mlflow.log_metric("decision_tree_accuracy", dt_accuracy)
    mlflow.log_metric("random_forest_accuracy", rf_accuracy)
    mlflow.log_metric("best_accuracy", best_accuracy)
    mlflow.log_param("best_model", best_model_name)
    mlflow.sklearn.log_model(rf, name="random_forest_model")

with (MODULE_DIR / "ml_results.json").open("w", encoding="utf-8") as result_file:
    json.dump(
        {
            "knn_accuracy": knn_accuracy,
            "decision_tree_accuracy": dt_accuracy,
            "random_forest_accuracy": rf_accuracy,
            "best_model": best_model_name,
            "best_accuracy": best_accuracy,
        },
        result_file,
    )

print("ML comparison results saved")
