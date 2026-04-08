from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

if __package__:
    from .ML.decision_tree import train_dt
    from .ML.knn import train_knn
    from .ML.random_forest import train_rf
else:
    from ML.decision_tree import train_dt
    from ML.knn import train_knn
    from ML.random_forest import train_rf


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "DATA" / "lungcancer_clean.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_dataset():
    df = pd.read_csv(DATASET_PATH)
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

    return df


def train_models():
    df = load_dataset()
    x = df.drop("LUNG_CANCER", axis=1)
    y = df["LUNG_CANCER"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    knn = train_knn(x_train_scaled, y_train)
    dt = train_dt(x_train_scaled, y_train)
    rf = train_rf(x_train_scaled, y_train)

    metrics = {
        "knn_acc": accuracy_score(y_test, knn.predict(x_test_scaled)) * 100,
        "dt_acc": accuracy_score(y_test, dt.predict(x_test_scaled)) * 100,
        "rf_acc": accuracy_score(y_test, rf.predict(x_test_scaled)) * 100,
    }

    return {
        "scaler": scaler,
        "knn": knn,
        "dt": dt,
        "rf": rf,
        "metrics": metrics,
        "feature_names": list(x.columns),
    }


MODEL_STATE = train_models()
FEATURE_NAMES = MODEL_STATE["feature_names"]


def predict_all(input_data):
    if len(input_data) != len(FEATURE_NAMES):
        raise ValueError(f"Expected {len(FEATURE_NAMES)} inputs, received {len(input_data)}.")

    input_data_scaled = MODEL_STATE["scaler"].transform([input_data])

    k = MODEL_STATE["knn"].predict(input_data_scaled)[0]
    d = MODEL_STATE["dt"].predict(input_data_scaled)[0]
    r = MODEL_STATE["rf"].predict(input_data_scaled)[0]
    final = round((k + d + r) / 3)

    metrics = MODEL_STATE["metrics"]
    return k, d, r, final, metrics["knn_acc"], metrics["dt_acc"], metrics["rf_acc"]
