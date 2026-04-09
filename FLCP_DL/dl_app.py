from pathlib import Path
import importlib.util
import warnings

import numpy as np
import pickle
from flask import Flask, render_template, request

warnings.filterwarnings("ignore")


app = Flask(__name__)

MODULE_DIR = Path(__file__).resolve().parent
ROOT_DIR = MODULE_DIR.parent
DEFAULT_PORT = 5001


def artifact_candidates(filename):
    return [MODULE_DIR / filename, ROOT_DIR / filename]


def resolve_artifact_path(filename):
    for candidate in artifact_candidates(filename):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Unable to locate {filename}. Checked: "
        + ", ".join(str(candidate) for candidate in artifact_candidates(filename))
    )


def load_artifacts():
    if importlib.util.find_spec("tensorflow") is None:
        raise RuntimeError("TensorFlow is not installed. DNN predictions are unavailable.")

    from tensorflow.keras.models import load_model

    model_path = resolve_artifact_path("dl_model.h5")
    scaler_path = resolve_artifact_path("scaler.pkl")
    accuracy_path = resolve_artifact_path("model_accuracy.pkl")

    model = load_model(model_path)
    with scaler_path.open("rb") as scaler_file:
        scaler = pickle.load(scaler_file)
    with accuracy_path.open("rb") as accuracy_file:
        model_info = pickle.load(accuracy_file)

    return model, scaler, model_info.get("accuracy", 0) * 100


def get_runtime_state():
    try:
        model, scaler, model_accuracy = load_artifacts()
        return {
            "model": model,
            "scaler": scaler,
            "model_accuracy": model_accuracy,
            "startup_error": None,
        }
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        return {
            "model": None,
            "scaler": None,
            "model_accuracy": 0,
            "startup_error": str(exc),
        }


def predict_from_values(values):
    runtime = get_runtime_state()
    if runtime["model"] is None or runtime["scaler"] is None:
        return {
            "available": False,
            "prediction": None,
            "label": "Unavailable",
            "probability": None,
            "confidence": None,
            "accuracy": runtime["model_accuracy"],
            "note": runtime["startup_error"] or "Model artifacts are not available.",
        }

    data = np.array([values], dtype=float)
    data_scaled = runtime["scaler"].transform(data)
    prediction_prob = float(runtime["model"].predict(data_scaled, verbose=0)[0][0])
    is_high_risk = prediction_prob > 0.5

    return {
        "available": True,
        "prediction": int(is_high_risk),
        "label": "High Risk" if is_high_risk else "Low Risk",
        "probability": prediction_prob * 100,
        "confidence": max(prediction_prob, 1 - prediction_prob) * 100,
        "accuracy": runtime["model_accuracy"],
        "note": "Probability-based DNN risk assessment",
    }


runtime_state = get_runtime_state()
model = runtime_state["model"]
scaler = runtime_state["scaler"]
model_accuracy = runtime_state["model_accuracy"]
startup_error = runtime_state["startup_error"]

if startup_error:
    print(f"ERROR: {startup_error}")
    print("Please run FLCP_DL/dl_model.py first to train the model.")
else:
    print("Model and scaler loaded successfully")
    print(f"Model Test Accuracy: {model_accuracy:.2f}%")


features = [
    "AGE",
    "GENDER",
    "SMOKING",
    "FINGER_DISCOLORATION",
    "MENTAL_STRESS",
    "EXPOSURE_TO_POLLUTION",
    "LONG_TERM_ILLNESS",
    "ENERGY_LEVEL",
    "IMMUNE_WEAKNESS",
    "BREATHING_ISSUE",
    "ALCOHOL_CONSUMPTION",
    "THROAT_DISCOMFORT",
    "OXYGEN_SATURATION",
    "CHEST_TIGHTNESS",
    "FAMILY_HISTORY",
    "SMOKING_FAMILY_HISTORY",
    "STRESS_IMMUNE",
]


@app.route("/")
def home():
    return render_template(
        "index.html",
        features=features,
        model_accuracy=f"{model_accuracy:.2f}%",
        startup_error=startup_error,
    )


@app.route("/predict", methods=["POST"])
def predict():
    if model is None or scaler is None:
        return render_template(
            "index.html",
            features=features,
            model_accuracy=f"{model_accuracy:.2f}%",
            startup_error=startup_error or "Model artifacts are not available.",
        ), 500

    try:
        values = []
        for feature in features:
            value = request.form.get(feature)
            if value is None or value == "":
                raise ValueError(f"Please select a value for {feature}.")
            values.append(float(value))
        result = predict_from_values(values)

        return render_template(
            "index.html",
            features=features,
            prediction_text=result["label"],
            probability=f"{result['probability']:.1f}%",
            confidence=f"{result['confidence']:.1f}%",
            model_accuracy=f"{model_accuracy:.2f}%",
        )
    except ValueError as exc:
        return render_template(
            "index.html",
            features=features,
            model_accuracy=f"{model_accuracy:.2f}%",
            startup_error=str(exc),
        ), 400
    except Exception as exc:
        return render_template(
            "index.html",
            features=features,
            model_accuracy=f"{model_accuracy:.2f}%",
            startup_error=f"Prediction error: {exc}",
        ), 500


if __name__ == "__main__":
    app.run(debug=True, port=DEFAULT_PORT)
