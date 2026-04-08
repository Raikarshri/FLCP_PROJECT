from pathlib import Path
import warnings

import numpy as np
import pickle
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model

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
    model_path = resolve_artifact_path("dl_model.h5")
    scaler_path = resolve_artifact_path("scaler.pkl")
    accuracy_path = resolve_artifact_path("model_accuracy.pkl")

    model = load_model(model_path)
    with scaler_path.open("rb") as scaler_file:
        scaler = pickle.load(scaler_file)
    with accuracy_path.open("rb") as accuracy_file:
        model_info = pickle.load(accuracy_file)

    return model, scaler, model_info.get("accuracy", 0) * 100


try:
    model, scaler, model_accuracy = load_artifacts()
    startup_error = None
    print("Model and scaler loaded successfully")
    print(f"Model Test Accuracy: {model_accuracy:.2f}%")
except (FileNotFoundError, OSError, ValueError) as exc:
    model = None
    scaler = None
    model_accuracy = 0
    startup_error = str(exc)
    print(f"ERROR: {startup_error}")
    print("Please run FLCP_DL/dl_model.py first to train the model.")


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

        data = np.array([values])
        data_scaled = scaler.transform(data)
        prediction_prob = float(model.predict(data_scaled, verbose=0)[0][0])

        prob_percentage = prediction_prob * 100
        risk_level = "High Risk" if prediction_prob > 0.5 else "Low Risk"
        risk_emoji = "High" if prediction_prob > 0.5 else "Low"

        return render_template(
            "index.html",
            features=features,
            prediction_text=f"{risk_emoji} - {risk_level}",
            probability=f"{prob_percentage:.1f}%",
            confidence=f"{(max(prediction_prob, 1 - prediction_prob) * 100):.1f}%",
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
