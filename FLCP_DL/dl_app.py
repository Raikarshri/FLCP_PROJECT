from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import os
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Load model, scaler, and accuracy with error handling
try:
    model = load_model("dl_model.h5")
    scaler = pickle.load(open("scaler.pkl", "rb"))
    model_info = pickle.load(open("model_accuracy.pkl", "rb"))
    model_accuracy = model_info.get("accuracy", 0) * 100
    print(f"✓ Model and scaler loaded successfully")
    print(f"✓ Model Test Accuracy: {model_accuracy:.2f}%")
except FileNotFoundError as e:
    print(f"ERROR: {e}")
    print("Please run dl_model.py first to train the model")
    model = None
    scaler = None
    model_accuracy = 0

features = [
    'AGE', 'GENDER', 'SMOKING', 'FINGER_DISCOLORATION', 'MENTAL_STRESS',
    'EXPOSURE_TO_POLLUTION', 'LONG_TERM_ILLNESS', 'ENERGY_LEVEL',
    'IMMUNE_WEAKNESS', 'BREATHING_ISSUE', 'ALCOHOL_CONSUMPTION',
    'THROAT_DISCOMFORT', 'OXYGEN_SATURATION', 'CHEST_TIGHTNESS',
    'FAMILY_HISTORY', 'SMOKING_FAMILY_HISTORY', 'STRESS_IMMUNE'
]

@app.route("/")
def home():
    return render_template("index.html", features=features, model_accuracy=f"{model_accuracy:.2f}%")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None or scaler is None:
        return "Error: Model not loaded. Please train the model first.", 500
    
    try:
        values = []
        
        # Collect and validate input values
        for f in features:
            val = request.form.get(f)
            
            if val is None or val == "":
                return f"❌ Error: Please select a value for {f}", 400
            
            try:
                values.append(float(val))
            except ValueError:
                return f"❌ Error: Invalid value for {f}", 400
        
        # Make prediction
        data = np.array([values])
        data_scaled = scaler.transform(data)
        prediction_prob = model.predict(data_scaled, verbose=0)[0][0]
        
        # Generate result
        prob_percentage = prediction_prob * 100
        risk_level = "High Risk" if prediction_prob > 0.5 else "Low Risk"
        risk_emoji = "🔴" if prediction_prob > 0.5 else "🟢"
        
        return render_template(
            "index.html",
            features=features,
            prediction_text=f"{risk_emoji} {risk_level}",
            probability=f"{prob_percentage:.1f}%",
            confidence=f"{(max(prediction_prob, 1-prediction_prob)*100):.1f}%",
            model_accuracy=f"{model_accuracy:.2f}%"
        )
    
    except Exception as e:
        return f"❌ Prediction Error: {str(e)}", 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)