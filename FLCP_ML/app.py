from flask import Flask, render_template, request
from models import predict_all

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.form.values()
    input_data = [int(x) for x in data]

    k, d, r, final, knn_acc, dt_acc, rf_acc = predict_all(input_data)

    return render_template("index.html",
                               knn=k, dt=d, rf=r, final=final,
                               knn_acc=round(knn_acc,2),
                               dt_acc=round(dt_acc,2),
                               rf_acc=round(rf_acc,2)
                            )
if __name__ == "__main__":
    app.run(debug=True)

import json

try:
    accuracy = classical_accuracy  # or your actual variable
except:
    accuracy = 0

with open("FLCP_ML/ml_results.json", "w") as f:
    json.dump({"accuracy": accuracy}, f)

print(" ML results saved")