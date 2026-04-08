from flask import Flask, render_template, request

if __package__:
    from .models import FEATURE_NAMES, predict_all
else:
    from models import FEATURE_NAMES, predict_all


app = Flask(__name__)


def parse_input(form):
    raw_values = list(form.values())
    if len(raw_values) != len(FEATURE_NAMES):
        raise ValueError(f"Expected {len(FEATURE_NAMES)} values, received {len(raw_values)}.")

    values = []
    for index, raw in enumerate(raw_values, start=1):
        if raw is None or str(raw).strip() == "":
            raise ValueError(f"Value {index} is required.")
        try:
            values.append(float(raw))
        except ValueError as exc:
            raise ValueError(f"Value {index} must be numeric.") from exc
    return values


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = parse_input(request.form)
        k, d, r, final, knn_acc, dt_acc, rf_acc = predict_all(input_data)
    except ValueError as exc:
        return render_template("index.html", error_message=str(exc)), 400

    return render_template(
        "index.html",
        knn=k,
        dt=d,
        rf=r,
        final=final,
        knn_acc=round(knn_acc, 2),
        dt_acc=round(dt_acc, 2),
        rf_acc=round(rf_acc, 2),
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
