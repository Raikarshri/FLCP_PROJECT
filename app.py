import base64
import hashlib
import hmac
import importlib.util
import json
import os
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from functools import wraps
from pathlib import Path

from flask import Flask, g, make_response, redirect, render_template, request, url_for
from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, scoped_session, sessionmaker
from werkzeug.security import check_password_hash, generate_password_hash

from FLCP_DL.dl_app import predict_from_values
from FLCP_ML.models import FEATURE_NAMES, predict_structured


BASE_DIR = Path(__file__).resolve().parent
QML_RESULTS_PATH = BASE_DIR / "FLCP_QML" / "model_results.json"

FEATURE_METADATA = {
    "AGE": {"label": "Age", "type": "number", "min": 18, "max": 120, "step": "1"},
    "GENDER": {"label": "Gender", "type": "select", "options": [("1", "Male"), ("0", "Female")]},
    "SMOKING": {"label": "Smoking", "type": "boolean"},
    "FINGER_DISCOLORATION": {"label": "Finger Discoloration", "type": "boolean"},
    "MENTAL_STRESS": {"label": "Mental Stress", "type": "boolean"},
    "EXPOSURE_TO_POLLUTION": {"label": "Exposure To Pollution", "type": "boolean"},
    "LONG_TERM_ILLNESS": {"label": "Long Term Illness", "type": "boolean"},
    "ENERGY_LEVEL": {"label": "Energy Level", "type": "number", "step": "0.1"},
    "IMMUNE_WEAKNESS": {"label": "Immune Weakness", "type": "boolean"},
    "BREATHING_ISSUE": {"label": "Breathing Issue", "type": "boolean"},
    "ALCOHOL_CONSUMPTION": {"label": "Alcohol Consumption", "type": "boolean"},
    "THROAT_DISCOMFORT": {"label": "Throat Discomfort", "type": "boolean"},
    "OXYGEN_SATURATION": {"label": "Oxygen Saturation", "type": "number", "step": "0.1"},
    "CHEST_TIGHTNESS": {"label": "Chest Tightness", "type": "boolean"},
    "FAMILY_HISTORY": {"label": "Family History", "type": "boolean"},
    "SMOKING_FAMILY_HISTORY": {"label": "Smoking Family History", "type": "boolean"},
    "STRESS_IMMUNE": {"label": "Stress Immune", "type": "boolean"},
}

SessionLocal = scoped_session(sessionmaker())
ENGINE = None


class Base(DeclarativeBase):
    pass


def utc_now():
    return datetime.utcnow()


def normalize_datetime(value):
    if value is None:
        return None
    if value.tzinfo is not None:
        return value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    email: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    auth_tokens: Mapped[list["AuthToken"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    predictions: Mapped[list["Prediction"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class AuthToken(Base):
    __tablename__ = "auth_tokens"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    token_jti: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    issued_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    is_revoked: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    user: Mapped["User"] = relationship(back_populates="auth_tokens")


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    input_payload_json: Mapped[str] = mapped_column(Text, nullable=False)
    ml_predictions_json: Mapped[str] = mapped_column(Text, nullable=False)
    dl_prediction_json: Mapped[str] = mapped_column(Text, nullable=False)
    qml_metrics_json: Mapped[str] = mapped_column(Text, nullable=False)
    final_prediction: Mapped[str] = mapped_column(String(120), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    user: Mapped["User"] = relationship(back_populates="predictions")


def normalize_database_url(database_url):
    if database_url.startswith("postgres://"):
        return "postgresql://" + database_url[len("postgres://") :]
    return database_url


def resolve_database_url(default_database):
    return normalize_database_url(
        os.getenv("SUPABASE_DB_URL")
        or os.getenv("DATABASE_URL")
        or default_database
    )


def build_engine_options(database_url, app_config):
    engine_options = {
        "future": True,
        "pool_pre_ping": True,
    }
    if database_url.startswith("sqlite"):
        engine_options["connect_args"] = {"check_same_thread": False}
    elif database_url.startswith("postgresql"):
        connect_args = {}
        ssl_mode = app_config.get("DATABASE_SSLMODE")
        if ssl_mode:
            connect_args["sslmode"] = ssl_mode
        if connect_args:
            engine_options["connect_args"] = connect_args
        engine_options["pool_recycle"] = 1800
    return engine_options


def load_json(path):
    if not Path(path).exists():
        return None
    with Path(path).open("r", encoding="utf-8") as result_file:
        return json.load(result_file)


def to_percentage(value):
    if value is None:
        return None
    return round(float(value) * 100, 2)


def format_percentage(value):
    if value is None:
        return "N/A"
    return f"{float(value):.2f}%"


def format_prediction_label(value):
    return "High Risk" if int(value) == 1 else "Low Risk"


def json_dumps(value):
    return json.dumps(value, ensure_ascii=True)


def json_loads(value):
    return json.loads(value) if value else {}


def b64url_encode(value):
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def b64url_decode(value):
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def encode_jwt(payload, secret):
    header = {"alg": "HS256", "typ": "JWT"}
    header_segment = b64url_encode(json_dumps(header).encode("utf-8"))
    payload_segment = b64url_encode(json_dumps(payload).encode("utf-8"))
    signing_input = f"{header_segment}.{payload_segment}".encode("ascii")
    signature = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    return f"{header_segment}.{payload_segment}.{b64url_encode(signature)}"


def decode_jwt(token, secret):
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("Malformed token.")

    header_segment, payload_segment, signature_segment = parts
    signing_input = f"{header_segment}.{payload_segment}".encode("ascii")
    expected_signature = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    provided_signature = b64url_decode(signature_segment)

    if not hmac.compare_digest(expected_signature, provided_signature):
        raise ValueError("Invalid token signature.")

    payload = json.loads(b64url_decode(payload_segment))
    if "exp" not in payload:
        raise ValueError("Token missing expiry.")
    if datetime.now(timezone.utc).timestamp() >= float(payload["exp"]):
        raise ValueError("Token expired.")
    return payload


def get_feature_fields():
    fields = []
    for feature_name in FEATURE_NAMES:
        metadata = FEATURE_METADATA.get(feature_name, {"label": feature_name.replace("_", " ").title(), "type": "number"})
        field = {"name": feature_name, "label": metadata["label"], "type": metadata["type"]}
        if metadata["type"] == "boolean":
            field["options"] = [("1", "Yes"), ("0", "No")]
        elif metadata["type"] == "select":
            field["options"] = metadata["options"]
        else:
            field["step"] = metadata.get("step", "1")
            field["min"] = metadata.get("min")
            field["max"] = metadata.get("max")
        fields.append(field)
    return fields


def parse_feature_inputs(form):
    values = []
    input_payload = {}
    for feature_name in FEATURE_NAMES:
        raw_value = form.get(feature_name)
        if raw_value is None or str(raw_value).strip() == "":
            label = FEATURE_METADATA.get(feature_name, {}).get("label", feature_name)
            raise ValueError(f"{label} is required.")
        try:
            numeric_value = float(raw_value)
        except ValueError as exc:
            label = FEATURE_METADATA.get(feature_name, {}).get("label", feature_name)
            raise ValueError(f"{label} must be numeric.") from exc
        values.append(numeric_value)
        input_payload[feature_name] = numeric_value
    return values, input_payload


def get_db():
    return SessionLocal()


def init_database(app):
    global ENGINE
    database_url = resolve_database_url(app.config["DATABASE_URL"])
    engine_options = build_engine_options(database_url, app.config)
    ENGINE = create_engine(database_url, **engine_options)
    SessionLocal.configure(bind=ENGINE, autoflush=False, autocommit=False, expire_on_commit=False)
    Base.metadata.create_all(ENGINE)
    app.config["RESOLVED_DATABASE_URL"] = database_url


def reset_database():
    if ENGINE is None:
        return
    Base.metadata.drop_all(ENGINE)
    Base.metadata.create_all(ENGINE)


def issue_token(app, user, db_session):
    now = utc_now()
    expires_at = now + timedelta(hours=app.config["JWT_EXP_HOURS"])
    token_jti = uuid.uuid4().hex
    payload = {
        "sub": str(user.id),
        "jti": token_jti,
        "iat": int(now.timestamp()),
        "exp": int(expires_at.timestamp()),
    }
    token = encode_jwt(payload, app.config["JWT_SECRET"])
    db_session.add(
        AuthToken(
            user_id=user.id,
            token_jti=token_jti,
            issued_at=now,
            expires_at=expires_at,
            is_revoked=False,
        )
    )
    db_session.commit()
    return token


def set_auth_cookie(response, app, token):
    response.set_cookie(
        app.config["JWT_COOKIE_NAME"],
        token,
        httponly=True,
        secure=app.config["COOKIE_SECURE"],
        samesite="Lax",
        max_age=app.config["JWT_EXP_HOURS"] * 3600,
    )
    return response


def clear_auth_cookie(response, app):
    response.set_cookie(
        app.config["JWT_COOKIE_NAME"],
        "",
        expires=0,
        httponly=True,
        secure=app.config["COOKIE_SECURE"],
        samesite="Lax",
    )
    return response


def unauthorized_response(app):
    response = make_response(redirect(url_for("login")))
    return clear_auth_cookie(response, app)


def resolve_current_user(app):
    token = request.cookies.get(app.config["JWT_COOKIE_NAME"])
    if not token:
        return None

    try:
        payload = decode_jwt(token, app.config["JWT_SECRET"])
    except ValueError:
        return None

    db_session = get_db()
    auth_token = (
        db_session.query(AuthToken)
        .filter(AuthToken.token_jti == payload["jti"], AuthToken.is_revoked.is_(False))
        .first()
    )
    if auth_token is None or normalize_datetime(auth_token.expires_at) <= utc_now():
        return None

    user = db_session.query(User).filter(User.id == int(payload["sub"])).first()
    if user is None:
        return None

    g.auth_token_id = auth_token.id
    g.current_user = user
    return user


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        app = g.flask_app
        if resolve_current_user(app) is None:
            return unauthorized_response(app)
        return view(*args, **kwargs)

    return wrapped


def build_model_rows(ml_result, dl_result, qml_metrics):
    rows = [
        {
            "model": "KNN",
            "output": format_prediction_label(ml_result["predictions"]["knn"]),
            "accuracy": format_percentage(ml_result["metrics"]["knn"]),
            "notes": "Classical model prediction",
        },
        {
            "model": "Decision Tree",
            "output": format_prediction_label(ml_result["predictions"]["decision_tree"]),
            "accuracy": format_percentage(ml_result["metrics"]["decision_tree"]),
            "notes": "Classical model prediction",
        },
        {
            "model": "Random Forest",
            "output": format_prediction_label(ml_result["predictions"]["random_forest"]),
            "accuracy": format_percentage(ml_result["metrics"]["random_forest"]),
            "notes": "Best classical model is determined from saved metrics",
        },
    ]

    if dl_result["available"]:
        rows.append(
            {
                "model": "DNN",
                "output": dl_result["label"],
                "accuracy": format_percentage(dl_result["accuracy"]),
                "notes": (
                    f"Risk probability {dl_result['probability']:.1f}% "
                    f"with confidence {dl_result['confidence']:.1f}%"
                ),
            }
        )
    else:
        rows.append(
            {
                "model": "DNN",
                "output": "Unavailable",
                "accuracy": format_percentage(dl_result.get("accuracy")),
                "notes": dl_result["note"],
            }
        )

    qml_accuracy = to_percentage((qml_metrics or {}).get("quantum_accuracy"))
    classical_accuracy = to_percentage((qml_metrics or {}).get("classical_accuracy"))
    qml_note = "Comparison metric only"
    if classical_accuracy is not None:
        qml_note += f"; classical baseline {classical_accuracy:.2f}%"

    rows.append(
        {
            "model": "QML",
            "output": "Comparison only",
            "accuracy": format_percentage(qml_accuracy),
            "notes": qml_note,
        }
    )
    return rows


def build_dashboard_context(app, user, error_message=None, success_message=None, result_bundle=None):
    qml_metrics = load_json(QML_RESULTS_PATH) or {}
    history_records = (
        get_db().query(Prediction).filter(Prediction.user_id == user.id).order_by(Prediction.created_at.desc()).limit(5).all()
    )
    history = []
    for record in history_records:
        ml_payload = json_loads(record.ml_predictions_json)
        dl_payload = json_loads(record.dl_prediction_json)
        history.append(
            {
                "created_at": normalize_datetime(record.created_at).strftime("%Y-%m-%d %H:%M UTC"),
                "final_prediction": record.final_prediction,
                "random_forest": format_prediction_label(ml_payload["predictions"]["random_forest"]),
                "dnn_probability": (
                    f"{float(dl_payload['probability']):.1f}%"
                    if dl_payload.get("probability") is not None
                    else "N/A"
                ),
            }
        )

    qml_accuracy = to_percentage(qml_metrics.get("quantum_accuracy"))
    classical_accuracy = to_percentage(qml_metrics.get("classical_accuracy"))

    return {
        "page_title": "Prediction Dashboard",
        "current_user": user,
        "fields": get_feature_fields(),
        "error_message": error_message,
        "success_message": success_message,
        "result_bundle": result_bundle,
        "history": history,
        "qml_summary": {
            "quantum_accuracy": format_percentage(qml_accuracy),
            "classical_accuracy": format_percentage(classical_accuracy),
        },
        "has_tensorflow": importlib.util.find_spec("tensorflow") is not None,
    }


def create_prediction_record(user_id, input_payload, ml_result, dl_result, qml_metrics):
    return Prediction(
        user_id=user_id,
        input_payload_json=json_dumps(input_payload),
        ml_predictions_json=json_dumps(ml_result),
        dl_prediction_json=json_dumps(dl_result),
        qml_metrics_json=json_dumps(qml_metrics or {}),
        final_prediction=format_prediction_label(ml_result["final_prediction"]),
    )


def create_app(test_config=None):
    app = Flask(__name__)
    default_database = f"sqlite:///{BASE_DIR / 'flcp_app.db'}"
    app.config.update(
        SECRET_KEY=os.getenv("FLASK_SECRET_KEY", secrets.token_hex(16)),
        JWT_SECRET=os.getenv("JWT_SECRET_KEY", secrets.token_hex(32)),
        JWT_COOKIE_NAME="flcp_auth",
        JWT_EXP_HOURS=int(os.getenv("JWT_EXP_HOURS", "12")),
        COOKIE_SECURE=os.getenv("COOKIE_SECURE", "false").lower() == "true",
        DATABASE_URL=os.getenv("DATABASE_URL", default_database),
        DATABASE_SSLMODE=os.getenv("DATABASE_SSLMODE", "require"),
    )
    if test_config:
        app.config.update(test_config)

    init_database(app)

    @app.before_request
    def attach_app_context():
        g.flask_app = app
        g.current_user = None
        g.auth_token_id = None

    @app.teardown_appcontext
    def remove_session(exception=None):
        SessionLocal.remove()

    @app.route("/")
    def root():
        if resolve_current_user(app):
            return redirect(url_for("dashboard"))
        return redirect(url_for("login"))

    @app.route("/signup", methods=["GET", "POST"])
    def signup():
        if resolve_current_user(app):
            return redirect(url_for("dashboard"))

        if request.method == "POST":
            name = request.form.get("name", "").strip()
            email = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "")

            if not name or not email or not password:
                return render_template(
                    "signup.html",
                    page_title="Create Account",
                    error_message="Name, email, and password are required.",
                ), 400

            db_session = get_db()
            existing_user = db_session.query(User).filter(User.email == email).first()
            if existing_user:
                return render_template(
                    "signup.html",
                    page_title="Create Account",
                    error_message="An account with that email already exists.",
                ), 400

            user = User(name=name, email=email, password_hash=generate_password_hash(password))
            db_session.add(user)
            db_session.commit()
            db_session.refresh(user)

            token = issue_token(app, user, db_session)
            response = make_response(redirect(url_for("dashboard")))
            return set_auth_cookie(response, app, token)

        return render_template("signup.html", page_title="Create Account")

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if resolve_current_user(app):
            return redirect(url_for("dashboard"))

        if request.method == "POST":
            email = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "")
            db_session = get_db()
            user = db_session.query(User).filter(User.email == email).first()

            if user is None or not check_password_hash(user.password_hash, password):
                return render_template(
                    "login.html",
                    page_title="Sign In",
                    error_message="Invalid email or password.",
                ), 401

            token = issue_token(app, user, db_session)
            response = make_response(redirect(url_for("dashboard")))
            return set_auth_cookie(response, app, token)

        return render_template("login.html", page_title="Sign In")

    @app.route("/logout", methods=["POST"])
    @login_required
    def logout():
        db_session = get_db()
        auth_token = db_session.query(AuthToken).filter(AuthToken.id == g.auth_token_id).first()
        if auth_token is not None:
            auth_token.is_revoked = True
            db_session.commit()

        response = make_response(redirect(url_for("login")))
        return clear_auth_cookie(response, app)

    @app.route("/dashboard")
    @login_required
    def dashboard():
        return render_template("index.html", **build_dashboard_context(app, g.current_user))

    @app.route("/predict", methods=["POST"])
    @login_required
    def predict():
        try:
            values, input_payload = parse_feature_inputs(request.form)
        except ValueError as exc:
            return render_template(
                "index.html",
                **build_dashboard_context(app, g.current_user, error_message=str(exc)),
            ), 400

        ml_result = predict_structured(values)
        dl_result = predict_from_values(values)
        qml_metrics = load_json(QML_RESULTS_PATH) or {}
        result_bundle = {
            "rows": build_model_rows(ml_result, dl_result, qml_metrics),
            "final_prediction": format_prediction_label(ml_result["final_prediction"]),
            "dnn_probability": (
                f"{dl_result['probability']:.1f}%"
                if dl_result.get("probability") is not None
                else "N/A"
            ),
            "dnn_confidence": (
                f"{dl_result['confidence']:.1f}%"
                if dl_result.get("confidence") is not None
                else "N/A"
            ),
        }

        db_session = get_db()
        db_session.add(
            create_prediction_record(
                g.current_user.id,
                input_payload,
                ml_result,
                dl_result,
                qml_metrics,
            )
        )
        db_session.commit()

        return render_template(
            "index.html",
            **build_dashboard_context(
                app,
                g.current_user,
                success_message="Prediction completed and saved to your recent history.",
                result_bundle=result_bundle,
            ),
        )

    return app


def close_database():
    SessionLocal.remove()
    if ENGINE is not None:
        ENGINE.dispose()


app = create_app()


if __name__ == "__main__":
    app.run(debug=True, port=5000)
