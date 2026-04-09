import importlib.util
import os
import tempfile
import unittest
from pathlib import Path


class AuthenticatedAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import app as dashboard_module

        cls.dashboard_module = dashboard_module

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        database_path = Path(self.tempdir.name) / "test_app.db"
        self.app = self.dashboard_module.create_app(
            {
                "TESTING": True,
                "SECRET_KEY": "test-secret",
                "JWT_SECRET": "jwt-secret",
                "DATABASE_URL": f"sqlite:///{database_path}",
                "COOKIE_SECURE": False,
            }
        )
        self.dashboard_module.reset_database()
        self.client = self.app.test_client()

    def tearDown(self):
        self.dashboard_module.close_database()
        self.tempdir.cleanup()

    def signup(self, email="demo@example.com", password="pass123", name="Demo User"):
        return self.client.post(
            "/signup",
            data={"name": name, "email": email, "password": password},
            follow_redirects=False,
        )

    def login(self, email="demo@example.com", password="pass123"):
        return self.client.post(
            "/login",
            data={"email": email, "password": password},
            follow_redirects=False,
        )

    def prediction_payload(self):
        return {
            "AGE": "55",
            "GENDER": "1",
            "SMOKING": "1",
            "FINGER_DISCOLORATION": "0",
            "MENTAL_STRESS": "1",
            "EXPOSURE_TO_POLLUTION": "1",
            "LONG_TERM_ILLNESS": "0",
            "ENERGY_LEVEL": "63.5",
            "IMMUNE_WEAKNESS": "0",
            "BREATHING_ISSUE": "1",
            "ALCOHOL_CONSUMPTION": "0",
            "THROAT_DISCOMFORT": "1",
            "OXYGEN_SATURATION": "95.4",
            "CHEST_TIGHTNESS": "0",
            "FAMILY_HISTORY": "1",
            "SMOKING_FAMILY_HISTORY": "0",
            "STRESS_IMMUNE": "1",
        }

    def test_signup_creates_user_and_redirects_to_dashboard(self):
        response = self.signup()

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["Location"], "/dashboard")
        self.assertIn("flcp_auth=", response.headers.get("Set-Cookie", ""))

        with self.app.app_context():
            db_session = self.dashboard_module.get_db()
            self.assertEqual(db_session.query(self.dashboard_module.User).count(), 1)
            self.assertEqual(db_session.query(self.dashboard_module.AuthToken).count(), 1)

    def test_duplicate_email_is_rejected(self):
        self.signup()
        self.client = self.app.test_client()
        response = self.signup()

        self.assertEqual(response.status_code, 400)
        self.assertIn(b"already exists", response.data)

    def test_login_sets_jwt_cookie(self):
        self.signup()
        self.client = self.app.test_client()

        response = self.login()

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["Location"], "/dashboard")
        self.assertIn("flcp_auth=", response.headers.get("Set-Cookie", ""))

    def test_invalid_login_shows_error(self):
        self.signup()
        self.client = self.app.test_client()

        response = self.login(password="wrong-password")

        self.assertEqual(response.status_code, 401)
        self.assertIn(b"Invalid email or password", response.data)

    def test_dashboard_redirects_when_not_authenticated(self):
        response = self.client.get("/dashboard", follow_redirects=False)

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["Location"], "/login")

    def test_prediction_persists_and_shows_all_model_rows(self):
        self.signup()

        response = self.client.post("/predict", data=self.prediction_payload(), follow_redirects=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"KNN", response.data)
        self.assertIn(b"Decision Tree", response.data)
        self.assertIn(b"Random Forest", response.data)
        self.assertIn(b"DNN", response.data)
        self.assertIn(b"QML", response.data)
        self.assertIn(b"Prediction completed and saved", response.data)

        with self.app.app_context():
            db_session = self.dashboard_module.get_db()
            prediction = db_session.query(self.dashboard_module.Prediction).one()
            self.assertEqual(prediction.user_id, 1)

    def test_prediction_validation_rejects_missing_values(self):
        self.signup()
        payload = self.prediction_payload()
        payload.pop("AGE")

        response = self.client.post("/predict", data=payload)

        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Age is required", response.data)

    def test_logout_revokes_token_and_redirects(self):
        self.signup()
        response = self.client.post("/logout", follow_redirects=False)

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["Location"], "/login")

        with self.app.app_context():
            db_session = self.dashboard_module.get_db()
            token = db_session.query(self.dashboard_module.AuthToken).one()
            self.assertTrue(token.is_revoked)

    def test_dashboard_no_longer_contains_lecturer_text(self):
        self.signup()
        response = self.client.get("/dashboard")

        self.assertEqual(response.status_code, 200)
        self.assertNotIn(b"Lecturer", response.data)

    def test_history_only_shows_current_users_records(self):
        self.signup(email="user1@example.com")
        self.client.post("/predict", data=self.prediction_payload(), follow_redirects=True)

        second_client = self.app.test_client()
        second_client.post(
            "/signup",
            data={"name": "User Two", "email": "user2@example.com", "password": "pass456"},
            follow_redirects=True,
        )
        response = second_client.get("/dashboard")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"saved predictions will appear here", response.data)

    def test_supabase_database_url_takes_precedence_when_provided(self):
        default_database = "sqlite:///fallback.db"
        previous_supabase = os.environ.get("SUPABASE_DB_URL")
        previous_database = os.environ.get("DATABASE_URL")
        try:
            os.environ["SUPABASE_DB_URL"] = "postgres://supabase.example/project"
            os.environ["DATABASE_URL"] = "sqlite:///ignored.db"
            resolved = self.dashboard_module.resolve_database_url(default_database)
        finally:
            if previous_supabase is None:
                os.environ.pop("SUPABASE_DB_URL", None)
            else:
                os.environ["SUPABASE_DB_URL"] = previous_supabase
            if previous_database is None:
                os.environ.pop("DATABASE_URL", None)
            else:
                os.environ["DATABASE_URL"] = previous_database

        self.assertEqual(resolved, "postgresql://supabase.example/project")

    def test_postgres_engine_options_include_sslmode(self):
        options = self.dashboard_module.build_engine_options(
            "postgresql://db.example/flcp",
            {"DATABASE_SSLMODE": "require"},
        )

        self.assertEqual(options["connect_args"]["sslmode"], "require")
        self.assertTrue(options["pool_pre_ping"])


class MLAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec("sklearn") is None:
            raise unittest.SkipTest("scikit-learn is not installed in this environment.")

        from FLCP_ML.app import app
        from FLCP_ML.models import FEATURE_NAMES

        cls.app = app
        cls.app.testing = True
        cls.client = app.test_client()
        cls.feature_count = len(FEATURE_NAMES)

    def test_parse_accepts_decimal_inputs(self):
        from FLCP_ML.app import parse_input

        form_data = {f"f{i}": str(i + 0.5) for i in range(1, self.feature_count + 1)}
        parsed = parse_input(form_data)

        self.assertEqual(len(parsed), self.feature_count)
        self.assertTrue(all(isinstance(value, float) for value in parsed))

    def test_predict_route_handles_decimal_inputs(self):
        payload = {
            "f1": "55",
            "f2": "1",
            "f3": "1",
            "f4": "0",
            "f5": "1",
            "f6": "1",
            "f7": "0",
            "f8": "63.5",
            "f9": "0",
            "f10": "1",
            "f11": "0",
            "f12": "1",
            "f13": "95.4",
            "f14": "0",
            "f15": "1",
            "f16": "0",
            "f17": "1",
        }

        response = self.client.post("/predict", data=payload)

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Final Prediction", response.data)

    def test_predict_route_returns_400_for_missing_values(self):
        response = self.client.post("/predict", data={"f1": "55"})

        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Expected", response.data)


class DLAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec("tensorflow") is None:
            raise unittest.SkipTest("tensorflow is not installed in this environment.")

        from FLCP_DL import dl_app

        cls.dl_app = dl_app

    def test_resolve_artifact_path_finds_existing_model(self):
        path = self.dl_app.resolve_artifact_path("dl_model.h5")
        self.assertTrue(path.exists())

    def test_resolve_artifact_path_raises_for_missing_artifact(self):
        with self.assertRaises(FileNotFoundError):
            self.dl_app.resolve_artifact_path("missing-artifact.bin")


class QMLReportTests(unittest.TestCase):
    def test_generate_report_uses_requested_paths(self):
        from FLCP_QML.generate_report import generate_html_report

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            results_path = tmp_path / "model_results.json"
            report_path = tmp_path / "results.html"
            results_path.write_text(
                (
                    '{"classical_accuracy": 0.8, "quantum_accuracy": 0.7, '
                    '"num_features": 4, "test_size": 50, "total_samples": 250}'
                ),
                encoding="utf-8",
            )

            generate_html_report(results_path=results_path, report_path=report_path)

            self.assertTrue(report_path.exists())
            self.assertIn(
                "Classical SVM vs Quantum Machine Learning Comparison",
                report_path.read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
