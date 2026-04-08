import importlib.util
import tempfile
import unittest
from unittest import mock
from pathlib import Path


class DashboardAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import app as dashboard_app

        cls.dashboard_app = dashboard_app
        dashboard_app.app.testing = True
        cls.client = dashboard_app.app.test_client()

    def test_dashboard_renders_with_missing_files(self):
        with mock.patch.object(self.dashboard_app, "ML_RESULTS_PATH", Path("missing_ml.json")), \
             mock.patch.object(self.dashboard_app, "DL_RESULTS_PATH", Path("missing_dl.json")), \
             mock.patch.object(self.dashboard_app, "QML_RESULTS_PATH", Path("missing_qml.json")):
            response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Lung Cancer Risk Prediction", response.data)
        self.assertIn(b"No comparison metrics are available yet", response.data)

    def test_dashboard_renders_all_sections_from_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            ml_path = tmp_path / "ml_results.json"
            dl_path = tmp_path / "dl_results.json"
            qml_path = tmp_path / "model_results.json"

            ml_path.write_text(
                (
                    '{"knn_accuracy": 0.81, "decision_tree_accuracy": 0.77, '
                    '"random_forest_accuracy": 0.88, "best_model": "Random Forest", '
                    '"best_accuracy": 0.88}'
                ),
                encoding="utf-8",
            )
            dl_path.write_text('{"accuracy": 0.91}', encoding="utf-8")
            qml_path.write_text(
                (
                    '{"classical_accuracy": 0.85, "quantum_accuracy": 0.72, '
                    '"num_features": 4, "test_size": 50, "total_samples": 250}'
                ),
                encoding="utf-8",
            )

            with mock.patch.object(self.dashboard_app, "ML_RESULTS_PATH", ml_path), \
                 mock.patch.object(self.dashboard_app, "DL_RESULTS_PATH", dl_path), \
                 mock.patch.object(self.dashboard_app, "QML_RESULTS_PATH", qml_path):
                response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"KNN", response.data)
        self.assertIn(b"Decision Tree", response.data)
        self.assertIn(b"Random Forest", response.data)
        self.assertIn(b"DNN", response.data)
        self.assertIn(b"QML", response.data)
        self.assertIn(b"91.00%", response.data)
        self.assertIn(b"Conclusion", response.data)


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
            self.assertIn("Classical SVM vs Quantum Machine Learning Comparison", report_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
