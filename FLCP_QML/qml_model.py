from pathlib import Path
import json

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_algorithms.optimizers import COBYLA
from qiskit_machine_learning.algorithms.classifiers import VQC

if __package__:
    from .generate_report import REPORT_PATH, RESULTS_PATH, generate_html_report
else:
    from generate_report import REPORT_PATH, RESULTS_PATH, generate_html_report


mlflow.set_experiment("QML Experiment")

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "DATA" / "lungcancer_clean.csv"
RANDOM_STATE = 42


df = pd.read_csv(DATASET_PATH)
df.columns = df.columns.str.strip().str.replace(" ", "_")
df = df.replace({
    "YES": 1,
    "NO": 0,
    "Yes": 1,
    "No": 0,
    "M": 1,
    "F": 0,
})
df = df.apply(pd.to_numeric)

x = df.drop("LUNG_CANCER", axis=1).values
y = df["LUNG_CANCER"].values

x = MinMaxScaler().fit_transform(x)
x = PCA(n_components=4, random_state=RANDOM_STATE).fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y,
)

with mlflow.start_run(run_name="QML_Classical_Model"):
    svc = SVC()
    svc.fit(x_train, y_train)

    classical_accuracy = svc.score(x_test, y_test)
    print("Classical Accuracy:", classical_accuracy)
    mlflow.log_metric("classical_accuracy", classical_accuracy)
    mlflow.sklearn.log_model(svc, name="classical_model")


num_features = x.shape[1]
feature_map = ZZFeatureMap(feature_dimension=num_features, reps=1)
ansatz = RealAmplitudes(num_qubits=num_features, reps=2)
optimizer = COBYLA(maxiter=20)

with mlflow.start_run(run_name="QML_Quantum_Model"):
    vqc = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
    )

    print("Training Quantum Model...")
    vqc.fit(x_train, y_train)

    quantum_accuracy = vqc.score(x_test, y_test)
    print("Quantum Accuracy:", quantum_accuracy)

    mlflow.log_metric("quantum_accuracy", quantum_accuracy)
    mlflow.log_param("num_features", num_features)
    mlflow.log_param("feature_map_reps", 1)
    mlflow.log_param("ansatz_reps", 2)
    mlflow.log_param("optimizer_maxiter", 20)


results = {
    "classical_accuracy": classical_accuracy,
    "quantum_accuracy": quantum_accuracy,
    "num_features": num_features,
    "test_size": len(x_test),
    "total_samples": len(x),
}

with RESULTS_PATH.open("w", encoding="utf-8") as results_file:
    json.dump(results, results_file)

print(f"Results saved to {RESULTS_PATH}")

generate_html_report()
print("\nTraining complete. Open your results at:")
print(REPORT_PATH.resolve())
