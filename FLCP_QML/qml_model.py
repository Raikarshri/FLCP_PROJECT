import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import json

# QML
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit_algorithms.optimizers import COBYLA


df = pd.read_csv("DATA/lungcancer_clean.csv")

df.columns = df.columns.str.strip()
df.columns = df.columns.str.replace(" ", "_")

df = df.replace({
    "YES": 1, "NO": 0,
    "Yes": 1, "No": 0,
    "M": 1, "F": 0
})

df = df.apply(pd.to_numeric)




X = df.drop("LUNG_CANCER", axis=1).values
y = df["LUNG_CANCER"].values

# Scale
X = MinMaxScaler().fit_transform(X)

# Reduce features
X = PCA(n_components=4).fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# CLASSICAL MODEL

svc = SVC()
svc.fit(X_train, y_train)

classical_accuracy = svc.score(X_test, y_test)
print("Classical Accuracy:", classical_accuracy)


# QML MODEL

num_features = X.shape[1]   # ✅ now it works

feature_map = ZZFeatureMap(feature_dimension=num_features, reps=1)
ansatz = RealAmplitudes(num_qubits=num_features, reps=2)

optimizer = COBYLA(maxiter=20)

vqc = VQC(
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer
)

print("Training Quantum Model... ⏳")
vqc.fit(X_train, y_train)

quantum_accuracy = vqc.score(X_test, y_test)
print("Quantum Accuracy:", quantum_accuracy)


# SAVE RESULTS

results = {
    "classical_accuracy": classical_accuracy,
    "quantum_accuracy": quantum_accuracy,
    "num_features": num_features,
    "test_size": len(X_test),
    "total_samples": len(X)
}

with open("model_results.json", "w") as f:
    json.dump(results, f)

print("✅ Results saved to model_results.json")


# GENERATE REPORT

import subprocess
import os

# Run the report generator
subprocess.run(["python", "generate_report.py"])

# Open the webpage
html_path = os.path.abspath("results.html")
print(f"\n🎉 Training complete! Open your results at:")
print(f"file://{html_path}")
print("Or simply double-click results.html in your file explorer.")