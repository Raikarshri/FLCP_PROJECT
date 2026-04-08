# FLCP Project

## Overview

FLCP Project is a lung cancer prediction and model comparison system built around a shared structured healthcare dataset. The project studies the same prediction problem through three separate approaches:

- Classical Machine Learning
- Deep Learning
- Quantum Machine Learning

The central idea is to use patient-style clinical and lifestyle attributes such as age, smoking status, oxygen saturation, breathing issues, family history, and related health indicators to predict the likelihood of lung cancer. The repository is organized as a multi-module academic/research-style project rather than a single production application.

## Project Purpose

This project has two main purposes:

1. Build predictive models for lung cancer classification from tabular health data.
2. Compare how different AI approaches perform on the same dataset and problem statement.

The project is useful for:

- understanding end-to-end ML pipeline design
- comparing classical, neural-network, and quantum models
- learning how model training, prediction, experiment tracking, and reporting connect in one repository

## Repository Structure

```text
FLCP_PROJECT/
|-- DATA/
|   |-- lungcancer_clean.csv
|   `-- lungcancer_clean.csv.dvc
|-- FLCP_ML/
|   |-- app.py
|   |-- models.py
|   |-- train_ml.py
|   |-- ML/
|   |   |-- __init__.py
|   |   |-- knn.py
|   |   |-- decision_tree.py
|   |   `-- random_forest.py
|   |-- templates/
|   |   `-- index.html
|   `-- Theory/
|-- FLCP_DL/
|   |-- dl_app.py
|   |-- dl_model.py
|   |-- train_dl.py
|   |-- dl_model.h5
|   |-- scaler.pkl
|   |-- model_accuracy.pkl
|   |-- dl_results.json
|   `-- templates/
|       `-- index.html
|-- FLCP_QML/
|   |-- qml_model.py
|   |-- generate_report.py
|   |-- test_model.py
|   `-- results.html
|-- dvc.yaml
|-- dvc.lock
|-- mlflow.db
|-- app.py
|-- templates/
|   `-- index.html
|-- dl_model.h5
|-- scaler.pkl
|-- model_accuracy.pkl
`-- results.html
```

## Core Technologies

### Programming Language

- Python

### Backend and Web Framework

- Flask

### Data Processing

- Pandas
- NumPy

### Classical Machine Learning

- scikit-learn
  - KNeighborsClassifier
  - DecisionTreeClassifier
  - RandomForestClassifier
  - SVC
  - PCA
  - StandardScaler
  - MinMaxScaler
  - train_test_split

### Deep Learning

- TensorFlow
- Keras

### Quantum Machine Learning

- Qiskit
- qiskit-machine-learning
- qiskit-algorithms

### Experiment and Pipeline Management

- MLflow
- DVC

### Frontend

- HTML
- CSS
- Bootstrap
- Font Awesome

## Dataset

The shared dataset used by all modules is:

- [`DATA/lungcancer_clean.csv`](D:/sri/FLCP_PROJECT/DATA/lungcancer_clean.csv)

### Dataset Characteristics

- 5000 rows
- 18 columns total
- 17 input features
- 1 target column

### Main Input Features

- AGE
- GENDER
- SMOKING
- FINGER_DISCOLORATION
- MENTAL_STRESS
- EXPOSURE_TO_POLLUTION
- LONG_TERM_ILLNESS
- ENERGY_LEVEL
- IMMUNE_WEAKNESS
- BREATHING_ISSUE
- ALCOHOL_CONSUMPTION
- THROAT_DISCOMFORT
- OXYGEN_SATURATION
- CHEST_TIGHTNESS
- FAMILY_HISTORY
- SMOKING_FAMILY_HISTORY
- STRESS_IMMUNE

### Target

- LUNG_CANCER

All three modeling tracks load this dataset, clean column names, convert categorical values into numeric form, split the data into training and testing subsets, and then train their respective models.

## High-Level Architecture

```text
                          +---------------------------+
                          |   DATA/lungcancer_clean   |
                          +------------+--------------+
                                       |
               +-----------------------+-----------------------+
               |                       |                       |
               v                       v                       v
      +----------------+     +----------------+     +-------------------+
      |    FLCP_ML     |     |    FLCP_DL     |     |     FLCP_QML      |
      | Classical ML   |     | Deep Learning  |     | Quantum vs        |
      | Web App        |     | Web App        |     | Classical Compare |
      +-------+--------+     +--------+-------+     +---------+---------+
              |                       |                       |
              v                       v                       v
     +------------------+    +-------------------+    +-------------------+
     | KNN / DT / RF    |    | Keras NN Model    |    | SVM + VQC Model   |
     | Ensemble Output  |    | Probability Score |    | Accuracy Report   |
     +------------------+    +-------------------+    +-------------------+
              |                       |                       |
              +-----------+-----------+-----------+-----------+
                          |                       |
                          v                       v
                 +----------------+      +------------------+
                 |    MLflow      |      |       DVC        |
                 | Experiment Log |      | Pipeline Stages  |
                 +----------------+      +------------------+
```

## Detailed Module Explanation

## 1. FLCP_ML: Classical Machine Learning Module

### Purpose

This module builds a classical machine learning prediction system and exposes it through a Flask web application.

### Main Files

- [`FLCP_ML/app.py`](D:/sri/FLCP_PROJECT/FLCP_ML/app.py)
- [`FLCP_ML/models.py`](D:/sri/FLCP_PROJECT/FLCP_ML/models.py)
- [`FLCP_ML/train_ml.py`](D:/sri/FLCP_PROJECT/FLCP_ML/train_ml.py)
- [`FLCP_ML/ML/knn.py`](D:/sri/FLCP_PROJECT/FLCP_ML/ML/knn.py)
- [`FLCP_ML/ML/decision_tree.py`](D:/sri/FLCP_PROJECT/FLCP_ML/ML/decision_tree.py)
- [`FLCP_ML/ML/random_forest.py`](D:/sri/FLCP_PROJECT/FLCP_ML/ML/random_forest.py)
- [`FLCP_ML/templates/index.html`](D:/sri/FLCP_PROJECT/FLCP_ML/templates/index.html)

### Implementation Flow

1. Load the dataset from `DATA/lungcancer_clean.csv`.
2. Clean column names by trimming spaces and replacing spaces with underscores.
3. Encode or convert feature values into numeric format.
4. Split the data into training and testing sets.
5. Train three separate classical models:
   - KNN
   - Decision Tree
   - Random Forest
6. Evaluate each model separately.
7. Accept user input through a Flask form.
8. Run all three models on the same input.
9. Produce a final prediction by averaging the three model outputs and rounding the result.

### Functionalities Included

- user form for entering 17 patient attributes
- individual prediction from each classical model
- display of model-wise accuracy
- final ensemble prediction

### Internal Connectivity

- `app.py` handles HTTP routes and form submission.
- `app.py` imports `predict_all()` from `models.py`.
- `models.py` loads data, trains models, evaluates them, and serves inference.
- `models.py` imports the individual training helpers from the `ML/` package.
- `templates/index.html` renders the input form and prediction results.

### Important Design Note

The current implementation in `models.py` performs training at import time. That means when the Flask app starts and imports `models.py`, model training also happens immediately. This is acceptable for a student project, but in a production-grade system training and serving would normally be separated.

## 2. FLCP_DL: Deep Learning Module

### Purpose

This module trains a neural network for lung cancer prediction and provides a more polished web-based risk assessment interface.

### Main Files

- [`FLCP_DL/dl_app.py`](D:/sri/FLCP_PROJECT/FLCP_DL/dl_app.py)
- [`FLCP_DL/dl_model.py`](D:/sri/FLCP_PROJECT/FLCP_DL/dl_model.py)
- [`FLCP_DL/train_dl.py`](D:/sri/FLCP_PROJECT/FLCP_DL/train_dl.py)
- [`FLCP_DL/templates/index.html`](D:/sri/FLCP_PROJECT/FLCP_DL/templates/index.html)

### Implementation Flow

There are two deep-learning-related training scripts in the project:

- `train_dl.py` is the simpler DVC and MLflow pipeline script.
- `dl_model.py` is the more complete script that trains and saves the model artifacts used by the Flask app.

### `dl_model.py` Workflow

1. Load the dataset.
2. Clean and normalize column names.
3. Convert all values to numeric format.
4. Split data into:
   - training set
   - validation set
   - test set
5. Scale features with `StandardScaler`.
6. Build a multi-layer neural network using Keras.
7. Apply dropout to reduce overfitting.
8. Train the model with early stopping.
9. Evaluate model accuracy on the test set.
10. Save:
    - `FLCP_DL/dl_model.h5`
    - `FLCP_DL/scaler.pkl`
    - `FLCP_DL/model_accuracy.pkl`

### `dl_app.py` Workflow

1. Load the trained model and preprocessing artifacts.
2. Render a structured HTML form.
3. Validate each submitted field.
4. Scale user input using the saved scaler.
5. Predict probability using the trained neural network.
6. Convert probability into:
   - risk label
   - risk probability
   - confidence level
7. Return the result to the user interface.

### Functionalities Included

- structured health-risk form
- field validation
- deep learning probability-based prediction
- low-risk or high-risk categorization
- displayed model accuracy
- confidence output for the prediction

### Internal Connectivity

- `dl_model.py` produces the saved model files.
- `dl_app.py` consumes those saved files for inference.
- `templates/index.html` renders the user-facing interface.
- `train_dl.py` logs the deep learning experiment through MLflow and writes `dl_results.json`.

## 3. FLCP_QML: Quantum Machine Learning Module

### Purpose

This module compares a classical Support Vector Machine against a quantum Variational Quantum Classifier.

### Main Files

- [`FLCP_QML/qml_model.py`](D:/sri/FLCP_PROJECT/FLCP_QML/qml_model.py)
- [`FLCP_QML/generate_report.py`](D:/sri/FLCP_PROJECT/FLCP_QML/generate_report.py)
- [`FLCP_QML/test_model.py`](D:/sri/FLCP_PROJECT/FLCP_QML/test_model.py)

### Implementation Flow

1. Load the dataset.
2. Clean and convert all values to numeric format.
3. Scale the features using `MinMaxScaler`.
4. Reduce dimensionality to 4 features using PCA.
5. Split data into train and test sets.
6. Train a classical SVM baseline.
7. Log the classical baseline in MLflow.
8. Configure the quantum model:
   - `ZZFeatureMap`
   - `RealAmplitudes`
   - `COBYLA` optimizer
9. Train the Variational Quantum Classifier.
10. Measure the quantum model accuracy.
11. Save the comparison results to `model_results.json`.
12. Generate an HTML report using `generate_report.py`.

### Functionalities Included

- classical SVM baseline
- quantum classifier training
- classical vs quantum accuracy comparison
- generated HTML comparison report

### Internal Connectivity

- `qml_model.py` performs training and stores result metrics.
- `generate_report.py` reads those result metrics and builds the report page.
- `test_model.py` provides a lightweight testing path for report generation.

## End-to-End Workflow

The repository follows this general workflow:

1. The common dataset is stored in `DATA/`.
2. Each modeling track reads the same dataset independently.
3. Each track applies its own preprocessing and model training logic.
4. MLflow is used to log experiment metrics.
5. DVC can be used to orchestrate the three tracked pipeline stages.
6. The ML and DL modules expose prediction interfaces.
7. The QML module generates a static comparative report.

## DVC Pipeline

The file [`dvc.yaml`](D:/sri/FLCP_PROJECT/dvc.yaml) defines three pipeline stages:

- `ml`
- `dl`
- `qml`

### Stage Outputs

- `FLCP_ML/ml_results.json`
- `FLCP_DL/dl_results.json`
- `FLCP_QML/model_results.json`

This shows that the project was designed to treat each modeling track as a reproducible pipeline step.

## MLflow Usage

MLflow is used in:

- [`FLCP_ML/train_ml.py`](D:/sri/FLCP_PROJECT/FLCP_ML/train_ml.py)
- [`FLCP_DL/train_dl.py`](D:/sri/FLCP_PROJECT/FLCP_DL/train_dl.py)
- [`FLCP_QML/qml_model.py`](D:/sri/FLCP_PROJECT/FLCP_QML/qml_model.py)

It stores experiment runs, metrics, and some model artifacts. The repository also contains `mlflow.db`, which indicates local tracking data has already been created.

## How the Different Parts Are Connected

### Data Connection

All modules depend on the same dataset. This is the strongest connection in the repository.

### Training Connection

Each module independently trains a model on the same problem statement:

- ML uses multiple scikit-learn classifiers
- DL uses a neural network
- QML uses SVM and VQC

### Serving Connection

- `FLCP_ML` serves classical ensemble predictions through Flask.
- `FLCP_DL` serves neural-network predictions through Flask.
- `FLCP_QML` does not serve a live web app; it creates a static HTML result page.

### Tracking Connection

- DVC connects all training stages at the pipeline level.
- MLflow connects all training stages at the experiment-tracking level.

## How to Run the Project

## Complete Setup and Run Guide

This section explains how to set up the project from GitHub and run it end to end on a Windows machine using PowerShell.

## Recommended Python Version

Use **Python 3.13** or **Python 3.12** for the best compatibility.

Reason:

- `tensorflow` support is not reliable on Python `3.14`
- the project depends on `tensorflow`, `scikit-learn`, `mlflow`, `dvc`, and `qiskit`

## 1. Clone the Repository

If the project is on GitHub, clone it first:

```powershell
cd D:\
git clone <your-github-repo-url>
cd sri
cd FLCP_PROJECT
```

If you already have the folder on your system, just navigate into it:

```powershell
cd D:\sri\FLCP_PROJECT
```

## 2. Verify That You Are in the Project Root

You should be inside the folder that contains:

- `app.py`
- `dvc.yaml`
- `README.md`
- `FLCP_ML`
- `FLCP_DL`
- `FLCP_QML`

You can check with:

```powershell
Get-ChildItem
```

## 3. Create a Virtual Environment

From the project root:

```powershell
cd D:\sri\FLCP_PROJECT
python -m venv .venv
```

If you installed Python 3.13 separately, use:

```powershell
py -3.13 -m venv .venv
```

## 4. Activate the Virtual Environment

```powershell
.venv\Scripts\Activate.ps1
```

After activation, your terminal should usually show `(.venv)` at the beginning of the prompt.

## 5. Upgrade pip

```powershell
python -m pip install --upgrade pip
```

## 6. Install All Required Packages

Install the project dependencies manually:

```powershell
pip install flask pandas numpy scikit-learn tensorflow mlflow dvc qiskit qiskit-machine-learning qiskit-algorithms
```

## 7. Verify the Environment

Run this import check:

```powershell
python -c "import flask, sklearn, tensorflow, mlflow, dvc, qiskit, qiskit_machine_learning; print('all imports ok')"
```

If you see:

```text
all imports ok
```

then the environment is ready.

## 8. Understand the Main Folders Before Running

### Project root

Contains:

- `app.py`
- `dvc.yaml`
- `mlflow.db`
- `README.md`

This is the main place from which you should run commands.

### `FLCP_ML`

Contains the classical machine learning code:

- KNN
- Decision Tree
- Random Forest
- training script
- ML prediction web form

### `FLCP_DL`

Contains the deep learning code:

- DNN training script
- saved model files
- DL web app

### `FLCP_QML`

Contains the quantum machine learning comparison code:

- QML training script
- report generator
- HTML results page

## 9. Run the Classical ML Training

From the root folder:

```powershell
cd D:\sri\FLCP_PROJECT
python FLCP_ML\train_ml.py
```

This generates:

- `FLCP_ML/ml_results.json`

It also logs the run to MLflow under:

- `ML Experiment`

## 10. Run the Deep Learning Training

From the root folder:

```powershell
cd D:\sri\FLCP_PROJECT
python FLCP_DL\train_dl.py
```

This generates:

- `FLCP_DL/dl_results.json`

If you want to regenerate the saved DL model artifacts used by the DL Flask app, also run:

```powershell
python FLCP_DL\dl_model.py
```

This creates or updates:

- `FLCP_DL/dl_model.h5`
- `FLCP_DL/scaler.pkl`
- `FLCP_DL/model_accuracy.pkl`

## 11. Run the QML Training

From the root folder:

```powershell
cd D:\sri\FLCP_PROJECT
python FLCP_QML\qml_model.py
```

This generates:

- `FLCP_QML/model_results.json`
- `FLCP_QML/results.html`

Note:

- this script usually takes longer than ML and DL
- the quantum part may take several minutes

If you want a quick test version instead of full QML training:

```powershell
python FLCP_QML\test_model.py
```

## 12. Run the Main Lecturer Demo Page

The main academic comparison dashboard is the root Flask app.

Run:

```powershell
cd D:\sri\FLCP_PROJECT
python app.py
```

Open in browser:

```text
http://127.0.0.1:5000
```

This page shows:

- ML comparison cards for KNN, Decision Tree, and Random Forest
- DNN accuracy
- QML accuracy
- final conclusion

## 13. Run the Deep Learning Web App

If you want to open the dedicated DL web application:

```powershell
cd D:\sri\FLCP_PROJECT
python FLCP_DL\dl_app.py
```

Open in browser:

```text
http://127.0.0.1:5001
```

## 14. Run the DVC Pipeline

DVC is used to reproduce the pipeline outputs from the root folder.

Run:

```powershell
cd D:\sri\FLCP_PROJECT
dvc repro
```

This executes the stages defined in `dvc.yaml`.

Expected stage outputs:

- `FLCP_ML/ml_results.json`
- `FLCP_DL/dl_results.json`
- `FLCP_QML/model_results.json`

## 15. Run MLflow UI

MLflow is used to inspect experiments, metrics, and logged models.

Run:

```powershell
cd D:\sri\FLCP_PROJECT
python -m mlflow ui --port 5002
```

Then open:

```text
http://127.0.0.1:5002
```

In MLflow, you should expect experiments like:

- `ML Experiment`
- `DL Experiment`
- `QML Experiment`

## 16. Recommended End-to-End Run Order

If you want to run the complete project in the correct order, use:

```powershell
cd D:\sri\FLCP_PROJECT
.venv\Scripts\Activate.ps1
python FLCP_ML\train_ml.py
python FLCP_DL\train_dl.py
python FLCP_QML\qml_model.py
python app.py
```

In another terminal, start MLflow:

```powershell
cd D:\sri\FLCP_PROJECT
.venv\Scripts\Activate.ps1
python -m mlflow ui --port 5002
```

Then open:

- `http://127.0.0.1:5000` for the main lecturer dashboard
- `http://127.0.0.1:5002` for MLflow experiment tracking

## 17. How to Test That Everything Is Working

### Check the result files

After training, these files should exist:

- `FLCP_ML/ml_results.json`
- `FLCP_DL/dl_results.json`
- `FLCP_QML/model_results.json`
- `FLCP_QML/results.html`

### Run the automated tests

```powershell
cd D:\sri\FLCP_PROJECT
python -m unittest discover -s tests -p "test_*.py" -v
```

### Run syntax verification

```powershell
python -m py_compile app.py FLCP_ML\train_ml.py FLCP_DL\train_dl.py FLCP_QML\qml_model.py
```

### Check the dashboard

Start:

```powershell
python app.py
```

Then confirm the page shows:

- ML section
- DNN section
- QML section
- Conclusion section

## 18. Common Issues

### `ModuleNotFoundError: No module named 'tensorflow'`

Cause:

- TensorFlow is not installed in the active environment
- or you are using an unsupported Python version

Fix:

- activate the correct virtual environment
- use Python `3.12` or `3.13`
- run `pip install tensorflow`

### `DL` or `QML` results do not appear in MLflow

Cause:

- the training script for that module has not been run yet

Fix:

- run `python FLCP_DL\train_dl.py`
- run `python FLCP_QML\qml_model.py`

### `MLflow UI opens but the Overview page is empty`

Cause:

- you are on the observability page, not the training runs page

Fix:

- open the experiment
- click `Training runs`

### `qml_model.py` takes a long time

Cause:

- quantum model training is computationally expensive

Fix:

- wait for the run to complete
- or use `python FLCP_QML\test_model.py` for a fast demo result

## 19. Final Presentation Workflow

For presentation or viva/demo, use this simple flow:

1. Open terminal in `D:\sri\FLCP_PROJECT`
2. Activate `.venv`
3. Run all training scripts
4. Start `python app.py`
5. Start `python -m mlflow ui --port 5002`
6. Open dashboard on port `5000`
7. Open MLflow on port `5002`
8. Show comparison results and conclusion

## Project Architecture in Detail

## Layer 1: Data Layer

Responsible for:

- storing the cleaned CSV dataset
- providing the same input source to all three model tracks
- enabling reproducible pipeline inputs through DVC

Primary file:

- [`DATA/lungcancer_clean.csv`](D:/sri/FLCP_PROJECT/DATA/lungcancer_clean.csv)

## Layer 2: Model Training Layer

Responsible for:

- preprocessing data
- splitting train and test data
- fitting ML, DL, and QML models
- generating evaluation metrics

Primary files:

- [`FLCP_ML/train_ml.py`](D:/sri/FLCP_PROJECT/FLCP_ML/train_ml.py)
- [`FLCP_ML/models.py`](D:/sri/FLCP_PROJECT/FLCP_ML/models.py)
- [`FLCP_DL/train_dl.py`](D:/sri/FLCP_PROJECT/FLCP_DL/train_dl.py)
- [`FLCP_DL/dl_model.py`](D:/sri/FLCP_PROJECT/FLCP_DL/dl_model.py)
- [`FLCP_QML/qml_model.py`](D:/sri/FLCP_PROJECT/FLCP_QML/qml_model.py)

## Layer 3: Model Artifact Layer

Responsible for storing trained outputs and metrics.

Examples:

- `FLCP_DL/dl_model.h5`
- `FLCP_DL/scaler.pkl`
- `FLCP_DL/model_accuracy.pkl`
- `FLCP_DL/dl_results.json`
- `FLCP_ML/ml_results.json`
- `FLCP_QML/model_results.json`

## Layer 4: Serving and Presentation Layer

Responsible for:

- collecting user input
- generating prediction responses
- rendering templates and reports

Primary files:

- [`FLCP_ML/app.py`](D:/sri/FLCP_PROJECT/FLCP_ML/app.py)
- [`FLCP_ML/templates/index.html`](D:/sri/FLCP_PROJECT/FLCP_ML/templates/index.html)
- [`FLCP_DL/dl_app.py`](D:/sri/FLCP_PROJECT/FLCP_DL/dl_app.py)
- [`FLCP_DL/templates/index.html`](D:/sri/FLCP_PROJECT/FLCP_DL/templates/index.html)
- [`FLCP_QML/generate_report.py`](D:/sri/FLCP_PROJECT/FLCP_QML/generate_report.py)

## Layer 5: Experiment and Pipeline Control Layer

Responsible for:

- reproducibility
- run tracking
- pipeline orchestration

Primary files:

- [`dvc.yaml`](D:/sri/FLCP_PROJECT/dvc.yaml)
- `mlflow.db`

## Current State and Practical Notes

While the project structure is clear, there are a few implementation details to be aware of:

- `FLCP_ML/train_ml.py`, `FLCP_DL/train_dl.py`, and `FLCP_QML/qml_model.py` are the correct sources for pipeline result files.
- The deep learning web app expects model artifacts to exist before running and now resolves them consistently from `FLCP_DL` first, then the project root as a fallback.
- The ML web app runs on port `5000` and the DL web app runs on port `5001` by default.
- Some root-level artifacts are duplicated with files inside module folders, which suggests the project evolved over time during experimentation.

## Recommended Reading Order for Understanding the Codebase

If you want to understand the repository step by step, read the files in this order:

1. [`dvc.yaml`](D:/sri/FLCP_PROJECT/dvc.yaml)
2. [`DATA/lungcancer_clean.csv`](D:/sri/FLCP_PROJECT/DATA/lungcancer_clean.csv)
3. [`FLCP_ML/models.py`](D:/sri/FLCP_PROJECT/FLCP_ML/models.py)
4. [`FLCP_ML/app.py`](D:/sri/FLCP_PROJECT/FLCP_ML/app.py)
5. [`FLCP_DL/dl_model.py`](D:/sri/FLCP_PROJECT/FLCP_DL/dl_model.py)
6. [`FLCP_DL/dl_app.py`](D:/sri/FLCP_PROJECT/FLCP_DL/dl_app.py)
7. [`FLCP_QML/qml_model.py`](D:/sri/FLCP_PROJECT/FLCP_QML/qml_model.py)
8. [`FLCP_QML/generate_report.py`](D:/sri/FLCP_PROJECT/FLCP_QML/generate_report.py)

## Summary

FLCP Project is a multi-approach lung cancer prediction system designed for comparison, learning, and experimentation. It combines:

- a classical ML ensemble
- a deep learning prediction interface
- a quantum-vs-classical comparison workflow
- MLflow experiment tracking
- DVC pipeline structure

It is best understood as a complete research-style repository that demonstrates how one healthcare classification problem can be approached through different AI paradigms within the same codebase.
