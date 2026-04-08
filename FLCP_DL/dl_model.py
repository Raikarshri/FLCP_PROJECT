from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')


np.random.seed(42)
BASE_DIR = Path(__file__).resolve().parent.parent
MODULE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "DATA" / "lungcancer_clean.csv"

# Load dataset - FIXED PATH
df = pd.read_csv(DATASET_PATH)

# Clean column names
df.columns = df.columns.str.strip()
df.columns = df.columns.str.replace(" ", "_")

# Convert categorical safely
df = df.replace({
    "YES": 1, "NO": 0,
    "Yes": 1, "No": 0,
    "M": 1, "F": 0
})

# Ensure all numeric
df = df.apply(pd.to_numeric)

print("Dataset shape:", df.shape)
print("Columns:", list(df.columns))
print("Class distribution:\n", df["LUNG_CANCER"].value_counts())

# Split features and target
X = df.drop("LUNG_CANCER", axis=1)
y = df["LUNG_CANCER"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Further split for validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print(f"\nTraining set: {X_train.shape[0]}")
print(f"Validation set: {X_val.shape[0]}")
print(f"Test set: {X_test.shape[0]}")

# Build improved model architecture
model = Sequential([
    Dense(128, activation='relu', input_dim=X_train.shape[1]),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile with better optimizer
optimizer = Adam(learning_rate=0.001)
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Early stopping callback
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Train model
print("\nTraining model...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=16,
    callbacks=[early_stop],
    verbose=1
)

# Evaluate on test set
print("\nEvaluating on test set...")
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

test_accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Cancer', 'Cancer']))

# Save model, scaler, and accuracy
print("\nSaving model, scaler, and accuracy...")
model.save(MODULE_DIR / "dl_model.h5")
with (MODULE_DIR / "scaler.pkl").open("wb") as scaler_file:
    pickle.dump(scaler, scaler_file)
with (MODULE_DIR / "model_accuracy.pkl").open("wb") as accuracy_file:
    pickle.dump({"accuracy": float(test_accuracy)}, accuracy_file)
print("Done! Model, scaler, and accuracy saved.")
