import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

from ML.knn import train_knn
from ML.decision_tree import train_dt
from ML.random_forest import train_rf

# Load dataset
df = pd.read_csv("DATA/lungcancer_clean.csv")

# Clean column names properly
df.columns = df.columns.str.strip().str.upper().str.replace(" ", "_")

print(df.columns)

# Correct target
X = df.drop("LUNG_CANCER", axis=1)
y = df["LUNG_CANCER"]
# Encode categorical values
le = LabelEncoder()
for col in df.columns:
    df[col] = le.fit_transform(df[col])

# Split features and target
X = df.drop("LUNG_CANCER", axis=1)
y = df["LUNG_CANCER"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Train models
knn = train_knn(X_train, y_train)
dt = train_dt(X_train, y_train)
rf = train_rf(X_train, y_train)

knn_acc = accuracy_score(y_test, knn.predict(X_test)) * 100
dt_acc = accuracy_score(y_test, dt.predict(X_test)) * 100
rf_acc = accuracy_score(y_test, rf.predict(X_test)) * 100

def predict_all(input_data):
    input_data = scaler.transform([input_data])

    k = knn.predict(input_data)[0]
    d = dt.predict(input_data)[0]
    r = rf.predict(input_data)[0]

    final = round((k + d + r) / 3)

    return k, d, r, final, knn_acc, dt_acc, rf_acc
    