import os
import json
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

os.makedirs("models", exist_ok=True)

# load dataset
df = pd.read_csv("data/multidisease_dataset.csv")

features = [
    "age",
    "gender",
    "bmi",
    "blood_pressure",
    "cholesterol",
    "blood_sugar",
    "heart_rate",
    "smoking",
    "exercise_hours",
    "family_history"
]

targets = {
    "diabetes": "diabetes_risk",
    "heart_disease": "heart_disease_risk",
    "hypertension": "hypertension_risk",
    "stroke": "stroke_risk"
}

accuracies = {}

for disease_name, target_col in targets.items():
    print(f"\nTraining {disease_name}...")

    X = df[features]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    accuracies[disease_name] = acc

    with open(f"models/{disease_name}_model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open(f"models/{disease_name}_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print(f"{disease_name} accuracy: {acc:.4f}")

# save features
with open("models/feature_names.pkl", "wb") as f:
    pickle.dump(features, f)

# save metadata
metadata = {"accuracies": accuracies}
with open("models/multidisease_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

print("\n✅ All 4 models created successfully!")