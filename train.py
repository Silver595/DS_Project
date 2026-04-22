# create_and_train_multidisease.py
"""
Multi-Disease Risk Prediction System
Predicts: Diabetes, Heart Disease, Hypertension, Stroke
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings("ignore")

print("\n" + "=" * 80)
print("MULTI-DISEASE RISK PREDICTION SYSTEM - TRAINING")
print("=" * 80)
print("Diseases: Diabetes, Heart Disease, Hypertension, Stroke")
print("=" * 80)

os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# ============================================================================
# STEP 1: Generate Multi-Disease Dataset
# ============================================================================
print("\n[1/3] Generating Multi-Disease Dataset...")

np.random.seed(42)
n_samples = 5000

# Base patient data
age = np.random.randint(20, 85, n_samples)
gender = np.random.randint(0, 2, n_samples)
bmi_base = 22 + (age - 40) * 0.08 + np.random.normal(0, 4, n_samples)
bmi = np.clip(bmi_base, 16, 45)

# Vitals
bp_base = 100 + (age * 0.4) + (bmi - 22) * 1.2 + np.random.normal(0, 12, n_samples)
blood_pressure = np.clip(bp_base, 90, 200).astype(int)

chol_base = 150 + (age * 0.8) + (bmi - 22) * 2.0 + np.random.normal(0, 25, n_samples)
cholesterol = np.clip(chol_base, 120, 350).astype(int)

family_history = np.random.binomial(1, 0.35, n_samples)
sugar_base = (
    80
    + (age * 0.3)
    + (bmi - 22) * 1.5
    + (family_history * 15)
    + np.random.normal(0, 18, n_samples)
)
blood_sugar = np.clip(sugar_base, 70, 250).astype(int)

heart_rate = np.clip(
    75 - (age - 40) * 0.1 + np.random.normal(0, 10, n_samples), 50, 120
).astype(int)
smoking = np.random.binomial(1, 0.22, n_samples)
exercise_base = (
    8 - (bmi - 22) * 0.2 - (age - 40) * 0.05 + np.random.normal(0, 2.5, n_samples)
)
exercise_hours = np.clip(exercise_base, 0, 20)

df = pd.DataFrame(
    {
        "age": age,
        "gender": gender,
        "bmi": np.round(bmi, 1),
        "blood_pressure": blood_pressure,
        "cholesterol": cholesterol,
        "blood_sugar": blood_sugar,
        "heart_rate": heart_rate,
        "smoking": smoking,
        "exercise_hours": np.round(exercise_hours, 1),
        "family_history": family_history,
    }
)

# ============================================================================
# CREATE DISEASE-SPECIFIC TARGETS
# ============================================================================

print("  Generating disease-specific risk scores...")

# 1. DIABETES Risk Score
diabetes_score = (
    (df["blood_sugar"] > 126) * 50
    + (df["blood_sugar"] > 100) * 25
    + (df["bmi"] > 30) * 30
    + (df["bmi"] > 25) * 15
    + (df["age"] > 45) * 20
    + df["family_history"] * 35
    + (df["exercise_hours"] < 3) * 15
    + np.random.normal(0, 10, n_samples)
)
df["diabetes_risk"] = (diabetes_score > 80).astype(int)

# 2. HEART DISEASE Risk Score
heart_score = (
    (df["cholesterol"] > 240) * 40
    + (df["cholesterol"] > 200) * 20
    + (df["blood_pressure"] > 140) * 35
    + (df["blood_pressure"] > 130) * 20
    + df["smoking"] * 40
    + (df["age"] > 55) * 30
    + (df["bmi"] > 30) * 25
    + df["family_history"] * 30
    + (df["exercise_hours"] < 2) * 20
    + np.random.normal(0, 12, n_samples)
)
df["heart_disease_risk"] = (heart_score > 90).astype(int)

# 3. HYPERTENSION Risk Score
hypertension_score = (
    (df["blood_pressure"] > 160) * 60
    + (df["blood_pressure"] > 140) * 40
    + (df["blood_pressure"] > 130) * 25
    + (df["bmi"] > 30) * 30
    + (df["age"] > 50) * 25
    + df["smoking"] * 20
    + (df["exercise_hours"] < 3) * 15
    + df["family_history"] * 25
    + np.random.normal(0, 10, n_samples)
)
df["hypertension_risk"] = (hypertension_score > 85).astype(int)

# 4. STROKE Risk Score
stroke_score = (
    (df["blood_pressure"] > 160) * 45
    + (df["blood_pressure"] > 140) * 30
    + (df["age"] > 65) * 40
    + (df["age"] > 55) * 25
    + df["smoking"] * 35
    + (df["cholesterol"] > 240) * 25
    + (df["blood_sugar"] > 126) * 20
    + df["family_history"] * 30
    + (df["bmi"] > 30) * 20
    + np.random.normal(0, 12, n_samples)
)
df["stroke_risk"] = (stroke_score > 95).astype(int)

# Save dataset
df.to_csv("data/multidisease_dataset.csv", index=False)

print(f"\n  ✓ Generated {len(df)} patient records")
print(f"\n  Disease Distribution:")
print(
    f"    Diabetes:     {df['diabetes_risk'].sum():4d} ({df['diabetes_risk'].sum() / len(df) * 100:.1f}%)"
)
print(
    f"    Heart Disease:{df['heart_disease_risk'].sum():4d} ({df['heart_disease_risk'].sum() / len(df) * 100:.1f}%)"
)
print(
    f"    Hypertension: {df['hypertension_risk'].sum():4d} ({df['hypertension_risk'].sum() / len(df) * 100:.1f}%)"
)
print(
    f"    Stroke:       {df['stroke_risk'].sum():4d} ({df['stroke_risk'].sum() / len(df) * 100:.1f}%)"
)

# ============================================================================
# STEP 2: Train Models for Each Disease
# ============================================================================
print("\n[2/3] Training Disease-Specific Models...")

X = df[
    [
        "age",
        "gender",
        "bmi",
        "blood_pressure",
        "cholesterol",
        "blood_sugar",
        "heart_rate",
        "smoking",
        "exercise_hours",
        "family_history",
    ]
]
feature_names = X.columns.tolist()

diseases = {
    "diabetes": "diabetes_risk",
    "heart_disease": "heart_disease_risk",
    "hypertension": "hypertension_risk",
    "stroke": "stroke_risk",
}

models = {}
scalers = {}
results = []

for disease_name, target_col in diseases.items():
    print(f"\n  {'─' * 76}")
    print(f"  Training: {disease_name.replace('_', ' ').title()}")
    print(f"  {'─' * 76}")

    y = df[target_col]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Balance
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

    # Train
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss",
    )

    model.fit(X_train_balanced, y_train_balanced)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"    Accuracy: {accuracy * 100:.2f}%")

    models[disease_name] = model
    scalers[disease_name] = scaler
    results.append(
        {"Disease": disease_name.replace("_", " ").title(), "Accuracy": accuracy}
    )

# ============================================================================
# STEP 3: Save All Models
# ============================================================================
print("\n[3/3] Saving Models...")

# Save each disease model
for disease_name, model in models.items():
    with open(f"models/{disease_name}_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"  ✓ Saved: models/{disease_name}_model.pkl")

# Save each scaler
for disease_name, scaler in scalers.items():
    with open(f"models/{disease_name}_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

# Save feature names
with open("models/feature_names.pkl", "wb") as f:
    pickle.dump(feature_names, f)

# Save metadata
import json

metadata = {
    "diseases": list(diseases.keys()),
    "features": feature_names,
    "training_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    "training_samples": len(X_train_balanced),
    "accuracies": {r["Disease"]: float(r["Accuracy"]) for r in results},
}

with open("models/multidisease_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

print(f"  ✓ Saved: models/multidisease_metadata.json")

# Display results
print("\n" + "=" * 80)
print("MODEL TRAINING COMPLETED")
print("=" * 80)
print(f"\n{'Disease':<20} {'Accuracy':<15}")
print("-" * 35)
for r in results:
    print(f"{r['Disease']:<20} {r['Accuracy'] * 100:>6.2f}%")

print("\n" + "=" * 80)
print("✅ ALL MODELS TRAINED SUCCESSFULLY!")
print("=" * 80)
print("\nNext step: Run the enhanced Streamlit app")
print("Command: streamlit run app_multidisease.py")
print("=" * 80 + "\n")
