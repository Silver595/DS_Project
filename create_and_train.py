# train_models.py
"""
Train Machine Learning Models for Waterborne Disease Prediction
Northeast India - Community Health Monitoring System
"""

import pandas as pd
import numpy as np
import pickle
import os
import json
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

print("\n" + "=" * 80)
print("WATERBORNE DISEASE PREDICTION - MODEL TRAINING")
print("Northeast India Community Health Monitoring System")
print("=" * 80)

os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# ============================================================================
# STEP 1: Load and Preprocess Data
# ============================================================================
print("\n[1/4] Loading Dataset...")

df = pd.read_csv("data/raw/water_quality_data.csv")
df["sample_date"] = pd.to_datetime(df["sample_date"])

print(f"✓ Loaded {len(df):,} records")
print(
    f"✓ Date range: {df['sample_date'].min().date()} to {df['sample_date'].max().date()}"
)

# Encode categorical variables
print("\n[2/4] Preprocessing Data...")

label_encoders = {}
categorical_cols = ["state", "location_type", "water_source", "season"]

for col in categorical_cols:
    le = LabelEncoder()
    df[f"{col}_encoded"] = le.fit_transform(df[col])
    label_encoders[col] = le

# Save encoders
with open("models/label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

print("✓ Categorical variables encoded")

# Select features
feature_cols = [
    "state_encoded",
    "location_type_encoded",
    "water_source_encoded",
    "season_encoded",
    "ph",
    "turbidity_ntu",
    "tds_mg_l",
    "dissolved_oxygen_mg_l",
    "bod_mg_l",
    "fecal_coliform_mpn",
    "total_coliform_mpn",
    "nitrate_mg_l",
    "fluoride_mg_l",
    "chloride_mg_l",
    "hardness_mg_l",
    "temperature_c",
    "arsenic_ug_l",
    "iron_mg_l",
    "population_served",
    "sanitation_access_percent",
]

X = df[feature_cols]
feature_names = feature_cols

print(f"✓ Selected {len(feature_names)} features")

# ============================================================================
# STEP 3: Train Models for Each Disease
# ============================================================================
print("\n[3/4] Training Disease-Specific Models...")
print("This will take 3-5 minutes...\n")

diseases = {
    "cholera": "cholera_outbreak",
    "typhoid": "typhoid_outbreak",
    "dysentery": "dysentery_outbreak",
    "hepatitis_a": "hepatitis_a_outbreak",
    "overall": "overall_outbreak",
}

disease_models = {}
disease_scalers = {}
disease_results = []

for idx, (disease_name, target_col) in enumerate(diseases.items(), 1):
    print(f"[{idx}/5] Training: {disease_name.replace('_', ' ').title()}")

    y = df[target_col]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

    # Train XGBoost model
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1,
        verbosity=0,
    )

    model.fit(X_train_balanced, y_train_balanced)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # Cross-validation
    cv_scores = cross_val_score(
        model, X_train_balanced, y_train_balanced, cv=5, scoring="f1"
    )

    print(
        f"    Accuracy: {accuracy * 100:>6.2f}% | Precision: {precision * 100:>6.2f}% | Recall: {recall * 100:>6.2f}% | F1: {f1 * 100:>6.2f}%"
    )
    print(
        f"    CV F1-Score: {cv_scores.mean() * 100:>6.2f}% ± {cv_scores.std() * 100:>5.2f}%\n"
    )

    # Save model and scaler
    disease_models[disease_name] = model
    disease_scalers[disease_name] = scaler

    disease_results.append(
        {
            "Disease": disease_name.replace("_", " ").title(),
            "Accuracy": round(accuracy, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1-Score": round(f1, 4),
            "CV_F1_Mean": round(cv_scores.mean(), 4),
            "CV_F1_Std": round(cv_scores.std(), 4),
        }
    )

    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Outbreak", "Outbreak"],
        yticklabels=["No Outbreak", "Outbreak"],
    )
    plt.title(
        f"{disease_name.replace('_', ' ').title()} - Confusion Matrix",
        fontweight="bold",
    )
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(
        f"reports/{disease_name}_confusion_matrix.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

# ============================================================================
# STEP 4: Save Everything
# ============================================================================
print("=" * 80)
print("[4/4] Saving Models and Metadata")
print("=" * 80 + "\n")

# Save models
for disease_name, model in disease_models.items():
    with open(f"models/{disease_name}_model.pkl", "wb") as f:
        pickle.dump(model, f)

# Save scalers
for disease_name, scaler in disease_scalers.items():
    with open(f"models/{disease_name}_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

# Save feature names
with open("models/feature_names.pkl", "wb") as f:
    pickle.dump(feature_names, f)

# Save metadata
metadata = {
    "project": "Water-Borne Disease Early Warning System",
    "region": "Northeast India",
    "dataset_size": int(len(df)),
    "training_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    "diseases": list(diseases.keys()),
    "features": feature_names,
    "categorical_features": categorical_cols,
    "results": disease_results,
}

with open("models/metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

# Save results
results_df = pd.DataFrame(disease_results)
results_df.to_csv("reports/model_performance.csv", index=False)

# Feature importance for overall model
overall_model = disease_models["overall"]
feature_importance = pd.DataFrame(
    {"Feature": feature_names, "Importance": overall_model.feature_importances_}
).sort_values("Importance", ascending=False)

feature_importance.to_csv("reports/feature_importance.csv", index=False)

# Plot feature importance
plt.figure(figsize=(12, 8))
top_features = feature_importance.head(15)
sns.barplot(x="Importance", y="Feature", data=top_features, palette="viridis")
plt.title(
    "Top 15 Most Important Features for Disease Prediction",
    fontsize=14,
    fontweight="bold",
)
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("reports/feature_importance.png", dpi=300, bbox_inches="tight")
plt.close()

print("  ✓ All models saved to: models/")
print("  ✓ Metadata saved to: models/model_metadata.json")
print("  ✓ Results saved to: reports/model_performance.csv")
print("  ✓ Feature importance saved to: reports/feature_importance.csv")
print("  ✓ Visualizations saved to: reports/\n")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("=" * 80)
print("✅ MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 80)
print(f"\nDataset Size:     {len(df):,} water quality records")
print(f"Models Trained:   {len(diseases)}")
print(f"Features Used:    {len(feature_names)}")

print("\nModel Performance Summary:")
print("-" * 100)
print(
    f"{'Disease':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'CV F1':<12}"
)
print("-" * 100)
for result in disease_results:
    print(
        f"{result['Disease']:<20} {result['Accuracy'] * 100:>6.2f}%     {result['Precision'] * 100:>6.2f}%     {result['Recall'] * 100:>6.2f}%     {result['F1-Score'] * 100:>6.2f}%     {result['CV_F1_Mean'] * 100:>6.2f}%"
    )
print("-" * 100)

avg_accuracy = sum(r["Accuracy"] for r in disease_results) / len(disease_results)
print(f"\nAverage Accuracy: {avg_accuracy * 100:.2f}%")

print("\nTop 10 Most Important Features:")
print("-" * 50)
for idx, row in feature_importance.head(10).iterrows():
    print(f"  {row['Feature']:30s}: {row['Importance']:.4f}")

print("\n" + "=" * 80)
print("NEXT STEP: Run the Streamlit Dashboard")
print("=" * 80)
print("\nCommand: streamlit run app.py\n")
print("The system is now ready to monitor water quality and predict disease outbreaks!")
print("=" * 80 + "\n")
