# create_dataset.py
"""
Water Quality Dataset Generator - Northeast India
Enhanced with realistic correlations
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

print("\n" + "=" * 80)
print("WATERBORNE DISEASE DATASET GENERATION")
print("Northeast India - Community Health Monitoring")
print("=" * 80)

os.makedirs("data", exist_ok=True)
os.makedirs("data/raw", exist_ok=True)

np.random.seed(42)
n_samples = 15000  # Increased dataset size

print(f"\nGenerating {n_samples:,} water quality records...")

# Location data
states = [
    "Assam",
    "Meghalaya",
    "Tripura",
    "Nagaland",
    "Manipur",
    "Mizoram",
    "Arunachal Pradesh",
]
location_types = ["Rural", "Urban"]
water_sources = ["River", "Stream", "Well", "Hand Pump", "Pond", "Spring"]
seasons = ["Monsoon", "Pre-Monsoon", "Post-Monsoon", "Winter"]

# Generate with realistic distributions
data = {
    "state": np.random.choice(
        states, n_samples, p=[0.30, 0.15, 0.12, 0.10, 0.10, 0.08, 0.15]
    ),
    "location_type": np.random.choice(location_types, n_samples, p=[0.85, 0.15]),
    "water_source": np.random.choice(
        water_sources, n_samples, p=[0.20, 0.15, 0.15, 0.20, 0.15, 0.15]
    ),
    "season": np.random.choice(seasons, n_samples, p=[0.35, 0.25, 0.25, 0.15]),
}

df = pd.DataFrame(data)

print("  Generating water quality parameters...")

# pH (WHO: 6.5-8.5)
ph_base = np.random.normal(7.0, 0.8, n_samples)
df["ph"] = np.clip(ph_base - (df["season"] == "Monsoon") * 0.4, 5.0, 9.0)

# Turbidity (India: <10 NTU)
turbidity_base = np.random.exponential(7, n_samples)
df["turbidity_ntu"] = np.where(
    df["season"] == "Monsoon", turbidity_base * 2.5, turbidity_base
)
df["turbidity_ntu"] = np.clip(df["turbidity_ntu"], 0.1, 120)

# TDS (India: <500 mg/L)
df["tds_mg_l"] = np.clip(np.random.normal(350, 140, n_samples), 50, 1200)

# Dissolved Oxygen (Good: >5 mg/L)
do_base = np.where(
    df["water_source"].isin(["Stream", "Spring"]),
    np.random.normal(7.5, 1.5, n_samples),
    np.random.normal(5.5, 2.0, n_samples),
)
df["dissolved_oxygen_mg_l"] = np.clip(do_base, 1.0, 12.0)

# BOD (Good: <3 mg/L)
bod_base = np.random.exponential(2.8, n_samples)
df["bod_mg_l"] = np.where(df["location_type"] == "Urban", bod_base * 1.5, bod_base)
df["bod_mg_l"] = np.clip(df["bod_mg_l"], 0.5, 18)

# Fecal Coliform (India: <10 MPN/100ml)
fecal_base = np.random.exponential(35, n_samples)
df["fecal_coliform_mpn"] = np.where(
    df["water_source"] == "Pond",
    fecal_base * 3,
    np.where(df["season"] == "Monsoon", fecal_base * 2, fecal_base),
)
df["fecal_coliform_mpn"] = np.clip(df["fecal_coliform_mpn"], 0, 800)

# Total Coliform
df["total_coliform_mpn"] = df["fecal_coliform_mpn"] * np.random.uniform(
    2.0, 3.5, n_samples
)
df["total_coliform_mpn"] = np.clip(df["total_coliform_mpn"], 0, 2000)

# Nitrate (WHO: <50 mg/L)
df["nitrate_mg_l"] = np.clip(np.random.exponential(11, n_samples), 0, 60)

# Fluoride (India: <1.0 mg/L)
df["fluoride_mg_l"] = np.clip(np.random.exponential(0.55, n_samples), 0, 3.5)

# Chloride (WHO: <250 mg/L)
df["chloride_mg_l"] = np.clip(np.random.normal(38, 32, n_samples), 5, 220)

# Hardness
df["hardness_mg_l"] = np.clip(np.random.normal(95, 45, n_samples), 15, 280)

# Temperature
temp_base = np.where(
    df["season"] == "Winter",
    np.random.normal(18, 3, n_samples),
    np.random.normal(26, 3, n_samples),
)
df["temperature_c"] = np.clip(temp_base, 14, 33)

# Arsenic (WHO: <10 μg/L) - Higher in Assam groundwater
arsenic_base = np.random.exponential(7, n_samples)
df["arsenic_ug_l"] = np.where(
    (df["state"] == "Assam") & (df["water_source"].isin(["Hand Pump", "Well"])),
    arsenic_base * 2,
    arsenic_base,
)
df["arsenic_ug_l"] = np.clip(df["arsenic_ug_l"], 0, 150)

# Iron (India: <0.3 mg/L)
df["iron_mg_l"] = np.clip(np.random.exponential(0.6, n_samples), 0.01, 4.0)

# Population and sanitation
df["population_served"] = np.random.choice(
    [50, 100, 200, 500, 1000, 2000], n_samples, p=[0.25, 0.25, 0.20, 0.15, 0.10, 0.05]
)
df["sanitation_access_percent"] = np.random.choice(
    [20, 40, 60, 80, 100], n_samples, p=[0.15, 0.25, 0.30, 0.20, 0.10]
)

# Round values
for col in df.select_dtypes(include=[np.float64]).columns:
    df[col] = np.round(df[col], 2)

print("  ✓ Water quality parameters generated")

# Create disease outbreak labels
print("  Calculating disease outbreak risks...")

# CHOLERA - Main factors: fecal contamination, turbidity, low DO, monsoon
cholera_score = (
    (df["fecal_coliform_mpn"] > 80) * 35
    + (df["fecal_coliform_mpn"] > 40) * 25
    + (df["turbidity_ntu"] > 15) * 22
    + (df["dissolved_oxygen_mg_l"] < 4) * 23
    + (df["season"] == "Monsoon") * 28
    + (df["water_source"] == "Pond") * 25
    + (df["sanitation_access_percent"] < 40) * 18
    + np.random.normal(0, 11, n_samples)
)
df["cholera_outbreak"] = (cholera_score > 65).astype(int)

# TYPHOID - Fecal contamination, poor sanitation
typhoid_score = (
    (df["fecal_coliform_mpn"] > 60) * 33
    + (df["total_coliform_mpn"] > 300) * 28
    + (df["turbidity_ntu"] > 12) * 20
    + (df["sanitation_access_percent"] < 60) * 24
    + (df["season"].isin(["Monsoon", "Pre-Monsoon"])) * 22
    + (df["water_source"].isin(["Well", "Pond"])) * 20
    + np.random.normal(0, 10, n_samples)
)
df["typhoid_outbreak"] = (typhoid_score > 62).astype(int)

# DYSENTERY - High fecal contamination
dysentery_score = (
    (df["fecal_coliform_mpn"] > 100) * 38
    + (df["total_coliform_mpn"] > 400) * 32
    + (df["sanitation_access_percent"] < 50) * 27
    + (df["turbidity_ntu"] > 20) * 22
    + (df["season"] == "Monsoon") * 24
    + (df["water_source"] == "Pond") * 28
    + np.random.normal(0, 12, n_samples)
)
df["dysentery_outbreak"] = (dysentery_score > 68).astype(int)

# HEPATITIS A - Fecal-oral transmission
hepatitis_score = (
    (df["fecal_coliform_mpn"] > 70) * 32
    + (df["total_coliform_mpn"] > 350) * 28
    + (df["sanitation_access_percent"] < 55) * 23
    + (df["turbidity_ntu"] > 14) * 20
    + (df["season"].isin(["Pre-Monsoon", "Monsoon"])) * 22
    + (df["population_served"] > 500) * 15
    + np.random.normal(0, 11, n_samples)
)
df["hepatitis_a_outbreak"] = (hepatitis_score > 64).astype(int)

# Overall outbreak
df["overall_outbreak"] = (
    (df["cholera_outbreak"] == 1)
    | (df["typhoid_outbreak"] == 1)
    | (df["dysentery_outbreak"] == 1)
    | (df["hepatitis_a_outbreak"] == 1)
).astype(int)

print("  ✓ Disease risks calculated")

# Add timestamps
dates = pd.date_range(start="2023-01-01", end="2024-12-31", periods=n_samples)
df.insert(0, "sample_date", dates)
df.insert(1, "sample_id", [f"WQ{str(i).zfill(6)}" for i in range(1, n_samples + 1)])

# Save
df.to_csv("data/raw/water_quality_data.csv", index=False)

print(f"\n✓ Dataset saved: data/raw/water_quality_data.csv")
print(
    f"✓ File size: {os.path.getsize('data/raw/water_quality_data.csv') / (1024 * 1024):.2f} MB"
)

# Statistics
print("\n" + "=" * 80)
print("DATASET STATISTICS")
print("=" * 80)
print(f"\nTotal Records: {len(df):,}")
print(
    f"Date Range: {df['sample_date'].min().date()} to {df['sample_date'].max().date()}"
)

print(f"\nState Distribution:")
for state in states:
    count = (df["state"] == state).sum()
    print(f"  {state:20s}: {count:>5,} ({count / len(df) * 100:>5.1f}%)")

print(f"\nWater Source Distribution:")
for source in water_sources:
    count = (df["water_source"] == source).sum()
    print(f"  {source:15s}: {count:>5,} ({count / len(df) * 100:>5.1f}%)")

print(f"\nDisease Outbreak Distribution:")
diseases = [
    "cholera_outbreak",
    "typhoid_outbreak",
    "dysentery_outbreak",
    "hepatitis_a_outbreak",
    "overall_outbreak",
]
disease_names = ["Cholera", "Typhoid", "Dysentery", "Hepatitis A", "Overall"]
for name, col in zip(disease_names, diseases):
    count = df[col].sum()
    print(f"  {name:12s}: {count:>5,} ({df[col].mean() * 100:>5.2f}%)")

print("\n" + "=" * 80)
print("✅ DATASET GENERATION COMPLETE!")
print("=" * 80)
print("\nNext step: Train models")
print("Command: python train_models.py")
print("=" * 80 + "\n")
