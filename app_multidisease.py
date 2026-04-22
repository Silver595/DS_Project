# app_multidisease.py
"""
Multi-Disease Risk Prediction System
Shows specific disease predictions: Diabetes, Heart Disease, Hypertension, Stroke
"""

import os

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json

st.set_page_config(
    page_title="Multi-Disease Prediction System", page_icon="🏥", layout="wide"
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1f77b4 0%, #ff7f0e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .disease-card {
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .disease-high {
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        border-left: 5px solid #f44336;
    }
    .disease-low {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border-left: 5px solid #4caf50;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Session state
if "history" not in st.session_state:
    st.session_state.history = []


# Load models
# The canonical feature list for all 4 disease models.
# This is defined here explicitly so the app never depends on a
# potentially stale / wrong feature_names.pkl from a different project.
DISEASE_FEATURES = [
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


def train_and_save_models():
    """
    Trains all 4 disease models from synthetic data and saves them to ./models/.
    Called automatically on first run (locally or on Streamlit Cloud) when
    pre-trained .pkl files are not present in the repo.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    os.makedirs("models", exist_ok=True)
    np.random.seed(42)
    n = 5000

    age          = np.random.randint(20, 90, n)
    gender       = np.random.randint(0, 2, n)
    bmi          = np.random.uniform(15, 45, n)
    blood_pressure = np.random.randint(90, 200, n)
    cholesterol  = np.random.randint(120, 350, n)
    blood_sugar  = np.random.randint(70, 250, n)
    heart_rate   = np.random.randint(50, 120, n)
    smoking      = np.random.randint(0, 2, n)
    exercise_hours = np.random.uniform(0, 20, n)
    family_history = np.random.randint(0, 2, n)

    X = pd.DataFrame({
        "age": age, "gender": gender, "bmi": bmi,
        "blood_pressure": blood_pressure, "cholesterol": cholesterol,
        "blood_sugar": blood_sugar, "heart_rate": heart_rate,
        "smoking": smoking, "exercise_hours": exercise_hours,
        "family_history": family_history,
    })[DISEASE_FEATURES]

    # Clinically-inspired label functions
    labels = {
        "diabetes": (
            (blood_sugar > 140).astype(int) |
            ((bmi > 30) & (age > 45)).astype(int) |
            family_history
        ).clip(0, 1),
        "heart_disease": (
            (cholesterol > 240).astype(int) |
            ((blood_pressure > 140) & (smoking == 1)).astype(int) |
            ((age > 55) & (gender == 1)).astype(int)
        ).clip(0, 1),
        "hypertension": (
            (blood_pressure > 140).astype(int) |
            ((bmi > 28) & (age > 40)).astype(int) |
            (smoking == 1).astype(int)
        ).clip(0, 1),
        "stroke": (
            (blood_pressure > 160).astype(int) |
            ((age > 65) & (smoking == 1)).astype(int) |
            ((cholesterol > 260) & (blood_pressure > 150)).astype(int)
        ).clip(0, 1),
    }

    accuracies = {}
    for disease, y in labels.items():
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_tr_s, y_tr)
        accuracies[disease] = float(model.score(X_te_s, y_te))

        with open(f"models/{disease}_model.pkl", "wb") as f:
            pickle.dump(model, f)
        with open(f"models/{disease}_scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)

    with open("models/feature_names.pkl", "wb") as f:
        pickle.dump(DISEASE_FEATURES, f)

    metadata = {
        "accuracies": accuracies,
        "trained_at": datetime.now().isoformat(),
        "n_samples": n,
    }
    with open("models/multidisease_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


@st.cache_resource
def load_models():
    """
    Loads pre-trained models. If they don't exist (first deploy / fresh clone),
    trains them automatically — no manual step required.
    """
    models_missing = not os.path.exists("models/diabetes_model.pkl")

    if models_missing:
        with st.spinner("🔧 First-time setup: training models... (30–60 sec)"):
            train_and_save_models()

    try:
        diseases = ["diabetes", "heart_disease", "hypertension", "stroke"]
        models = {}
        scalers = {}

        for disease in diseases:
            with open(f"models/{disease}_model.pkl", "rb") as f:
                models[disease] = pickle.load(f)
            with open(f"models/{disease}_scaler.pkl", "rb") as f:
                scalers[disease] = pickle.load(f)

        with open("models/multidisease_metadata.json", "r") as f:
            metadata = json.load(f)

        return models, scalers, DISEASE_FEATURES, metadata, True

    except Exception as e:
        st.error(f"❌ Failed to load models: {e}")
        return None, None, None, None, False


def create_gauge(value, title):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": title, "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#ff7f0e"},
                "steps": [
                    {"range": [0, 30], "color": "#c8e6c9"},
                    {"range": [30, 70], "color": "#fff9c4"},
                    {"range": [70, 100], "color": "#ffcdd2"},
                ],
            },
        )
    )
    fig.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def main():
    models, scalers, features, metadata, loaded = load_models()

    if not loaded:
        st.error("❌ Model loading failed. Check the logs above for details.")
        st.stop()

    # Header
    st.markdown(
        '<h1 class="main-header">🏥 Multi-Disease Risk Prediction System</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: center; font-size: 1.2rem; color: #666;'>Predicts: Diabetes • Heart Disease • Hypertension • Stroke</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/medical-heart.png", width=100)
        st.markdown("## 📋 Navigation")
        page = st.radio(
            "",
            ["🏠 Home", "🔍 Predict Diseases", "📜 History"],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("### 🎯 Model Accuracies")
        for disease, acc in metadata["accuracies"].items():
            st.metric(disease, f"{acc * 100:.1f}%")

    if page == "🏠 Home":
        show_home()
    elif page == "🔍 Predict Diseases":
        show_prediction(models, scalers, features)
    elif page == "📜 History":
        show_history()


def show_home():
    st.markdown("## Welcome to Multi-Disease Risk Assessment")

    col1, col2, col3, col4 = st.columns(4)

    disease_info = [
        ("🩸 Diabetes", "Blood sugar disorder", "#e91e63"),
        ("❤️ Heart Disease", "Cardiovascular risk", "#f44336"),
        ("💉 Hypertension", "High blood pressure", "#ff9800"),
        ("🧠 Stroke", "Brain blood flow", "#9c27b0"),
    ]

    for col, (name, desc, color) in zip([col1, col2, col3, col4], disease_info):
        with col:
            st.markdown(
                f"""
            <div style="background: {color}15; padding: 1.5rem; border-radius: 10px; border-left: 4px solid {color};">
                <h3 style="color: {color}; margin: 0;">{name}</h3>
                <p style="margin: 0.5rem 0 0 0; color: #666;">{desc}</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.info(
        "👈 Click **'🔍 Predict Diseases'** in sidebar to get your personalized risk assessment for all 4 diseases!"
    )


def show_prediction(models, scalers, features):
    st.markdown("## 🔍 Multi-Disease Risk Assessment")

    with st.form("prediction_form"):
        st.markdown("### Enter Patient Health Information")

        tab1, tab2 = st.tabs(["📋 Basic Info", "🩺 Health Metrics"])

        with tab1:
            col1, col2, col3 = st.columns(3)
            with col1:
                age = st.slider("Age", 20, 90, 45)
            with col2:
                gender = st.selectbox("Gender", ["Female", "Male"])
                gender_val = 1 if gender == "Male" else 0
            with col3:
                family = st.selectbox("Family History", ["No", "Yes"])
                family_val = 1 if family == "Yes" else 0

        with tab2:
            col1, col2, col3 = st.columns(3)
            with col1:
                bmi = st.slider("BMI", 15.0, 45.0, 25.0, 0.1)
                bp = st.slider("Blood Pressure", 90, 200, 120)
                cholesterol = st.slider("Cholesterol", 120, 350, 200)
            with col2:
                blood_sugar = st.slider("Blood Sugar", 70, 250, 100)
                hr = st.slider("Heart Rate", 50, 120, 72)
            with col3:
                smoking = st.selectbox("Smoking", ["No", "Yes"])
                smoking_val = 1 if smoking == "Yes" else 0
                exercise = st.slider("Exercise (hrs/week)", 0.0, 20.0, 3.0, 0.5)

        submitted = st.form_submit_button(
            "🔍 Analyze All Disease Risks", use_container_width=True
        )

    if submitted:
        # Build input with the canonical feature order so column selection
        # is always consistent, regardless of what feature_names.pkl contains.
        input_data = pd.DataFrame(
            {
                "age": [age],
                "gender": [gender_val],
                "bmi": [bmi],
                "blood_pressure": [bp],
                "cholesterol": [cholesterol],
                "blood_sugar": [blood_sugar],
                "heart_rate": [hr],
                "smoking": [smoking_val],
                "exercise_hours": [exercise],
                "family_history": [family_val],
            }
        )[DISEASE_FEATURES]

        # Predict all diseases
        disease_names = {
            "diabetes": "🩸 Diabetes",
            "heart_disease": "❤️ Heart Disease",
            "hypertension": "💉 Hypertension",
            "stroke": "🧠 Stroke",
        }

        predictions = {}

        for disease_key, disease_label in disease_names.items():
            scaler = scalers[disease_key]
            model = models[disease_key]

            input_scaled = scaler.transform(input_data)
            pred = model.predict(input_scaled)[0]
            prob = model.predict_proba(input_scaled)[0]

            predictions[disease_key] = {
                "label": disease_label,
                "risk": pred,
                "score": prob[1] * 100,
            }

        # Display results
        st.markdown("---")
        st.markdown("## 📊 Disease Risk Assessment Results")

        # Overview metrics
        high_risk_count = sum(1 for p in predictions.values() if p["risk"] == 1)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Diseases Analyzed", "4")
        with col2:
            st.metric(
                "High Risk Detected",
                high_risk_count,
                delta="⚠️" if high_risk_count > 0 else "✅",
                delta_color="inverse",
            )
        with col3:
            avg_risk = np.mean([p["score"] for p in predictions.values()])
            st.metric("Average Risk Score", f"{avg_risk:.1f}%")

        # Individual disease cards
        st.markdown("---")
        st.markdown("### 🎯 Individual Disease Risk Breakdown")

        for disease_key, pred_data in predictions.items():
            is_high = pred_data["risk"] == 1
            score = pred_data["score"]

            col1, col2 = st.columns([3, 1])

            with col1:
                card_class = "disease-high" if is_high else "disease-low"
                status = "⚠️ HIGH RISK" if is_high else "✅ LOW RISK"
                color = "#d32f2f" if is_high else "#388e3c"

                st.markdown(
                    f"""
                <div class="disease-card {card_class}">
                    <h2 style="color: {color}; margin: 0;">{pred_data["label"]}</h2>
                    <h1 style="color: {color}; margin: 0.5rem 0;">{status}</h1>
                    <h3 style="margin: 0;">Risk Score: {score:.1f}%</h3>
                    <p style="margin: 0.5rem 0 0 0; color: #666;">
                        {"⚠️ Immediate medical consultation recommended" if is_high else "✅ Continue healthy lifestyle habits"}
                    </p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            with col2:
                fig = create_gauge(score, "")
                st.plotly_chart(fig, use_container_width=True)

        # Summary recommendations
        st.markdown("---")
        st.markdown("### 💡 Personalized Recommendations")

        if high_risk_count >= 2:
            st.error(f"""
            **🚨 URGENT: Multiple High Risks Detected ({high_risk_count} diseases)**

            - Schedule immediate appointment with healthcare provider
            - Request comprehensive medical evaluation
            - Discuss preventive medications and lifestyle changes
            - Consider specialist referrals
            """)
        elif high_risk_count == 1:
            high_risk_disease = [
                p["label"] for p in predictions.values() if p["risk"] == 1
            ][0]
            st.warning(f"""
            **⚠️ High Risk for {high_risk_disease}**

            - Consult healthcare provider about {high_risk_disease.lower()}
            - Get relevant diagnostic tests
            - Focus on disease-specific lifestyle modifications
            - Monitor relevant biomarkers regularly
            """)
        else:
            st.success("""
            **✅ All Diseases Show Low Risk**

            - Maintain current healthy lifestyle
            - Continue regular exercise routine
            - Schedule annual health checkups
            - Stay informed about preventive care
            """)

        # Save to history
        st.session_state.history.append(
            {
                "timestamp": datetime.now(),
                "age": age,
                "diabetes": predictions["diabetes"]["score"],
                "heart_disease": predictions["heart_disease"]["score"],
                "hypertension": predictions["hypertension"]["score"],
                "stroke": predictions["stroke"]["score"],
                "high_risk_count": high_risk_count,
            }
        )


def show_history():
    st.markdown("## 📜 Prediction History")

    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, use_container_width=True)

        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("No history yet. Make a prediction to see results here!")


if __name__ == "__main__":
    main()
