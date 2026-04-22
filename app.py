# app.py
"""
Water-Borne Disease Early Warning System
Northeast India Community Health Monitoring
COMPLETE VERSION WITH ALL FEATURES
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Water Disease Early Warning",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #0066cc 0%, #00cc99 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .alert-high {
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        border-left: 5px solid #f44336;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .alert-low {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border-left: 5px solid #4caf50;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        transition: transform 0.3s;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    .stButton>button {
        background: linear-gradient(90deg, #0066cc 0%, #00cc99 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 12px rgba(0, 102, 204, 0.3);
    }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================================
# SESSION STATE
# ============================================================================
if "test_history" not in st.session_state:
    st.session_state.test_history = []
if "alerts" not in st.session_state:
    st.session_state.alerts = []


# ============================================================================
# LOAD MODELS
# ============================================================================
@st.cache_resource
def load_all_models():
    try:
        diseases = ["cholera", "typhoid", "dysentery", "hepatitis_a", "overall"]
        models = {}
        scalers = {}

        for disease in diseases:
            with open(f"models/{disease}_model.pkl", "rb") as f:
                models[disease] = pickle.load(f)
            with open(f"models/{disease}_scaler.pkl", "rb") as f:
                scalers[disease] = pickle.load(f)

        with open("models/feature_names.pkl", "rb") as f:
            features = pickle.load(f)

        with open("models/label_encoders.pkl", "rb") as f:
            encoders = pickle.load(f)

        with open("models/metadata.json", "r") as f:
            metadata = json.load(f)

        return models, scalers, features, encoders, metadata, True
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, None, None, False


# ============================================================================
# DISEASE INFORMATION
# ============================================================================
DISEASES = {
    "cholera": {
        "name": "🦠 Cholera",
        "color": "#e53935",
        "description": "Acute diarrheal infection caused by Vibrio cholerae",
        "symptoms": "Severe watery diarrhea, vomiting, rapid dehydration",
        "prevention": "Safe water, proper sanitation, hand hygiene, vaccination",
    },
    "typhoid": {
        "name": "🌡️ Typhoid",
        "color": "#fb8c00",
        "description": "Bacterial infection caused by Salmonella typhi",
        "symptoms": "Prolonged fever, weakness, stomach pain, headache, loss of appetite",
        "prevention": "Clean water, vaccination, food safety, hygiene",
    },
    "dysentery": {
        "name": "💊 Dysentery",
        "color": "#8e24aa",
        "description": "Intestinal inflammation causing bloody diarrhea",
        "symptoms": "Bloody diarrhea, severe abdominal cramps, fever, nausea",
        "prevention": "Safe water, hygiene, proper sanitation, food safety",
    },
    "hepatitis_a": {
        "name": "🧬 Hepatitis A",
        "color": "#43a047",
        "description": "Liver infection caused by Hepatitis A virus",
        "symptoms": "Fatigue, nausea, jaundice, abdominal pain, dark urine",
        "prevention": "Vaccination, safe water, hand hygiene, food safety",
    },
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def create_gauge(value, title="Risk Score"):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": title, "font": {"size": 18}},
            number={"font": {"size": 36}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#00cc99", "thickness": 0.75},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "#0066cc",
                "steps": [
                    {"range": [0, 30], "color": "#c8e6c9"},
                    {"range": [30, 70], "color": "#fff9c4"},
                    {"range": [70, 100], "color": "#ffcdd2"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 70,
                },
            },
        )
    )
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def get_water_quality_status(params):
    """Assess overall water quality"""
    issues = []

    if params.get("ph", 7) < 6.5 or params.get("ph", 7) > 8.5:
        issues.append("pH abnormal")
    if params.get("turbidity_ntu", 0) > 10:
        issues.append("High turbidity")
    if params.get("fecal_coliform_mpn", 0) > 10:
        issues.append("Fecal contamination")
    if params.get("dissolved_oxygen_mg_l", 10) < 5:
        issues.append("Low oxygen")
    if params.get("arsenic_ug_l", 0) > 10:
        issues.append("Arsenic detected")
    if params.get("nitrate_mg_l", 0) > 45:
        issues.append("High nitrate")

    if len(issues) == 0:
        return "SAFE", "#4caf50", "✅ Water meets all safety standards"
    elif len(issues) <= 2:
        return "MODERATE", "#ff9800", f"⚠️ Issues found: {', '.join(issues)}"
    else:
        return "UNSAFE", "#f44336", f"❌ Multiple issues: {', '.join(issues)}"


# ============================================================================
# MAIN APP
# ============================================================================
def main():
    models, scalers, features, encoders, metadata, loaded = load_all_models()

    if not loaded:
        st.error("❌ **Models not found!**")
        st.info("Please run: `python train_models.py`")
        st.stop()

    # Header
    st.markdown(
        '<h1 class="main-header">💧 Water-Borne Disease Early Warning System</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: center; font-size: 1.2rem; color: #666;'>Smart Community Health Monitoring • Northeast India</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/water.png", width=80)
        st.markdown("## 📋 Navigation")

        # FIX: changed empty string "" to "Navigation" to satisfy Streamlit's label requirement
        page = st.radio(
            "Navigation",
            [
                "🏠 Dashboard",
                "🔬 Water Quality Test",
                "📊 Batch Analysis",
                "📈 Model Performance",
                "📜 Test History",
                "ℹ️ About System",
            ],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("### 📊 System Stats")
        st.metric("Total Tests", len(st.session_state.test_history))
        st.metric(
            "Active Alerts",
            len([a for a in st.session_state.alerts if a.get("active", True)]),
        )

        # Display model accuracy
        if metadata and "results" in metadata:
            st.markdown("---")
            st.markdown("### 🎯 Model Accuracy")
            for result in metadata["results"]:
                if result["Disease"] == "Overall":
                    st.metric(
                        result["Disease"],
                        f"{result['Accuracy'] * 100:.1f}%",
                        delta=f"F1: {result['F1-Score'] * 100:.1f}%",
                    )

        st.markdown("---")
        st.success(f"📅 {datetime.now().strftime('%d %B %Y')}")
        st.info("🕐 " + datetime.now().strftime("%I:%M %p"))

    # Route pages
    if page == "🏠 Dashboard":
        show_dashboard(metadata)
    elif page == "🔬 Water Quality Test":
        show_water_test(models, scalers, features, encoders)
    elif page == "📊 Batch Analysis":
        show_batch_analysis(models, scalers, features, encoders)
    elif page == "📈 Model Performance":
        show_model_performance(metadata)
    elif page == "📜 Test History":
        show_test_history()
    elif page == "ℹ️ About System":
        show_about(metadata)


# ============================================================================
# PAGE: DASHBOARD
# ============================================================================
def show_dashboard(metadata):
    st.markdown("## 🏠 System Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            """
        <div class="metric-card">
            <h3 style="color: #0066cc;">💧</h3>
            <h4>Water Quality</h4>
            <p>Real-time monitoring</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
        <div class="metric-card">
            <h3 style="color: #00cc99;">🦠</h3>
            <h4>4 Diseases</h4>
            <p>AI prediction models</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
        <div class="metric-card">
            <h3 style="color: #ff9800;">🚨</h3>
            <h4>Early Warning</h4>
            <p>Outbreak detection</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            """
        <div class="metric-card">
            <h3 style="color: #e91e63;">🌏</h3>
            <h4>NE India</h4>
            <p>7 states covered</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Disease information cards
    st.markdown("## 🦠 Monitored Water-Borne Diseases")

    col1, col2 = st.columns(2)

    for idx, (disease_key, disease_info) in enumerate(DISEASES.items()):
        with col1 if idx % 2 == 0 else col2:
            st.markdown(
                f"""
            <div style="background: {disease_info["color"]}10; padding: 1.5rem; border-radius: 10px;
                       border-left: 4px solid {disease_info["color"]}; margin: 0.5rem 0;">
                <h3 style="color: {disease_info["color"]}; margin: 0;">{disease_info["name"]}</h3>
                <p style="margin: 0.5rem 0 0 0;"><strong>{disease_info["description"]}</strong></p>
                <p style="margin: 0.3rem 0; font-size: 0.9rem; color: #666;"><strong>Symptoms:</strong> {disease_info["symptoms"]}</p>
                <p style="margin: 0.3rem 0; font-size: 0.9rem; color: #666;"><strong>Prevention:</strong> {disease_info["prevention"]}</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # System accuracy
    if metadata and "results" in metadata:
        st.markdown("## 🎯 System Performance")

        results_df = pd.DataFrame(metadata["results"])

        col1, col2 = st.columns([2, 1])

        with col1:
            # Bar chart
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    name="Accuracy",
                    x=results_df["Disease"],
                    y=results_df["Accuracy"] * 100,
                    marker_color="#0066cc",
                )
            )
            fig.add_trace(
                go.Bar(
                    name="F1-Score",
                    x=results_df["Disease"],
                    y=results_df["F1-Score"] * 100,
                    marker_color="#00cc99",
                )
            )
            fig.update_layout(
                title="Model Performance by Disease",
                xaxis_title="Disease",
                yaxis_title="Score (%)",
                barmode="group",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### 📊 Metrics")
            for result in results_df.to_dict("records"):
                st.metric(
                    result["Disease"],
                    f"{result['Accuracy'] * 100:.1f}%",
                    delta=f"F1: {result['F1-Score'] * 100:.1f}%",
                )

    st.markdown("---")
    st.info(
        "👈 **Click 'Water Quality Test'** in the sidebar to start testing water samples!"
    )


# ============================================================================
# PAGE: WATER QUALITY TEST (MAIN FEATURE)
# ============================================================================
def show_water_test(models, scalers, features, encoders):
    st.markdown("## 🔬 Water Quality Test & Disease Risk Assessment")

    st.info("""
    **📋 Instructions:** Enter water quality test results below. The system will analyze the data and
    predict outbreak risk for 4 waterborne diseases using AI/ML models.
    """)

    with st.form("water_test_form"):
        st.markdown("### 📍 Sample Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            state = st.selectbox(
                "State",
                [
                    "Assam",
                    "Meghalaya",
                    "Tripura",
                    "Nagaland",
                    "Manipur",
                    "Mizoram",
                    "Arunachal Pradesh",
                ],
            )
            location = st.selectbox("Location Type", ["Rural", "Urban"])

        with col2:
            source = st.selectbox(
                "Water Source",
                ["River", "Stream", "Well", "Hand Pump", "Pond", "Spring"],
            )
            season = st.selectbox(
                "Season", ["Monsoon", "Pre-Monsoon", "Post-Monsoon", "Winter"]
            )

        with col3:
            population = st.selectbox(
                "Population Served", [50, 100, 200, 500, 1000, 2000]
            )
            sanitation = st.slider("Sanitation Access (%)", 0, 100, 60, 10)

        st.markdown("---")
        st.markdown("### 🧪 Water Quality Parameters")

        tab1, tab2, tab3 = st.tabs(
            ["Basic Parameters", "Chemical Tests", "Biological Tests"]
        )

        with tab1:
            col1, col2, col3 = st.columns(3)

            with col1:
                ph = st.number_input(
                    "pH", 5.0, 9.0, 7.0, 0.1, help="WHO: 6.5-8.5 optimal"
                )
                turbidity = st.number_input(
                    "Turbidity (NTU)", 0.0, 120.0, 5.0, 0.5, help="India: <10 NTU"
                )

            with col2:
                tds = st.number_input(
                    "TDS (mg/L)", 50.0, 1200.0, 300.0, 10.0, help="India: <500 mg/L"
                )
                do = st.number_input(
                    "Dissolved Oxygen (mg/L)", 1.0, 12.0, 6.5, 0.1, help="Good: >5 mg/L"
                )

            with col3:
                bod = st.number_input(
                    "BOD (mg/L)", 0.5, 18.0, 2.0, 0.1, help="Good: <3 mg/L"
                )
                temp = st.number_input("Temperature (°C)", 14.0, 33.0, 25.0, 0.5)

        with tab2:
            col1, col2, col3 = st.columns(3)

            with col1:
                nitrate = st.number_input(
                    "Nitrate (mg/L)", 0.0, 60.0, 10.0, 0.5, help="WHO: <50 mg/L"
                )
                fluoride = st.number_input(
                    "Fluoride (mg/L)", 0.0, 3.5, 0.5, 0.1, help="India: <1.0 mg/L"
                )

            with col2:
                chloride = st.number_input(
                    "Chloride (mg/L)", 5.0, 220.0, 30.0, 1.0, help="WHO: <250 mg/L"
                )
                hardness = st.number_input(
                    "Hardness (mg/L)", 15.0, 280.0, 95.0, 5.0, help="Soft: <60 mg/L"
                )

            with col3:
                arsenic = st.number_input(
                    "Arsenic (μg/L)", 0.0, 150.0, 5.0, 0.5, help="WHO: <10 μg/L"
                )
                iron = st.number_input(
                    "Iron (mg/L)", 0.0, 4.0, 0.3, 0.05, help="India: <0.3 mg/L"
                )

        with tab3:
            col1, col2 = st.columns(2)

            with col1:
                fecal = st.number_input(
                    "Fecal Coliform (MPN/100ml)",
                    0.0,
                    800.0,
                    10.0,
                    1.0,
                    help="India: <10 MPN/100ml - Critical parameter",
                )

            with col2:
                total_col = st.number_input(
                    "Total Coliform (MPN/100ml)",
                    0.0,
                    2000.0,
                    50.0,
                    5.0,
                    help="Should be minimal",
                )

        submitted = st.form_submit_button(
            "🔍 Analyze Water Quality & Predict Disease Risk", use_container_width=True
        )

    if submitted:
        # Prepare test data
        test_params = {
            "ph": ph,
            "turbidity_ntu": turbidity,
            "tds_mg_l": tds,
            "dissolved_oxygen_mg_l": do,
            "bod_mg_l": bod,
            "fecal_coliform_mpn": fecal,
            "total_coliform_mpn": total_col,
            "nitrate_mg_l": nitrate,
            "fluoride_mg_l": fluoride,
            "chloride_mg_l": chloride,
            "hardness_mg_l": hardness,
            "temperature_c": temp,
            "arsenic_ug_l": arsenic,
            "iron_mg_l": iron,
            "population_served": population,
            "sanitation_access_percent": sanitation,
        }

        # Encode categorical
        input_encoded = {
            "state_encoded": encoders["state"].transform([state])[0],
            "location_type_encoded": encoders["location_type"].transform([location])[0],
            "water_source_encoded": encoders["water_source"].transform([source])[0],
            "season_encoded": encoders["season"].transform([season])[0],
        }

        # Create input dataframe
        input_data = pd.DataFrame([{**input_encoded, **test_params}])[features]

        # Predict all diseases
        disease_predictions = {}

        for disease_key in ["cholera", "typhoid", "dysentery", "hepatitis_a"]:
            scaled = scalers[disease_key].transform(input_data)
            pred = models[disease_key].predict(scaled)[0]
            prob = models[disease_key].predict_proba(scaled)[0]
            disease_predictions[disease_key] = {"risk": pred, "score": prob[1] * 100}

        # Overall prediction
        overall_scaled = scalers["overall"].transform(input_data)
        overall_pred = models["overall"].predict(overall_scaled)[0]
        overall_prob = models["overall"].predict_proba(overall_scaled)[0]
        overall_risk = overall_prob[1] * 100

        # Water quality assessment
        quality_status, quality_color, quality_msg = get_water_quality_status(
            test_params
        )

        # Save to history
        st.session_state.test_history.append(
            {
                "timestamp": datetime.now(),
                "state": state,
                "water_source": source,
                "overall_risk": overall_risk,
                "status": quality_status,
            }
        )

        # Check for alerts
        high_risk_count = sum(1 for p in disease_predictions.values() if p["risk"] == 1)
        if high_risk_count >= 2 or overall_risk > 70:
            st.session_state.alerts.append(
                {
                    "timestamp": datetime.now(),
                    "location": f"{state} - {source}",
                    "risk_level": "HIGH",
                    "diseases": [
                        k for k, v in disease_predictions.items() if v["risk"] == 1
                    ],
                    "score": overall_risk,
                    "active": True,
                }
            )

        # DISPLAY RESULTS
        st.markdown("---")
        st.markdown("## 📊 Analysis Results")

        # Water quality status
        st.markdown("### 💧 Water Quality Assessment")
        if quality_status == "SAFE":
            st.success(quality_msg)
        elif quality_status == "MODERATE":
            st.warning(quality_msg)
        else:
            st.error(quality_msg)

        # Overall risk
        st.markdown("---")
        st.markdown("### 🎯 Overall Disease Outbreak Risk")

        col1, col2 = st.columns([2, 1])

        with col1:
            if overall_pred == 1:
                st.markdown(
                    f"""
                <div class="alert-high">
                    <h2 style="color: #d32f2f; margin: 0;">🚨 HIGH OUTBREAK RISK DETECTED</h2>
                    <h1 style="color: #d32f2f; margin: 0.5rem 0;">Risk Score: {overall_risk:.1f}%</h1>
                    <p style="font-size: 1.1rem; margin: 0;">Immediate intervention required for this water source!</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                <div class="alert-low">
                    <h2 style="color: #388e3c; margin: 0;">✅ LOW OUTBREAK RISK</h2>
                    <h1 style="color: #388e3c; margin: 0.5rem 0;">Risk Score: {overall_risk:.1f}%</h1>
                    <p style="font-size: 1.1rem; margin: 0;">Continue regular monitoring of water quality</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

        with col2:
            st.plotly_chart(create_gauge(overall_risk), use_container_width=True)

        # Individual disease risks
        st.markdown("---")
        st.markdown("### 🦠 Disease-Specific Risk Assessment")

        col1, col2 = st.columns(2)

        for idx, (disease_key, disease_info) in enumerate(DISEASES.items()):
            pred_data = disease_predictions[disease_key]
            is_high = pred_data["risk"] == 1
            score = pred_data["score"]

            with col1 if idx % 2 == 0 else col2:
                status = "HIGH RISK" if is_high else "LOW RISK"
                bg_color = f"{disease_info['color']}15" if is_high else "#4caf5010"
                border_color = disease_info["color"] if is_high else "#4caf50"

                st.markdown(
                    f"""
                <div style="background: {bg_color}; padding: 1.2rem; border-radius: 8px;
                           border-left: 4px solid {border_color}; margin: 0.5rem 0;">
                    <h4 style="color: {border_color}; margin: 0;">{disease_info["name"]}</h4>
                    <h3 style="color: {border_color}; margin: 0.3rem 0;">{status}</h3>
                    <p style="margin: 0; font-size: 1.1rem;"><strong>Risk Score: {score:.1f}%</strong></p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

        # Recommendations
        st.markdown("---")
        st.markdown("### 💡 Recommended Actions")

        if high_risk_count >= 2:
            st.error(f"""
            **🚨 URGENT: Multiple Disease Risks Detected ({high_risk_count} diseases)**

            **Immediate Actions Required:**
            1. 🚫 **STOP using this water source immediately**
            2. 📢 Alert all community members and local health authorities
            3. 🏥 Set up medical screening camps for early symptom detection
            4. 💧 Arrange alternate safe water supply (tankers/packaged water)
            5. 🧪 Conduct comprehensive water quality re-testing
            6. 🛡️ Implement emergency water treatment (boiling/chlorination)
            7. 📋 Monitor community for disease symptoms daily
            8. 🏗️ Investigate and repair water source contamination
            """)
        elif high_risk_count == 1:
            high_disease = [
                DISEASES[k]["name"]
                for k, v in disease_predictions.items()
                if v["risk"] == 1
            ][0]
            st.warning(f"""
            **⚠️ Single Disease Risk Detected: {high_disease}**

            **Recommended Actions:**
            1. 💧 **Boil all water** for drinking and cooking (minimum 1 minute)
            2. 🧪 Conduct follow-up water testing within 48 hours
            3. 📢 Inform community members about the risk
            4. 🏥 Monitor for symptoms: {next(v["symptoms"] for k, v in DISEASES.items() if DISEASES[k]["name"] == high_disease)}
            5. 🛠️ Improve water treatment and sanitation facilities
            6. 📊 Increase testing frequency to weekly
            """)
        else:
            st.success("""
            **✅ All Disease Risks Are Low**

            **Maintenance Actions:**
            1. ✅ Continue regular water quality testing (monthly minimum)
            2. 🧹 Maintain cleanliness around water source
            3. 📊 Keep detailed records of all test results
            4. 🏥 Conduct periodic community health screenings
            5. 📚 Provide hygiene education to community members
            6. 🔍 Watch for seasonal changes affecting water quality
            """)

        # Parameter analysis
        st.markdown("---")
        st.markdown("### 📊 Parameter Analysis")

        param_analysis = []

        if ph < 6.5:
            param_analysis.append(("pH", f"{ph:.1f}", "Too acidic", "🔴"))
        elif ph > 8.5:
            param_analysis.append(("pH", f"{ph:.1f}", "Too alkaline", "🔴"))
        else:
            param_analysis.append(("pH", f"{ph:.1f}", "Normal range", "🟢"))

        if turbidity > 10:
            param_analysis.append(
                ("Turbidity", f"{turbidity:.1f} NTU", "Exceeds limit", "🔴")
            )
        else:
            param_analysis.append(
                ("Turbidity", f"{turbidity:.1f} NTU", "Within limit", "🟢")
            )

        if fecal > 10:
            param_analysis.append(
                ("Fecal Coliform", f"{fecal:.0f} MPN", "Contaminated", "🔴")
            )
        else:
            param_analysis.append(("Fecal Coliform", f"{fecal:.0f} MPN", "Safe", "🟢"))

        if do < 5:
            param_analysis.append(("Dissolved Oxygen", f"{do:.1f} mg/L", "Low", "🔴"))
        else:
            param_analysis.append(("Dissolved Oxygen", f"{do:.1f} mg/L", "Good", "🟢"))

        if arsenic > 10:
            param_analysis.append(
                ("Arsenic", f"{arsenic:.1f} μg/L", "Exceeds limit", "🔴")
            )
        else:
            param_analysis.append(("Arsenic", f"{arsenic:.1f} μg/L", "Safe", "🟢"))

        col1, col2, col3, col4 = st.columns(4)

        for idx, (param, value, status, icon) in enumerate(param_analysis):
            with [col1, col2, col3, col4][idx % 4]:
                st.markdown(
                    f"""
                <div class="metric-card">
                    <h2 style="margin: 0;">{icon}</h2>
                    <h4 style="margin: 0.5rem 0;">{param}</h4>
                    <p style="margin: 0; font-size: 1.1rem;"><strong>{value}</strong></p>
                    <p style="margin: 0.3rem 0; font-size: 0.9rem; color: #666;">{status}</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )


# ============================================================================
# PAGE: BATCH ANALYSIS
# ============================================================================
def show_batch_analysis(models, scalers, features, encoders):
    st.markdown("## 📊 Batch Water Quality Analysis")

    st.info("""
    **📋 Upload CSV file** with water quality data for multiple samples.
    The system will process all records and provide comprehensive analysis.
    """)

    st.markdown("### 📥 Required CSV Format")
    st.write("Your CSV should contain these columns:")

    required_cols = [
        "state",
        "location_type",
        "water_source",
        "season",
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

    st.code(", ".join(required_cols), language="text")

    uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)

            st.success(f"✅ Loaded {len(df)} water samples")

            with st.expander("👁️ Preview Data (First 10 rows)", expanded=True):
                st.dataframe(df.head(10), use_container_width=True)

            # Check columns
            missing = set(required_cols) - set(df.columns)

            if missing:
                st.error(f"❌ Missing columns: {', '.join(missing)}")
            else:
                if st.button("🔬 Analyze All Samples", use_container_width=True):
                    with st.spinner("Processing all water samples..."):
                        # Encode categorical variables
                        for col in ["state", "location_type", "water_source", "season"]:
                            df[f"{col}_encoded"] = encoders[col].transform(df[col])

                        # Prepare features
                        X = df[features]

                        # Predict for all diseases
                        for disease in [
                            "cholera",
                            "typhoid",
                            "dysentery",
                            "hepatitis_a",
                            "overall",
                        ]:
                            scaled = scalers[disease].transform(X)
                            preds = models[disease].predict(scaled)
                            probs = models[disease].predict_proba(scaled)

                            df[f"{disease}_risk"] = [
                                "High" if p == 1 else "Low" for p in preds
                            ]
                            df[f"{disease}_score"] = [prob[1] * 100 for prob in probs]

                        st.success("✅ Analysis complete!")

                        # Summary
                        st.markdown("### 📊 Summary Statistics")

                        col1, col2, col3, col4, col5 = st.columns(5)

                        with col1:
                            st.metric("Total Samples", len(df))
                        with col2:
                            cholera_high = (df["cholera_risk"] == "High").sum()
                            st.metric(
                                "Cholera Risk",
                                cholera_high,
                                f"{cholera_high / len(df) * 100:.1f}%",
                            )
                        with col3:
                            typhoid_high = (df["typhoid_risk"] == "High").sum()
                            st.metric(
                                "Typhoid Risk",
                                typhoid_high,
                                f"{typhoid_high / len(df) * 100:.1f}%",
                            )
                        with col4:
                            dysentery_high = (df["dysentery_risk"] == "High").sum()
                            st.metric(
                                "Dysentery Risk",
                                dysentery_high,
                                f"{dysentery_high / len(df) * 100:.1f}%",
                            )
                        with col5:
                            hepatitis_high = (df["hepatitis_a_risk"] == "High").sum()
                            st.metric(
                                "Hepatitis A Risk",
                                hepatitis_high,
                                f"{hepatitis_high / len(df) * 100:.1f}%",
                            )

                        # Results table
                        st.markdown("### 📋 Detailed Results")
                        st.dataframe(df, use_container_width=True)

                        # Visualizations
                        st.markdown("### 📈 Risk Analysis Charts")

                        col1, col2 = st.columns(2)

                        with col1:
                            # Overall risk pie chart
                            overall_counts = df["overall_risk"].value_counts()
                            fig = px.pie(
                                values=overall_counts.values,
                                names=overall_counts.index,
                                title="Overall Risk Distribution",
                                color=overall_counts.index,
                                color_discrete_map={
                                    "High": "#f44336",
                                    "Low": "#4caf50",
                                },
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        with col2:
                            # Disease-wise comparison
                            disease_high = {
                                "Cholera": cholera_high,
                                "Typhoid": typhoid_high,
                                "Dysentery": dysentery_high,
                                "Hepatitis A": hepatitis_high,
                            }
                            fig = px.bar(
                                x=list(disease_high.keys()),
                                y=list(disease_high.values()),
                                title="High-Risk Cases by Disease",
                                labels={"x": "Disease", "y": "Number of Cases"},
                                color=list(disease_high.values()),
                                color_continuous_scale="Reds",
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        # Download results
                        st.markdown("### 📥 Download Results")

                        csv = df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "📄 Download Full Results CSV",
                            csv,
                            f"water_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            "text/csv",
                            use_container_width=True,
                        )

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")


# ============================================================================
# PAGE: MODEL PERFORMANCE
# ============================================================================
def show_model_performance(metadata):
    st.markdown("## 📈 Model Performance & Accuracy")

    if not metadata or "results" not in metadata:
        st.warning("No performance data available")
        return

    results_df = pd.DataFrame(metadata["results"])

    # Overall statistics
    st.markdown("### 📊 Overall Performance Metrics")

    col1, col2, col3, col4, col5 = st.columns(5)

    avg_acc = results_df["Accuracy"].mean()
    avg_prec = results_df["Precision"].mean()
    avg_rec = results_df["Recall"].mean()
    avg_f1 = results_df["F1-Score"].mean()
    avg_auc = results_df["ROC-AUC"].mean()

    with col1:
        st.metric("Avg Accuracy", f"{avg_acc * 100:.2f}%")
    with col2:
        st.metric("Avg Precision", f"{avg_prec * 100:.2f}%")
    with col3:
        st.metric("Avg Recall", f"{avg_rec * 100:.2f}%")
    with col4:
        st.metric("Avg F1-Score", f"{avg_f1 * 100:.2f}%")
    with col5:
        st.metric("Avg ROC-AUC", f"{avg_auc:.3f}")

    # Performance table
    st.markdown("### 📋 Detailed Performance by Disease")

    display_df = results_df.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1-Score"]:
        display_df[col] = display_df[col].apply(lambda x: f"{x * 100:.2f}%")
    display_df["ROC-AUC"] = display_df["ROC-AUC"].apply(lambda x: f"{x:.4f}")

    st.dataframe(display_df, use_container_width=True)

    # Charts
    st.markdown("### 📊 Performance Visualizations")

    col1, col2 = st.columns(2)

    with col1:
        # Grouped bar chart
        fig = go.Figure()

        metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
        colors = ["#0066cc", "#00cc99", "#ff9800", "#e91e63"]

        for metric, color in zip(metrics, colors):
            fig.add_trace(
                go.Bar(
                    name=metric,
                    x=results_df["Disease"],
                    y=results_df[metric] * 100,
                    marker_color=color,
                )
            )

        fig.update_layout(
            title="Model Metrics Comparison",
            xaxis_title="Disease",
            yaxis_title="Score (%)",
            barmode="group",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # ROC-AUC radar chart
        fig = go.Figure()

        fig.add_trace(
            go.Scatterpolar(
                r=results_df["ROC-AUC"] * 100,
                theta=results_df["Disease"],
                fill="toself",
                name="ROC-AUC Score",
            )
        )

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            title="ROC-AUC Scores by Disease",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Model info
    st.markdown("---")
    st.markdown("### ℹ️ Model Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info(f"""
        **Model Type**
        {metadata.get("model_type", "XGBoost Classifier")}
        """)

    with col2:
        st.info(f"""
        **Training Date**
        {metadata.get("training_date", "N/A")}
        """)

    with col3:
        st.info(f"""
        **Dataset Size**
        {metadata.get("dataset_size", "N/A"):,} records
        """)


# ============================================================================
# PAGE: TEST HISTORY
# ============================================================================
def show_test_history():
    st.markdown("## 📜 Water Quality Test History")

    if not st.session_state.test_history:
        st.info(
            "No test history available yet. Conduct some water quality tests to see them here!"
        )
        return

    df = pd.DataFrame(st.session_state.test_history)

    st.metric("Total Tests Conducted", len(df))

    # Display history
    st.markdown("### 📊 Recent Tests")
    st.dataframe(df.sort_values("timestamp", ascending=False), use_container_width=True)

    # Analytics
    st.markdown("### 📈 History Analytics")

    col1, col2 = st.columns(2)

    with col1:
        # Risk distribution
        fig = px.histogram(
            df,
            x="overall_risk",
            title="Risk Score Distribution",
            labels={"overall_risk": "Risk Score (%)"},
            nbins=20,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # State distribution
        state_counts = df["state"].value_counts()
        fig = px.pie(
            values=state_counts.values, names=state_counts.index, title="Tests by State"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Clear history
    if st.button("🗑️ Clear All History"):
        st.session_state.test_history = []
        st.success("History cleared!")
        st.rerun()


# ============================================================================
# PAGE: ABOUT
# ============================================================================
def show_about(metadata):
    st.markdown("## ℹ️ About the System")

    st.markdown("""
    ### Smart Community Health Monitoring and Early Warning System

    **For Water-Borne Diseases in Rural Northeast India**

    This AI-powered system helps communities monitor water quality and predict outbreak risks
    for four major waterborne diseases affecting Northeast India.

    ---

    ### 🎯 Key Features

    ✅ **Real-time Water Quality Analysis** - Test water samples instantly
    ✅ **AI Disease Prediction** - Predict 4 waterborne diseases using ML
    ✅ **Early Warning Alerts** - Get outbreak warnings before they spread
    ✅ **Batch Processing** - Analyze multiple water samples at once
    ✅ **Performance Tracking** - View model accuracy and metrics
    ✅ **Test History** - Track all water quality tests over time

    ---

    ### 🦠 Monitored Diseases

    1. **Cholera** - Acute diarrheal infection (Vibrio cholerae)
    2. **Typhoid** - Bacterial infection (Salmonella typhi)
    3. **Dysentery** - Intestinal inflammation causing bloody diarrhea
    4. **Hepatitis A** - Viral liver infection

    ---

    ### 🛠️ Technology Stack

    - **Machine Learning**: XGBoost Classifier
    - **Frontend**: Streamlit
    - **Data Processing**: Pandas, NumPy
    - **Visualization**: Plotly
    - **Class Balancing**: SMOTE

    ---

    ### 📊 System Performance
    """)

    if metadata and "results" in metadata:
        results_df = pd.DataFrame(metadata["results"])
        avg_acc = results_df["Accuracy"].mean() * 100

        st.success(f"""
        **Average Model Accuracy: {avg_acc:.2f}%**

        Trained on {metadata.get("dataset_size", "N/A"):,} water quality records from Northeast India.
        """)

    st.markdown(
        """
    ---

    ### ⚠️ Important Disclaimer

    This system is designed as a **decision support tool** for community health workers and
    water quality monitoring. It should **NOT** replace professional laboratory testing or
    medical diagnosis.

    **Always consult:**
    - Certified water testing laboratories for official results
    - Healthcare professionals for medical concerns
    - Local health authorities for outbreak management

    ---

    ### 📧 Contact & Support

    For technical support, training, or deployment assistance:
    - Email: support@waterhealth.gov.in
    - Emergency: 1800-XXX-XXXX (Toll-free)

    ---

    ### 📝 Version Information

    - **Version**: 1.0.0
    - **Release Date**: October 2025
    - **Last Updated**: """
        + datetime.now().strftime("%B %d, %Y")
        + """
    - **Region**: Northeast India (7 states)
    - **Languages**: English

    ---

    **Developed for rural communities of Northeast India** 🌏💧
    """
    )


# ============================================================================
# RUN APP
# ============================================================================
if __name__ == "__main__":
    main()
