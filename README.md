# 🏥 Multi-Disease Prediction System

AI-powered disease risk prediction using machine learning. This project includes two Streamlit web applications:

| App | Command | Predicts |
|-----|---------|----------|
| 💧 **Water-Borne Disease Warning** | `streamlit run app.py` | Cholera, Typhoid, Dysentery, Hepatitis A |
| 🏥 **Multi-Disease Prediction** | `streamlit run app_multidisease.py` | Diabetes, Heart Disease, Hypertension, Stroke |

## 📁 Project Structure

```
multi-disease-prediction/
├── app.py                            # Water-borne disease Streamlit app
├── app_multidisease.py               # Multi-disease health Streamlit app
├── create_dataset.py                 # Generate water quality dataset
├── create_and_train.py               # Train water-disease models
├── create_and_train_multidisease.py  # Train multi-disease models
├── requirements.txt                  # Python dependencies
├── data/                             # Datasets
│   ├── raw/                          # Raw training data
│   ├── water_disease_dataset.csv
│   └── multidisease_dataset.csv
├── models/                           # Pre-trained ML models (.pkl)
├── reports/                          # Training reports & confusion matrices
└── notebooks/                        # Jupyter notebooks
```

## 🚀 Setup & Run

```bash
# 0. Clone the Repo
git clone https://github.com/Silver595/multi-disease-prediction.git
cd multi-disease-prediction

# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run either app
streamlit run app.py                  # Water-borne disease app
streamlit run app_multidisease.py     # Multi-disease app
```

> **Note:** Pre-trained models are already included in `models/`. No retraining needed.

## 🔄 Retraining Models (Optional)

```bash
# Water-borne disease models
python create_dataset.py        # Generate dataset
python create_and_train.py      # Train models

# Multi-disease models
python create_and_train_multidisease.py   # Generates data + trains models
```

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **ML:** XGBoost, scikit-learn, imbalanced-learn (SMOTE)
- **Data:** pandas, NumPy
- **Visualization:** Plotly, Matplotlib, Seaborn
