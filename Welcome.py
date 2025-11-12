import streamlit as st
import joblib

# ----------------------------------------------------------
# MAIN APP CONFIGURATION
# ----------------------------------------------------------
st.set_page_config(page_title="EMIPredict AI", page_icon="💰", layout="wide")

st.title("💰 EMIPredict AI (Developed By Faiz Hussain)")
st.markdown("### Intelligent Financial Risk Assessment Platform")
st.write("Use the sidebar to navigate between pages to predict EMI Eligibility or calculate Maximum EMI.")

# ----------------------------------------------------------
# LOAD MODELS AND SCALERS (shared across pages)
# ----------------------------------------------------------
@st.cache_resource
def load_models():
    classification_model = joblib.load("best_classification_model.pkl")
    regression_model = joblib.load("best_regression_model.pkl")
    scaler_clf = joblib.load("scaler_classification.pkl")
    scaler_reg = joblib.load("scaler_regression.pkl")
    return classification_model, regression_model, scaler_clf, scaler_reg

if "models" not in st.session_state:
    st.session_state.models = load_models()

st.info("✅ Models loaded successfully! Navigate using the sidebar.")


