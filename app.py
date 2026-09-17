from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "insurance_model.pkl"
METADATA_PATH = BASE_DIR / "model" / "metadata.json"


st.set_page_config(
    page_title="Medical Insurance Cost Predictor",
    page_icon="🩺",
    layout="centered",
)


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return model, metadata["feature_order"]


model, feature_order = load_model()


def predict_insurance(age, bmi, children, gender, smoker, region):
    row = {
        "age": int(age),
        "bmi": float(bmi),
        "children": int(children),
        "sex_male": 1 if gender == "Male" else 0,
        "smoker_yes": 1 if smoker == "Yes" else 0,
        "region_northwest": 1 if region == "Northwest" else 0,
        "region_southeast": 1 if region == "Southeast" else 0,
        "region_southwest": 1 if region == "Southwest" else 0,
    }

    input_df = pd.DataFrame([row], columns=feature_order)
    return float(model.predict(input_df)[0])


st.title("Medical Insurance Cost Predictor")
st.write(
    "Enter a customer profile below to estimate medical insurance charges "
    "using the Linear Regression model trained in this project."
)

with st.form("insurance_form"):
    left, right = st.columns(2)

    with left:
        age = st.number_input("Age", min_value=18, max_value=64, value=30, step=1)
        bmi = st.number_input(
            "BMI", min_value=15.0, max_value=55.0, value=25.0, step=0.1
        )
        children = st.number_input(
            "Number of children", min_value=0, max_value=5, value=0, step=1
        )

    with right:
        gender = st.selectbox("Gender", ["Female", "Male"])
        smoker = st.selectbox("Smoker", ["No", "Yes"])
        region = st.selectbox(
            "Region", ["Northeast", "Northwest", "Southeast", "Southwest"]
        )

    submitted = st.form_submit_button(
        "Estimate insurance cost", use_container_width=True
    )

if submitted:
    prediction = predict_insurance(age, bmi, children, gender, smoker, region)

    st.subheader("Prediction")
    st.metric("Estimated insurance cost", f"{prediction:,.2f}")
    st.caption(
        "This is a learning project based on the Medical Cost Personal Dataset. "
        "The result is a model estimate, not a real insurance quote."
    )

with st.expander("About this model"):
    st.write(
        "The model is a Multiple Linear Regression baseline trained on age, BMI, "
        "number of children, gender, smoking status, and region."
    )
    st.write("Test-set R²: 0.8069 · MAE: 4,177.05 · RMSE: 5,956.34")
