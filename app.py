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
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at top right, rgba(245, 158, 11, 0.10), transparent 28%),
                radial-gradient(circle at top left, rgba(14, 116, 144, 0.08), transparent 24%),
                #fbfcfe;
        }

        .block-container {
            max-width: 1040px;
            padding-top: 3.2rem;
            padding-bottom: 3rem;
        }

        .hero {
            padding: 0.4rem 0 1.6rem 0;
        }

        .eyebrow {
            display: inline-block;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            background: #eef8fa;
            color: #0f6f78;
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            margin-bottom: 0.9rem;
        }

        .hero h1 {
            font-size: clamp(2.15rem, 5vw, 3.65rem);
            line-height: 1.03;
            letter-spacing: -0.04em;
            margin: 0;
            color: #162033;
        }

        .hero p {
            max-width: 760px;
            margin-top: 0.95rem;
            margin-bottom: 0;
            color: #607086;
            font-size: 1.05rem;
            line-height: 1.7;
        }

        div[data-testid="stForm"] {
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid #e5eaf0;
            border-radius: 22px;
            padding: 1.35rem 1.4rem 1.1rem 1.4rem;
            box-shadow: 0 18px 48px rgba(21, 33, 52, 0.07);
        }

        div[data-baseweb="input"] > div,
        div[data-baseweb="select"] > div {
            border-radius: 12px !important;
        }

        div.stButton > button,
        div[data-testid="stFormSubmitButton"] > button {
            width: 100%;
            min-height: 3.2rem;
            border: 0;
            border-radius: 13px;
            background: linear-gradient(135deg, #f59e0b 0%, #ea580c 100%);
            color: #ffffff;
            font-weight: 750;
            font-size: 1rem;
            box-shadow: 0 10px 24px rgba(234, 88, 12, 0.20);
            transition: all 0.18s ease;
        }

        div.stButton > button:hover,
        div[data-testid="stFormSubmitButton"] > button:hover {
            color: #ffffff;
            transform: translateY(-1px);
            box-shadow: 0 13px 28px rgba(234, 88, 12, 0.26);
        }

        .result-wrap {
            margin-top: 1.15rem;
            padding: 1.4rem 1.5rem;
            border-radius: 20px;
            background: linear-gradient(135deg, #fff8ec 0%, #fff2e3 100%);
            border: 1px solid #fed7aa;
            box-shadow: 0 16px 38px rgba(120, 53, 15, 0.08);
        }

        .result-kicker {
            color: #9a4c11;
            font-size: 0.86rem;
            font-weight: 750;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }

        .result-value {
            margin-top: 0.25rem;
            color: #172033;
            font-size: clamp(2.25rem, 6vw, 3.35rem);
            line-height: 1.05;
            font-weight: 850;
            letter-spacing: -0.035em;
        }

        .result-note {
            color: #6f5a49;
            margin-top: 0.65rem;
            font-size: 0.92rem;
            line-height: 1.55;
        }

        .section-title {
            margin-top: 2.2rem;
            margin-bottom: 0.25rem;
            color: #172033;
            font-size: 1.2rem;
            font-weight: 800;
        }

        .section-copy {
            color: #69788d;
            margin-bottom: 0.95rem;
        }

        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.86);
            border: 1px solid #e7ebf0;
            padding: 0.85rem 1rem;
            border-radius: 16px;
            box-shadow: 0 8px 22px rgba(21, 33, 52, 0.04);
        }

        div[data-testid="stExpander"] {
            border: 1px solid #e5eaf0;
            border-radius: 15px;
            background: rgba(255,255,255,0.72);
        }

        .footer-note {
            color: #8a96a8;
            text-align: center;
            font-size: 0.82rem;
            margin-top: 2.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
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


st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Machine Learning Project</div>
        <h1>Medical Insurance<br>Cost Predictor</h1>
        <p>
            Enter a few customer details to estimate medical insurance charges
            using the Multiple Linear Regression model built for this project.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.form("insurance_form"):
    left, right = st.columns(2, gap="large")

    with left:
        age = st.number_input("Age", min_value=18, max_value=64, value=25, step=1)
        bmi = st.number_input(
            "BMI", min_value=15.0, max_value=55.0, value=22.5, step=0.1, format="%.2f"
        )
        children = st.number_input(
            "Number of children", min_value=0, max_value=5, value=0, step=1
        )

    with right:
        gender = st.selectbox("Gender", ["Female", "Male"])
        smoker = st.selectbox("Smoking status", ["No", "Yes"])
        region = st.selectbox(
            "Region", ["Northeast", "Northwest", "Southeast", "Southwest"]
        )

    submitted = st.form_submit_button("Estimate insurance cost", use_container_width=True)

if submitted:
    prediction = predict_insurance(age, bmi, children, gender, smoker, region)
    st.markdown(
        f"""
        <div class="result-wrap">
            <div class="result-kicker">Estimated insurance cost</div>
            <div class="result-value">${prediction:,.2f}</div>
            <div class="result-note">
                This is a model estimate created for a learning project and is not a real insurance quote.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="section-title">Model at a glance</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-copy">A quick summary of how the baseline model performed on unseen test data.</div>',
    unsafe_allow_html=True,
)

m1, m2, m3 = st.columns(3)
with m1:
    st.metric("R² score", "0.8069")
with m2:
    st.metric("MAE", "4,177.05")
with m3:
    st.metric("RMSE", "5,956.34")

with st.expander("About the model"):
    st.write(
        "This project uses Multiple Linear Regression with age, BMI, number of children, "
        "gender, smoking status, and region as inputs. Smoking status showed the strongest "
        "linear relationship with insurance charges during exploratory analysis."
    )
    st.caption(
        "Dataset: Medical Cost Personal Dataset · Final cleaned records: 1,337 · "
        "Train/test split: 80/20"
    )

st.markdown(
    '<div class="footer-note">Medical Insurance Cost Prediction · Learning Project</div>',
    unsafe_allow_html=True,
)
