from pathlib import Path

import joblib
import numpy as np
import streamlit as st


st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon=":credit_card:",
    layout="centered",
)


MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()


st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f8fafc 0%, #eef2f7 100%);
        color: #172033;
    }

    .block-container {
        max-width: 860px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 26px 28px;
        margin-bottom: 20px;
        box-shadow: 0 16px 38px rgba(15, 23, 42, 0.08);
        border-top: 6px solid #2563eb;
    }

    .hero h1 {
        color: #172033;
        font-size: 38px;
        font-weight: 850;
        line-height: 1.15;
        margin: 0;
        text-align: center;
    }

    .hero p {
        color: #64748b;
        font-size: 16px;
        line-height: 1.65;
        margin: 12px auto 0;
        max-width: 640px;
        text-align: center;
    }

    .input-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.07);
    }

    .section-title {
        color: #172033;
        font-size: 21px;
        font-weight: 800;
        margin-bottom: 4px;
    }

    .section-copy {
        color: #64748b;
        font-size: 14px;
        line-height: 1.55;
        margin-bottom: 18px;
    }

    div[data-testid="stNumberInput"] label {
        color: #334155;
        font-weight: 700;
    }

    .stButton > button {
        width: 100%;
        min-height: 48px;
        background: #2563eb;
        color: #ffffff;
        border: none;
        border-radius: 10px;
        font-size: 17px;
        font-weight: 800;
        box-shadow: 0 10px 22px rgba(37, 99, 235, 0.24);
    }

    .stButton > button:hover {
        background: #1d4ed8;
        color: #ffffff;
    }

    .result-safe,
    .result-risk {
        border-radius: 12px;
        margin-top: 20px;
        padding: 22px;
        text-align: center;
        font-size: 24px;
        font-weight: 900;
    }

    .result-safe {
        background: #dcfce7;
        border: 1px solid #86efac;
        color: #166534;
    }

    .result-risk {
        background: #fee2e2;
        border: 1px solid #fca5a5;
        color: #991b1b;
    }

    .result-note {
        font-size: 14px;
        font-weight: 600;
        margin-top: 8px;
        opacity: 0.8;
    }

    footer,
    #MainMenu {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <div class="hero">
        <h1>Credit Card Fraud Detection</h1>
        <p>
            Enter the transaction amount and the five model features below.
            The app will classify the transaction as legitimate or potentially fraudulent.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


st.markdown('<div class="input-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Transaction Details</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-copy">Amount is the transaction value. V1 to V5 are numeric model features from the dataset.</div>',
    unsafe_allow_html=True,
)

amount = st.number_input(
    "Transaction Amount",
    min_value=0.0,
    value=0.0,
    step=10.0,
    help="Enter the transaction amount.",
)

feature_col_1, feature_col_2 = st.columns(2)

with feature_col_1:
    v1 = st.number_input("Feature V1", value=0.0, help="Numeric model feature V1.")
    v3 = st.number_input("Feature V3", value=0.0, help="Numeric model feature V3.")
    v5 = st.number_input("Feature V5", value=0.0, help="Numeric model feature V5.")

with feature_col_2:
    v2 = st.number_input("Feature V2", value=0.0, help="Numeric model feature V2.")
    v4 = st.number_input("Feature V4", value=0.0, help="Numeric model feature V4.")

predict_clicked = st.button("Check Transaction")
st.markdown("</div>", unsafe_allow_html=True)


if predict_clicked:
    input_data = np.array([[amount, v1, v2, v3, v4, v5]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.markdown(
            """
            <div class="result-risk">
                Fraudulent Transaction Detected
                <div class="result-note">Review this transaction before approval.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="result-safe">
                Legitimate Transaction
                <div class="result-note">The model classified this transaction as normal.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
