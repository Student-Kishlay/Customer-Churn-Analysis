"""
Customer Churn Prediction – Streamlit Demo
==========================================
Run with:
    streamlit run app/streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="Telecom Churn Predictor",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# Paths (works whether you run from root or app/)
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.pkl"
FEATURE_NAMES_PATH = BASE_DIR / "models" / "feature_names.pkl"
CLEANED_DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_churn_data.csv"

# -------------------------------------------------
# Load Artifacts (cached)
# -------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = None
    preprocessor = None
    feature_names = None
    cleaned_df = None

    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
    if PREPROCESSOR_PATH.exists():
        preprocessor = joblib.load(PREPROCESSOR_PATH)
    if FEATURE_NAMES_PATH.exists():
        feature_names = joblib.load(FEATURE_NAMES_PATH)
    if CLEANED_DATA_PATH.exists():
        cleaned_df = pd.read_csv(CLEANED_DATA_PATH)

    return model, preprocessor, feature_names, cleaned_df


model, preprocessor, feature_names, cleaned_df = load_artifacts()

# -------------------------------------------------
# Helper: Feature Engineering (must match notebook 02)
# -------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # TenureGroup
    def tenure_group(t):
        if t <= 12:
            return "0-12 months"
        elif t <= 24:
            return "13-24 months"
        elif t <= 48:
            return "25-48 months"
        else:
            return "49+ months"

    df["TenureGroup"] = df["tenure"].apply(tenure_group)

    # AvgMonthlyCharges
    df["AvgMonthlyCharges"] = np.where(
        df["tenure"] > 0,
        df["TotalCharges"] / df["tenure"],
        df["MonthlyCharges"]
    )

    # NumServices
    service_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    # Only count if column exists and value is "Yes"
    existing_services = [c for c in service_cols if c in df.columns]
    df["NumServices"] = df[existing_services].apply(
        lambda x: (x == "Yes").sum(), axis=1
    )

    # PhoneAndInternet
    df["PhoneAndInternet"] = np.where(
        (df["PhoneService"] == "Yes") & (df["InternetService"] != "No"), 1, 0
    )

    # SeniorWithDependents
    df["SeniorWithDependents"] = np.where(
        (df["SeniorCitizen"] == 1) & (df["Dependents"] == "Yes"), 1, 0
    )

    return df


def risk_level(proba: float) -> str:
    if proba < 0.30:
        return "Low"
    elif proba < 0.60:
        return "Medium"
    else:
        return "High"


RECOMMENDATION_BY_RISK = {
    "High": "Immediate outreach",
    "Medium": "Monitor closely",
    "Low": "Standard care",
}


def predict_churn(customer_dict: dict):
    """Return probability and risk level for one customer."""
    if model is None or preprocessor is None:
        return None, None

    input_df = pd.DataFrame([customer_dict])
    input_df = engineer_features(input_df)

    # Ensure all columns expected by preprocessor exist
    # (preprocessor was fitted on the training columns)
    processed = preprocessor.transform(input_df)
    proba = model.predict_proba(processed)[0, 1]

    return float(proba), risk_level(proba)


# Raw columns the model expects (same schema as the Telco dataset,
# minus customerID/Churn which are identifiers/target, not features)
REQUIRED_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]


def clean_uploaded_data(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce raw uploaded columns to the types notebook 02 produces."""
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")
    df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce").fillna(0).astype(int)
    return df


def predict_batch(df: pd.DataFrame) -> np.ndarray:
    """Return churn probabilities for every row in df."""
    engineered = engineer_features(df)
    processed = preprocessor.transform(engineered)
    return model.predict_proba(processed)[:, 1]


# -------------------------------------------------
# Sidebar – Navigation
# -------------------------------------------------
st.sidebar.title("📉 Churn Predictor")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "🔮 Predict Churn", "📁 Batch Upload", "📊 Insights Dashboard", "ℹ️ About"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "This app uses a trained ML model to estimate the probability that a telecom customer will churn."
)

# -------------------------------------------------
# PAGE 1: HOME
# -------------------------------------------------
if page == "🏠 Home":
    st.title("Telecom Customer Churn Prediction")
    st.markdown("### Identify at-risk customers and take action before they leave.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Overall Churn Rate", "≈ 26.5%", help="Typical rate in the Telco dataset")
    with col2:
        st.metric("Best Focus Area", "Month-to-month contracts")
    with col3:
        st.metric("High-Risk Signal", "Short tenure + Fiber optic")

    st.markdown("---")
    st.subheader("Why this matters")
    st.write(
        """
        Acquiring a new customer costs **5–7× more** than retaining an existing one.  
        This tool helps you:
        - Score individual customers for churn risk
        - Understand the main drivers of churn
        - Prioritize retention efforts on the highest-risk segment
        """
    )

    st.subheader("How to use")
    st.markdown(
        """
        1. Go to **🔮 Predict Churn**
        2. Fill in the customer details
        3. Click **Predict** to get probability + risk level
        4. Explore **📊 Insights Dashboard** for overall patterns
        """
    )

    if model is None:
        st.warning(
            "⚠️ Model artifacts not found. Please run notebooks 02 and 03 first "
            "so that `models/best_model.pkl` and `models/preprocessor.pkl` exist."
        )
    else:
        st.success("✅ Model and preprocessor loaded successfully.")

# -------------------------------------------------
# PAGE 2: PREDICT CHURN
# -------------------------------------------------
elif page == "🔮 Predict Churn":
    st.title("🔮 Predict Customer Churn")
    st.markdown("Enter customer details to get a churn probability and risk level.")

    if model is None or preprocessor is None:
        st.error(
            "Model or preprocessor not found. "
            "Run the preprocessing and modeling notebooks first."
        )
        st.stop()

    with st.form("prediction_form"):
        st.subheader("Customer Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)

        with col2:
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])

        with col3:
            device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

        col4, col5 = st.columns(2)
        with col4:
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment = st.selectbox(
                "Payment Method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
            )
        with col5:
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=1.0)
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=500.0, step=10.0)

        submitted = st.form_submit_button("🚀 Predict Churn Risk", use_container_width=True)

    if submitted:
        customer = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }

        proba, risk = predict_churn(customer)

        if proba is None:
            st.error("Prediction failed. Check model artifacts.")
        else:
            st.markdown("---")
            st.subheader("Prediction Result")

            # Color coding
            if risk == "High":
                color = "#e74c3c"
                emoji = "🔴"
            elif risk == "Medium":
                color = "#f39c12"
                emoji = "🟠"
            else:
                color = "#2ecc71"
                emoji = "🟢"

            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Churn Probability", f"{proba*100:.1f}%")
            with m2:
                st.markdown(f"### {emoji} Risk Level: **:{color}[{risk}]**")
            with m3:
                st.metric("Recommendation", RECOMMENDATION_BY_RISK[risk])

            # Progress bar style visual
            st.progress(proba)

            # Simple interpretation
            st.markdown("#### Interpretation")
            if risk == "High":
                st.warning(
                    "This customer is in the **high-risk** segment. "
                    "Consider proactive retention offers (contract upgrade, support bundle, or loyalty discount)."
                )
            elif risk == "Medium":
                st.info(
                    "Medium risk. Monitor usage and satisfaction. "
                    "A light-touch offer (e.g., free add-on for 1–2 months) may help."
                )
            else:
                st.success(
                    "Low risk. Focus on maintaining satisfaction and exploring upsell opportunities."
                )

# -------------------------------------------------
# PAGE 3: BATCH UPLOAD
# -------------------------------------------------
elif page == "📁 Batch Upload":
    st.title("📁 Batch Churn Prediction")
    st.markdown("Upload a CSV of customers to score all of them at once.")

    if model is None or preprocessor is None:
        st.error(
            "Model or preprocessor not found. "
            "Run the preprocessing and modeling notebooks first."
        )
        st.stop()

    with st.expander("ℹ️ Expected file format"):
        st.markdown(
            "Your CSV needs these columns (an optional `customerID` is kept "
            "for reference; any `Churn` column is ignored):"
        )
        st.code(", ".join(REQUIRED_COLUMNS))

        template = pd.DataFrame([{
            "customerID": "0000-EXAMPLE",
            "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
            "tenure": 12, "PhoneService": "Yes", "MultipleLines": "No",
            "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
            "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
            "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check", "MonthlyCharges": 70.35, "TotalCharges": 845.5,
        }])
        st.download_button(
            "⬇️ Download template CSV",
            data=template.to_csv(index=False).encode("utf-8"),
            file_name="churn_upload_template.csv",
            mime="text/csv",
        )

    uploaded_file = st.file_uploader("Upload customer CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read that file as CSV: {e}")
            st.stop()

        missing = [c for c in REQUIRED_COLUMNS if c not in raw_df.columns]
        if missing:
            st.error(f"Missing required column(s): {', '.join(missing)}")
            st.stop()

        with st.spinner(f"Scoring {len(raw_df):,} customers..."):
            clean_df = clean_uploaded_data(raw_df)
            probabilities = predict_batch(clean_df)

        results = raw_df.copy()
        results["ChurnProbability"] = probabilities
        results["RiskLevel"] = [risk_level(p) for p in probabilities]
        results["Recommendation"] = results["RiskLevel"].map(RECOMMENDATION_BY_RISK)
        results = results.sort_values("ChurnProbability", ascending=False)

        st.markdown("---")
        st.subheader("Summary")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Customers Scored", f"{len(results):,}")
        with c2:
            st.metric("High Risk", f"{(results['RiskLevel'] == 'High').sum():,}")
        with c3:
            st.metric("Medium Risk", f"{(results['RiskLevel'] == 'Medium').sum():,}")
        with c4:
            st.metric("Avg. Churn Probability", f"{results['ChurnProbability'].mean()*100:.1f}%")

        st.bar_chart(results["RiskLevel"].value_counts())

        st.markdown("---")
        st.subheader("Results")
        st.dataframe(results, use_container_width=True)

        st.download_button(
            "⬇️ Download results as CSV",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="churn_predictions.csv",
            mime="text/csv",
        )

        # ---------------------------------------------
        # Summary & Prevention Recommendations
        # ---------------------------------------------
        st.markdown("---")
        st.subheader("📝 Summary & Prevention Recommendations")

        high_risk_df = results[results["RiskLevel"] == "High"]

        if high_risk_df.empty:
            st.success("No high-risk customers in this batch — no urgent action needed.")
        else:
            pct_high = len(high_risk_df) / len(results) * 100
            st.markdown(
                f"**{len(high_risk_df):,} of {len(results):,} customers ({pct_high:.1f}%) "
                "are at high risk of churning.** Patterns driving that risk in this batch:"
            )

            drivers = []
            if "Contract" in high_risk_df.columns:
                mtm_pct = (high_risk_df["Contract"] == "Month-to-month").mean()
                if mtm_pct >= 0.3:
                    drivers.append((
                        f"{mtm_pct*100:.0f}% are on **month-to-month contracts**",
                        "Offer a discount for switching to a 1–2 year contract.",
                    ))
            if "tenure" in high_risk_df.columns:
                short_tenure_pct = (pd.to_numeric(high_risk_df["tenure"], errors="coerce") <= 12).mean()
                if short_tenure_pct >= 0.3:
                    drivers.append((
                        f"{short_tenure_pct*100:.0f}% have **tenure ≤ 12 months**",
                        "Prioritize early-lifecycle check-ins and onboarding support.",
                    ))
            if {"InternetService", "TechSupport"}.issubset(high_risk_df.columns):
                fiber_no_support_pct = (
                    (high_risk_df["InternetService"] == "Fiber optic")
                    & (high_risk_df["TechSupport"] != "Yes")
                ).mean()
                if fiber_no_support_pct >= 0.3:
                    drivers.append((
                        f"{fiber_no_support_pct*100:.0f}% have **Fiber optic with no Tech Support**",
                        "Bundle free or discounted tech support for 3–6 months.",
                    ))
            if "PaymentMethod" in high_risk_df.columns:
                echeck_pct = (high_risk_df["PaymentMethod"] == "Electronic check").mean()
                if echeck_pct >= 0.3:
                    drivers.append((
                        f"{echeck_pct*100:.0f}% pay via **Electronic check**",
                        "Incentivize switching to auto-pay (card or bank transfer).",
                    ))
            if "MonthlyCharges" in high_risk_df.columns:
                avg_high = pd.to_numeric(high_risk_df["MonthlyCharges"], errors="coerce").mean()
                avg_all = pd.to_numeric(results["MonthlyCharges"], errors="coerce").mean()
                if avg_all > 0 and avg_high >= avg_all * 1.1:
                    drivers.append((
                        f"average **Monthly Charges** (${avg_high:.0f}) is well above the batch average (${avg_all:.0f})",
                        "Consider a personalized loyalty discount for high-paying, high-risk customers.",
                    ))

            if drivers:
                for finding, action in drivers:
                    st.markdown(f"- **Finding:** {finding} → **Prevention:** {action}")
            else:
                st.info(
                    "Risk in this batch is spread across varied profiles with no single "
                    "dominant pattern — review the flagged customers individually below."
                )

        st.markdown("#### General Prevention Playbook")
        st.markdown(
            """
            | Risk Level | Recommended Action |
            |------------|---------------------|
            | 🔴 High | Immediate outreach with a retention offer (contract upgrade, discount, or free service bundle) |
            | 🟠 Medium | Monitor closely; light-touch engagement (loyalty perks, satisfaction check-in) |
            | 🟢 Low | Standard care; focus on upsell and satisfaction opportunities |
            """
        )

        with st.expander("🔍 View high-risk customers"):
            st.dataframe(high_risk_df, use_container_width=True)

# -------------------------------------------------
# PAGE 4: INSIGHTS DASHBOARD
# -------------------------------------------------
elif page == "📊 Insights Dashboard":
    st.title("📊 Churn Insights Dashboard")

    if cleaned_df is None:
        st.warning(
            "Cleaned dataset not found. Run notebook 02 first "
            "(`data/processed/cleaned_churn_data.csv`)."
        )
        st.stop()

    df = cleaned_df.copy()

    # Overview metrics
    st.subheader("Overview")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Customers", f"{len(df):,}")
    with c2:
        st.metric("Churned Customers", f"{df['Churn'].sum():,}")
    with c3:
        st.metric("Overall Churn Rate", f"{df['Churn'].mean()*100:.1f}%")
    with c4:
        st.metric("Avg Tenure (months)", f"{df['tenure'].mean():.1f}")

    st.markdown("---")

    # Churn by key dimensions
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Churn Rate by Contract")
        contract_churn = df.groupby("Contract")["Churn"].mean().sort_values(ascending=False)
        st.bar_chart(contract_churn)

    with col_b:
        st.subheader("Churn Rate by Internet Service")
        internet_churn = df.groupby("InternetService")["Churn"].mean().sort_values(ascending=False)
        st.bar_chart(internet_churn)

    col_c, col_d = st.columns(2)

    with col_c:
        st.subheader("Churn Rate by Tenure Group")
        if "TenureGroup" in df.columns:
            tenure_churn = df.groupby("TenureGroup")["Churn"].mean().sort_values(ascending=False)
            st.bar_chart(tenure_churn)
        else:
            st.info("TenureGroup not found in cleaned data.")

    with col_d:
        st.subheader("Churn Rate by Payment Method")
        payment_churn = df.groupby("PaymentMethod")["Churn"].mean().sort_values(ascending=False)
        st.bar_chart(payment_churn)

    st.markdown("---")
    st.subheader("Key Takeaways")
    st.markdown(
        """
        - **Month-to-month** contracts show the highest churn.
        - Customers with **short tenure (≤ 12 months)** are significantly more likely to leave.
        - **Fiber optic** users without support services are a high-risk group.
        - **Electronic check** payment method is associated with elevated churn.
        - Focus retention budget on the top 10–20% highest predicted risk customers.
        """
    )

# -------------------------------------------------
# PAGE 5: ABOUT
# -------------------------------------------------
elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")

    st.markdown(
        """
        ### Customer Churn Prediction – Telecom

        This Streamlit application is part of an end-to-end data science portfolio project.

        **Pipeline:**
        1. Exploratory Data Analysis
        2. Data Preprocessing & Feature Engineering
        3. Model Training & Evaluation (Logistic Regression, Random Forest, XGBoost, etc.)
        4. Business Insights & Risk Scoring
        5. Interactive Demo (this app)

        **Tech Stack:**  
        Python · pandas · scikit-learn · XGBoost / LightGBM · SHAP · Streamlit · joblib

        **Dataset:**  
        [Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

        ---
        Built as a portfolio project to demonstrate practical churn analysis and deployment skills.
        """
    )

    st.markdown("---")
    st.caption("Run the full analysis notebooks first so the model artifacts are available.")
