# Streamlit Churn Prediction App

## How to Run

From the **project root**:

```bash
streamlit run app/streamlit_app.py
```

Or:

```bash
cd app
streamlit run streamlit_app.py
```

## Prerequisites

1. Run notebooks **02** and **03** first so these files exist:
   - `models/best_model.pkl`
   - `models/preprocessor.pkl`
   - `models/feature_names.pkl` (optional but recommended)
   - `data/processed/cleaned_churn_data.csv` (for the Insights Dashboard)

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Features

- **Home** – Project overview and quick metrics
- **Predict Churn** – Interactive form to score a single customer
- **Insights Dashboard** – Churn rates by Contract, Internet Service, Tenure, Payment Method
- **About** – Project description

## Notes

- The feature engineering inside the app **must match** notebook 02 (TenureGroup, AvgMonthlyCharges, NumServices, etc.).
- If you change feature engineering, update both the notebook and `engineer_features()` in this app.
