# Customer Churn Prediction & Analysis – Telecom

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

End-to-end **Customer Churn Analysis** project for a telecom company.  
Identifies high-risk customers, uncovers key churn drivers, and delivers actionable retention strategies.

---

## Business Problem

Telecom companies typically face **15–30% annual churn**.  
Acquiring a new customer costs **5–7× more** than retaining an existing one.

**Goal:**  
Build a predictive model that identifies customers most likely to churn and provide clear business recommendations to reduce revenue loss.

---

## Dataset

**Source:** [Telco Customer Churn – Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
**Size:** 7,043 customers × 21 features  
**Target:** `Churn` (Yes / No) – ~26.5% positive class

**Key Features:**
- Demographics: gender, SeniorCitizen, Partner, Dependents
- Services: PhoneService, InternetService, OnlineSecurity, TechSupport, StreamingTV, etc.
- Account: tenure, Contract, PaperlessBilling, PaymentMethod
- Billing: MonthlyCharges, TotalCharges

---

## Project Structure

```
customer-churn-analysis/
│
├── data/
│   ├── raw/                          # Place original CSV here
│   └── processed/                    # Cleaned & engineered data
│
├── notebooks/
│   ├── 01_data_exploration_eda.ipynb
│   ├── 02_data_preprocessing_feature_engineering.ipynb
│   ├── 03_model_building_evaluation.ipynb
│   └── 04_insights_recommendations.ipynb
│
├── models/                           # Saved models & preprocessor
├── reports/                          # High-risk customer lists, figures
├── src/                              # (Optional) production scripts
├── app/                              # Streamlit demo app
│   ├── streamlit_app.py
│   └── README.md
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## How to Run

### 1. Clone & Setup
```bash
git clone https://github.com/YOUR_USERNAME/customer-churn-analysis.git
cd customer-churn-analysis

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download Dataset
1. Go to [Kaggle Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
2. Download `WA_Fn-UseC_-Telco-Customer-Churn.csv`
3. Rename to `telco_customer_churn.csv` and place it in `data/raw/`

### 3. Run Notebooks (in order)
```bash
jupyter notebook notebooks/
```
1. `01_data_exploration_eda.ipynb` → Understand the data
2. `02_data_preprocessing_feature_engineering.ipynb` → Clean + engineer features
3. `03_model_building_evaluation.ipynb` → Train & evaluate models
4. `04_insights_recommendations.ipynb` → Business insights + risk scoring

---

## Methodology

| Step | Description |
|------|-------------|
| **EDA** | Churn rate analysis, univariate & bivariate plots, correlation |
| **Preprocessing** | Handle TotalCharges blanks, encode target, missing values |
| **Feature Engineering** | Tenure groups, AvgMonthlyCharges, NumServices, interaction features |
| **Modeling** | Logistic Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM |
| **Evaluation** | Accuracy, Precision, Recall, F1, ROC-AUC (focus on imbalanced metrics) |
| **Explainability** | Feature importance + optional SHAP |
| **Business Output** | Risk scoring (Low/Medium/High) + prioritized recommendations |

---

## Key Insights (Typical Results)

- **Month-to-month contracts** have the highest churn rate (~42%)
- Customers with **tenure ≤ 12 months** are significantly more likely to churn
- **Fiber optic** users without Tech Support / Online Security show elevated risk
- **Electronic check** payment method correlates with higher churn
- Churners tend to have **higher average MonthlyCharges**

---

## Model Performance

(Fill after running notebook 03)

| Model              | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|--------------------|----------|-----------|--------|----------|---------|
| Logistic Regression| ...      | ...       | ...    | ...      | ...     |
| Random Forest      | ...      | ...       | ...    | ...      | ...     |
| XGBoost            | ...      | ...       | ...    | ...      | ...     |

**Recommendation:** Prioritize models with higher **Recall** when the cost of missing a churner is high.

---

## Business Recommendations

| Priority | Segment | Suggested Action |
|----------|---------|------------------|
| High | Month-to-month + short tenure | Offer discounted longer-term contracts |
| High | Fiber optic + No Tech Support | Bundle free/discounted support for 3–6 months |
| High | Electronic check payers | Incentivize auto-pay / card payment |
| Medium | High MonthlyCharges + new customers | Personalized loyalty discount |
| Low | Long-tenure + two-year contract | Referral & loyalty rewards |

Focus retention budget on the **top 10–20% highest predicted risk** customers.

---

## Tech Stack

- **Python** 3.10+
- **Data:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **ML:** scikit-learn, XGBoost, LightGBM
- **Imbalance:** imbalanced-learn (SMOTE optional)
- **Explainability:** SHAP (optional)
- **Deployment ready:** joblib, Streamlit

---

## Streamlit Demo App

An interactive web app is included in the `app/` folder.

```bash
# From project root (after running notebooks 02 & 03)
streamlit run app/streamlit_app.py
```

**Features:**
- Single customer churn prediction with risk level (Low / Medium / High)
- Insights dashboard (churn by Contract, Internet Service, Tenure, Payment Method)
- Clean, portfolio-ready UI

---

## Future Improvements

- [x] Streamlit interactive demo
- [ ] Customer Lifetime Value (CLV) integration
- [ ] Automated re-training pipeline
- [ ] FastAPI scoring endpoint
- [ ] A/B testing framework for retention offers

---

## Author

**Your Name**  
[LinkedIn](https://linkedin.com/in/yourprofile) · [Portfolio](https://yourportfolio.com) · [GitHub](https://github.com/YOUR_USERNAME)

---

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
