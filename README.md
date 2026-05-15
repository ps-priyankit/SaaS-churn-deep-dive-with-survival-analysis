# SaaS Customer Churn — Survival Analysis & Retention Intelligence

> **Business question:** When do customers churn, which segments are most 
> at risk, and what is the financial impact of predicted departures?

---

## Live Dashboard
[View on Tableau Public](https://public.tableau.com/views/SaaSChurnIntelligence/ExecutiveOverview?:language=en-GB&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)

---

## Executive Summary

This project goes beyond standard churn rate reporting. Using survival 
analysis (Kaplan-Meier + Cox model), I identified the exact tenure window 
where churn risk peaks by segment — and quantified the revenue at risk.

**Key findings:**
- Month-to-month customers churn at **15× the rate** of two-year subscribers
- Fiber optic customers have a **50% higher churn hazard ratio** than DSL (Cox model, p < 0.001)
- Each additional support ticket raises churn risk by **45%**
- **1,767 high-risk customers** represent **$278,549 in LTV at risk**
- A targeted retention campaign costs $35,340 and delivers **$48,225 net ROI**

---

## Why This Project Is Different

| Standard churn analysis | This project |
|---|---|
| Who churned? | When do they churn and why? |
| Single churn rate % | Survival curves by segment |
| Feature importance | Cox hazard ratios with significance |
| Churn flag | Churn probability score per customer |
| No business impact | Revenue at risk + retention ROI |

---

## Tools & Techniques

| Tool | Usage |
|---|---|
| Python (Pandas, NumPy) | Data cleaning, EDA, feature engineering |
| lifelines | Kaplan-Meier survival curves, Cox model |
| XGBoost + SHAP | Churn prediction + explainability |
| MySQL | Business aggregation queries |
| Tableau Public | 3 interactive dashboards |
| Matplotlib, Seaborn | EDA visualisations |

---

## Dataset

IBM Telco Customer Churn (Kaggle) — 7,043 customers, 21 features.  
Augmented with 3 engineered columns: signup_channel, support_tickets, monthly_revenue.

---

## Project Structure
saas-churn-survival-analysis/
├── data/
│   ├── raw/                    ← original CSV
│   └── processed/              ← cleaned CSV
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_survival_analysis.ipynb
│   ├── 04_churn_model.ipynb
│   └── 05_ltv_financials.ipynb
├── outputs/
│   ├── charts/                 ← saved PNG exports
│   └── tableau_data/           ← CSVs for Tableau
├── churn_analysis.sql
├── requirements.txt
└── README.md

---

## Analysis Walkthrough

### 1. Data Cleaning & Engineering
- Fixed TotalCharges stored as string (11 hidden NaNs — all tenure=0 customers)
- Created survival_time and event columns for lifelines library
- Engineered 3 synthetic columns to enrich analysis

### 2. Exploratory Data Analysis
- Month-to-month churn rate: 42.7% vs 2.8% for two-year contracts
- Fiber optic churn rate: 41.9% — highest risk internet service
- Support tickets correlate 0.65 with churn — strongest single signal
- Critical window: 0-6 months is when most churners leave

### 3. Survival Analysis
- Kaplan-Meier curves show month-to-month median survival = 35 months
- One year and two year customers never crossed the 50% churn threshold
- Log-rank test: p < 0.0001 — differences are statistically significant
- Cox model concordance: 0.89

### 4. Churn Prediction Model
- XGBoost classifier — ROC-AUC: 0.940
- Catches 87% of actual churners before they leave
- SHAP explainability shows support_tickets as dominant feature
- Risk tiers: 1,767 High / 827 Medium / 4,449 Low

### 5. Financial Impact
- High risk customers: $278,549 total LTV at risk
- Annualised revenue at risk: $2,207,678
- Retention campaign ROI: $48,225 net on $35,340 spend

---

## Business Recommendations

1. **Convert month-to-month customers** to annual plans with a targeted 
   discount in months 3-6 — the peak churn window
2. **Flag fiber optic customers** with 2+ support tickets for immediate 
   proactive outreach — highest hazard ratio segment
3. **Invest in referral channel** — referral customers have highest LTV 
   ($946) and lowest churn rate (25.7%)
4. **Prioritise tech support quality** — TechSupport and OnlineSecurity 
   are top 5 churn drivers

---

## Model Performance

| Metric | Value |
|---|---|
| ROC-AUC | 0.940 |
| Accuracy | 87% |
| Recall (churners) | 0.87 |
| Precision (churners) | 0.70 |
| Cox Concordance | 0.89 |

---

## Setup & Installation

```bash
git clone https://github.com/ps-priyankit/SaaS-churn-deep-dive-with-survival-analysis
cd saas-churn-survival-analysis
pip install -r requirements.txt
```

Run notebooks in order: 01 → 02 → 03 → 04 → 05

---

## Contact

**Priyankit Singh**  
[LinkedIn](www.linkedin.com/in/priyankit) | [GitHub](https://github.com/ps-priyankit)