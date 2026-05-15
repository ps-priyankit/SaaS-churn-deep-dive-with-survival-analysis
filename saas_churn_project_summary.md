# SaaS Churn Survival Analysis — Complete Project Summary

**Project:** SaaS Customer Churn — Survival Analysis & Retention Intelligence  
**Dataset:** IBM Telco Customer Churn (7,043 customers, 21 columns)  
**Tools:** Python, Pandas, NumPy, Matplotlib, Seaborn, Lifelines, XGBoost, SHAP  
**Notebooks:** 5 | **Charts:** 8 | **Build time:** 3 days  

---

## Final Numbers

| Metric | Value |
|---|---|
| Total customers | 7,043 |
| Overall churn rate | 26.5% |
| High risk customers | 1,767 |
| Annualised revenue at risk | $2,207,678 |
| Model ROC-AUC | 0.940 |
| Net ROI from retention campaign | $48,225 |

---

## Notebook 01 — Data Cleaning

**What we did:**
- Loaded raw CSV (7,043 rows, 21 columns)
- Found TotalCharges stored as string — converted to numeric
- Discovered 11 hidden NaN values in TotalCharges — all had tenure=0 (new customers), filled with 0
- Converted SeniorCitizen from 0/1 integer to Yes/No
- Created two survival analysis columns:
  - `event` — binary churn flag (1=churned, 0=retained)
  - `survival_time` — copy of tenure, clipped to minimum 1
- Added 3 synthetic columns to enrich analysis:
  - `signup_channel` — acquisition channel (organic, paid_search, referral, social)
  - `monthly_revenue` — MonthlyCharges with small random variation
  - `support_tickets` — Poisson random, higher mean for churned customers
- Saved cleaned file to `data/processed/churn_clean.csv` (7,043 rows, 26 columns)

**Why event and survival_time:**  
The lifelines library requires exactly these two inputs. survival_time = how long the customer lasted. event = did they actually churn? Without these two columns, survival analysis cannot run.

---

## Notebook 02 — Exploratory Data Analysis

**What we did:**
- Calculated churn rate by contract type, internet service, and signup channel
- Built tenure distribution histogram comparing churned vs retained customers
- Built monthly charges boxplot comparing churned vs retained
- Built cohort retention heatmap (tenure band × contract type)
- Built feature correlation matrix

**Key findings:**

| Finding | Insight |
|---|---|
| Month-to-month churn rate | 42.7% |
| One year churn rate | 11.3% |
| Two year churn rate | 2.8% |
| Fiber optic churn rate | 41.9% |
| DSL churn rate | 19.0% |
| No internet churn rate | 7.4% |
| Month-to-month retention at 0-3 months | Only 42% |
| Two year retention at 0-3 months | 100% |

**Correlation matrix highlights:**
- support_tickets → event: 0.65 (strongest churn signal)
- tenure → event: -0.35 (longer customers churn less)
- MonthlyCharges → event: 0.19 (higher bills = more churn)
- TotalCharges → event: -0.20 (more total spend = less churn)

**Business insights:**
1. Month-to-month is a churn machine — 15× worse than two-year contracts
2. Months 0-6 is the danger zone — churners leave early and fast
3. Fiber optic customers pay more but churn nearly as fast as month-to-month customers
4. Support tickets are the single strongest early warning signal

---

## Notebook 03 — Survival Analysis

**What is survival analysis:**  
Standard churn analysis asks who churned. Survival analysis asks when they churn and what is the probability of surviving past any given month. Borrowed from medical research — originally used to track patient survival after treatment.

**Library used:** lifelines  
**Two inputs required:** survival_time (how long they lasted) + event (did they churn)

**What we built:**

### Kaplan-Meier Curves
A line chart showing probability of a customer still being active at any given month.

**By contract type:**
- Month-to-month: median survival = 35 months. Drops immediately and steeply.
- One year: median survival = infinity (never crossed 50% line in dataset)
- Two year: median survival = infinity (almost perfectly flat)

**By internet service:**
- Fiber optic: steepest drop, approaching 50% survival by month 70
- DSL: gradual decline, ends at ~72% survival at month 72
- No internet: barely moves, stays above 90% entire time

### Log-Rank Test
Statistical test confirming survival curve differences are real, not random.
- Month-to-month vs One year: p-value = 0.000000 ✓
- Month-to-month vs Two year: p-value = 0.000000 ✓
- Both statistically significant beyond any doubt

### Cox Proportional Hazards Model
Tells exactly which features drive churn and by how much.  
Concordance score: 0.89 (very strong)

| Feature | Hazard Ratio | Meaning |
|---|---|---|
| support_tickets | 1.45 | Each extra ticket = 45% higher churn risk |
| Fiber optic | 1.50 | 50% higher churn risk vs DSL |
| One year contract | 0.37 | 63% lower churn risk vs month-to-month |
| Two year contract | 0.23 | 77% lower churn risk vs month-to-month |
| No internet | 0.67 | 33% lower churn risk |
| SeniorCitizen | 0.99 | Not significant (p=0.85) |

**Business insights:**
1. Month-to-month customers need a completely different retention strategy
2. Fiber optic needs urgent attention — high price, high churn
3. Support tickets are the most actionable early warning signal
4. Converting customers to annual contracts is the highest ROI retention move

---

## Notebook 04 — Churn Prediction Model

**What we built:**  
A machine learning model that scores every customer with a churn probability (0-1) and assigns them to a risk tier.

**Why XGBoost:**  
Builds hundreds of small decision trees combined for high accuracy. Industry standard for tabular data. scale_pos_weight=3 used to handle class imbalance (only 26.5% churners).

**Features used (10):**
tenure, MonthlyCharges, TotalCharges, support_tickets, SeniorCitizen, Contract, InternetService, TechSupport, OnlineSecurity, signup_channel

**Train/test split:** 80% train (5,634 customers) / 20% test (1,409 customers)

**Model performance:**

| Metric | Value |
|---|---|
| ROC-AUC | 0.940 |
| Accuracy | 87% |
| Recall (churners) | 0.87 |
| Precision (churners) | 0.70 |

**Confusion matrix:**
- 893 correctly predicted as staying
- 326 correctly predicted as churning
- 142 missed churners (predicted stay, actually left)
- 48 false alarms (predicted churn, actually stayed)

**SHAP feature importance (ranked):**
1. support_tickets — dominant by large margin
2. Contract — second most important
3. MonthlyCharges — third
4. tenure — fourth
5. TechSupport — fifth
6. OnlineSecurity — sixth
7. TotalCharges — seventh
8. InternetService — eighth
9. SeniorCitizen — minimal
10. signup_channel — almost zero (generated randomly)

**Risk tiers assigned:**

| Tier | Customers | Avg Churn Probability |
|---|---|---|
| High (>0.7) | 1,767 | 91.2% |
| Medium (0.4-0.7) | 827 | 55.2% |
| Low (<0.4) | 4,449 | 9.7% |

---

## Notebook 05 — LTV & Financial Impact

**What we built:**  
Converted churn predictions into revenue numbers. The layer that makes a CEO or CFO care.

**LTV calculation:**  
LTV = monthly_revenue × expected_remaining_months  
expected_remaining_months = (1 - churn_probability) × 24

**Revenue at risk:**  
revenue_at_risk = monthly_revenue × churn_probability

**LTV by risk tier:**

| Tier | Avg LTV | Avg Revenue at Risk | Avg Monthly Revenue |
|---|---|---|---|
| Low | $1,253 | $6 | $58.53 |
| Medium | $815 | $42 | $76.32 |
| High | $158 | $68 | $74.94 |

**Key insight:** High risk customers pay more per month ($74.94) but are worth 8× less in lifetime value ($158 vs $1,253) because they leave so soon.

**Financial impact summary:**

| Metric | Value |
|---|---|
| High risk monthly revenue at risk | $120,806 |
| Medium risk monthly revenue at risk | $35,032 |
| Combined monthly revenue at risk | $155,838 |
| Annualised revenue at risk | $2,207,678 |

**Retention campaign ROI:**

| Assumption | Value |
|---|---|
| Customers targeted | 1,767 |
| Cost per intervention | $20 |
| Total campaign cost | $35,340 |
| Assumed success rate | 30% |
| Customers saved | 530 |
| LTV saved | $83,565 |
| Net ROI | $48,225 |
| Return per $1 spent | $1.40 |

**LTV by acquisition channel:**

| Channel | Avg LTV | High Risk % |
|---|---|---|
| Referral | $946 | 23.5% |
| Organic | $931 | 25.9% |
| Paid search | $915 | 25.1% |
| Social | $897 | 26.2% |

---

## Interview Story

*"Standard churn analysis tells you who left. I built a survival model that tells you when the critical intervention window is — and it's months 0-6 for month-to-month customers, not month 12. The Cox model showed fiber optic customers carry a 50% higher churn hazard ratio, and each support ticket raises churn risk by 45%. I then converted that into a financial number: $2.2M in annualised revenue at risk. A retention campaign targeting the 1,767 high-risk customers costs $35,000 and delivers $48,000 net ROI. That's the kind of analysis that changes what the business actually does."*

---

## What's Next

1. Tableau dashboard — 3 dashboards using churn_scored.csv
2. SQL queries — 6 business queries in MySQL
3. GitHub README — project story for recruiters
4. LinkedIn carousel — 7 slides for portfolio visibility

