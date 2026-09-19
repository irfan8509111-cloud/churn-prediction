# Telco Customer Churn Prediction & Retention Analytics

An end-to-end machine learning project analyzing subscription churn patterns on the IBM Telco dataset (7,043 customers). The project investigates customer tenure dynamics, price sensitivity, and service friction, and builds predictive models to identify at-risk subscribers before they cancel.

---

## Business Problem & Context

For subscription and recurring revenue services, customer acquisition cost (CAC) is typically 5x to 7x higher than retention cost. A 5% reduction in churn can increase customer lifetime value (LTV) significantly. 

The primary business goals of this project are:
1. Identify high-risk subscribers early so customer success teams can proactively intervene.
2. Differentiate between involuntary churn (billing/payment failures) and voluntary churn (dissatisfaction, competitor pricing).
3. Provide an interpretable, testable codebase that can be deployed into automated batch or real-time scoring pipelines.

---

## Key Findings from Exploratory Analysis

- **Contract Duration is the #1 Predictor**: Customers on month-to-month contracts churn at **42.7%**, compared to **11.3%** on one-year contracts and **2.8%** on two-year contracts.
- **Tenure Hazard Curve**: Attrition is heavily concentrated within the first 1 to 6 months. Accounts that reach 12+ months stabilize into predictable, long-term subscribers.
- **Fiber Optic Attrition**: Fiber optic subscribers show higher churn (**41.9%**) than DSL subscribers (**19.0%**). These accounts carry high monthly bills ($70–$100+/mo), making them sensitive to promotional expirations and competing offerings.
- **Payment Method Friction**: Customers paying via electronic check experience twice the churn rate of those enrolled in automated credit card or bank transfer withdrawals.

---

## Model Evaluation & Performance

Models were evaluated on a held-out test split of 1,409 customers (stratified 80/20 split):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression (Baseline)** | 80.62% | 0.6593 | 0.5588 | 0.6049 | 0.8422 | 0.6349 |
| **Random Forest (Tuned)** | 80.41% | 0.6775 | 0.5000 | 0.5754 | **0.8446** | **0.6585** |

> **Why PR-AUC and Recall matter here**: With a 26.5% base churn rate, a naive majority-class model achieves 73.5% accuracy while identifying 0 churners. Precision-Recall AUC and Recall are the primary optimization metrics to balance false alarm rates against missed churners.

Generated figures and diagnostic plots are saved in `reports/figures/`:
- `reports/figures/confusion_matrix.png`: Test set confusion matrix with raw counts.
- `reports/figures/model_curves.png`: Side-by-side ROC and Precision-Recall curves.
- `reports/figures/feature_importances.png`: Top predictive features (tenure, TotalCharges, MonthlyCharges, contract types).

---

## Case Study: 199 Subscribers with 19 Overdue Accounts

To test the system on operational billing scenarios, we analyzed a cohort of **199 active subscribers** where **19 subscribers missed last month's fee** (`demo_subscribers.py`):

```text
Cohort Size: 199
  - Active / Paid on Time : 180 (90.5%)
  - Missed Last Month Fee : 19 (9.5%)

Financial Impact:
  - Total Monthly Revenue  : $12,668.80
  - Current Collected MRR  : $11,620.15
  - Overdue Revenue at Risk: $1,048.65 (8.3% of total revenue)

Model Risk Assessment:
  - Overdue Group Avg Churn Probability : 43.9% (vs 28.0% for paying accounts)
  - Overdue Accounts Flagged to Churn   : 10 of 19 (52.6%)
```

### Action Plan
1. **Critical Retention (10 accounts, churn prob > 50%)**: Accounts with high monthly charges and month-to-month contracts. Action: Immediate outreach with annual plan discount.
2. **Administrative Billing Follow-up (9 accounts, churn prob < 50%)**: Low underlying cancellation risk. The non-payment is likely an expired card or transient bank error. Action: Automated payment update reminders.

---

## Repository Structure

```
.
├── config/
│   └── config.yaml              # Central pipeline configuration (features, paths, hyperparameters)
├── data/
│   ├── raw/                     # Raw Telco dataset (telco_customer_churn.csv)
│   └── processed/               # Stratified train.csv and test.csv splits
├── models/
│   └── churn_model.joblib       # Serialized Scikit-learn pipeline (preprocessor + model)
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb # Executed Jupyter notebook with EDA & insights
├── reports/
│   ├── figures/                 # Diagnostic curves, confusion matrix, feature importances
│   └── metrics.json             # Serialized test evaluation metrics
├── src/
│   ├── data/
│   │   └── make_dataset.py      # Cleans Telco whitespace quirks and creates splits
│   ├── features/
│   │   └── build_features.py    # Leakage-free ColumnTransformer pipeline
│   ├── models/
│   │   ├── train.py             # Model training, comparison, and evaluation
│   │   └── predict.py           # Scoring class for batch and single customer inference
│   └── visualization/
│       └── plots.py             # Reusable seaborn/matplotlib diagnostic plots
├── tests/
│   └── test_pipeline.py         # Pytest suite covering cleaning, transforms, and inference
├── demo_subscribers.py          # Operational cohort demo (199 subs, 19 overdue)
├── requirements.txt             # Frozen project dependencies
└── README.md
```

---

## Setup & Execution

### 1. Prerequisites & Virtual Environment

```bash
# Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the End-to-End Pipeline

```bash
# 1. Clean data and generate train/test splits
python -m src.data.make_dataset

# 2. Train baseline and tuned models (generates metrics and figures)
python -m src.models.train

# 3. Test inference on sample records
python -m src.models.predict

# 4. Run the 199-subscriber overdue cohort simulation
python demo_subscribers.py
```

### 3. Run Automated Tests

```bash
pytest tests/ -v
```

### 4. Interactive Analysis

Launch Jupyter to explore the executed analysis:

```bash
jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
```
