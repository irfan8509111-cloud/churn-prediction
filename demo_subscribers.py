"""Subscriber cohort analysis demo.

Scenario:
A cohort of 199 active subscribers is evaluated.
19 subscribers failed or missed last month's payment.
This script scores the entire cohort with our trained model, quantifies
revenue at risk, and segments subscribers into actionable retention categories.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from src.models.predict import ChurnPredictor
from src.data.make_dataset import clean_telco_data


def load_subscriber_cohort(seed: int = 42) -> pd.DataFrame:
    """Extract a realistic cohort of 199 subscribers from test data."""
    test_path = Path("data/processed/test.csv")
    if not test_path.exists():
        raise FileNotFoundError("Run 'python -m src.data.make_dataset' first.")

    df = pd.read_csv(test_path)

    # Pick 19 high-friction/short-tenure users to represent the 19 overdue accounts
    unpaid_candidates = df[
        (df["Contract"] == "Month-to-month") &
        (df["tenure"] <= 12) &
        (df["PaymentMethod"].isin(["Electronic check", "Mailed check"]))
    ]
    if len(unpaid_candidates) < 19:
        unpaid_candidates = df[df["Contract"] == "Month-to-month"]

    unpaid_sample = unpaid_candidates.sample(n=19, random_state=seed).copy()
    unpaid_sample["payment_status"] = "Overdue (Missed Last Month)"

    # Pick 180 paying subscribers from the remaining pool
    remaining = df.drop(index=unpaid_sample.index)
    paid_sample = remaining.sample(n=180, random_state=seed).copy()
    paid_sample["payment_status"] = "Paid on Time"

    cohort = pd.concat([paid_sample, unpaid_sample]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return cohort


def run_cohort_analysis():
    print("=" * 68)
    print(" SUBSCRIBER COHORT ANALYSIS (199 ACCOUNTS, 19 OVERDUE)")
    print("=" * 68)

    cohort = load_subscriber_cohort()
    predictor = ChurnPredictor("models/churn_model.joblib")
    scored = predictor.predict(cohort)

    # 1. Overview counts
    total_subs = len(scored)
    overdue_subs = scored[scored["payment_status"] == "Overdue (Missed Last Month)"]
    paid_subs = scored[scored["payment_status"] == "Paid on Time"]

    print(f"\nCohort Size: {total_subs}")
    print(f"  - Active / Paid on Time : {len(paid_subs)} ({len(paid_subs)/total_subs*100:.1f}%)")
    print(f"  - Missed Last Month Fee : {len(overdue_subs)} ({len(overdue_subs)/total_subs*100:.1f}%)")

    # 2. Revenue exposure
    total_mrr = scored["MonthlyCharges"].sum()
    overdue_mrr = overdue_subs["MonthlyCharges"].sum()
    collected_mrr = paid_subs["MonthlyCharges"].sum()

    print("\nFinancial Impact (Monthly Recurring Revenue - MRR):")
    print(f"  - Total MRR                : ${total_mrr:,.2f}")
    print(f"  - Collected Revenue        : ${collected_mrr:,.2f}")
    print(f"  - Overdue Revenue at Risk  : ${overdue_mrr:,.2f} ({overdue_mrr/total_mrr*100:.1f}% of total)")

    # 3. Model predicted churn risk
    avg_overdue_risk = overdue_subs["churn_probability"].mean() * 100
    avg_paid_risk = paid_subs["churn_probability"].mean() * 100
    overdue_predicted_churn = (overdue_subs["churn_prediction"] == 1).sum()

    print("\nModel Risk Assessment:")
    print(f"  - Overdue Group Avg Churn Probability : {avg_overdue_risk:.1f}%")
    print(f"  - Paying Group Avg Churn Probability  : {avg_paid_risk:.1f}%")
    print(f"  - Overdue Accounts Flagged to Churn   : {overdue_predicted_churn} of 19 ({overdue_predicted_churn/19*100:.1f}%)")

    # 4. Table of the 19 overdue accounts
    cols = ["customerID", "Contract", "tenure", "PaymentMethod", "MonthlyCharges", "churn_probability", "churn_prediction"]
    ranked_overdue = overdue_subs[cols].sort_values("churn_probability", ascending=False)

    print("\n--- The 19 Overdue Accounts Ranked by Churn Risk ---")
    print(ranked_overdue.to_string(index=False))

    # 5. Suggested actions
    critical = ranked_overdue[ranked_overdue["churn_probability"] >= 0.50]
    salvageable = ranked_overdue[ranked_overdue["churn_probability"] < 0.50]

    print("\n--- Recommended Action Plan ---")
    print(f"1. Critical Retention ({len(critical)} subscribers): High probability of permanent cancellation.")
    print(f"   -> Trigger immediate outbound retention call with 15% discount or annual plan contract.")
    print(f"2. Billing Follow-Up ({len(salvageable)} subscribers): Lower underlying churn risk; missed fee likely administrative.")
    print(f"   -> Send automated payment retry emails and update credit card notification.")


if __name__ == "__main__":
    run_cohort_analysis()
