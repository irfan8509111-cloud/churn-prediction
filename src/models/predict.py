"""Inference script for predicting customer churn on new account data."""

from pathlib import Path
from typing import Union, Dict, Any, List
import joblib
import pandas as pd

from src.data.make_dataset import load_config, clean_telco_data


class ChurnPredictor:
    """Loads trained churn model pipeline and scores customer records."""

    def __init__(self, model_path: Union[str, Path] = "models/churn_model.joblib"):
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}. Run train.py first.")
        self.model = joblib.load(path)

    def predict(self, data: Union[pd.DataFrame, List[Dict[str, Any]]]) -> pd.DataFrame:
        """Score customer records with binary churn prediction and probability."""
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy()

        cleaned_df = clean_telco_data(df)
        preds = self.model.predict(cleaned_df)
        probs = self.model.predict_proba(cleaned_df)[:, 1]

        result = df.copy()
        result["churn_prediction"] = preds
        result["churn_probability"] = [round(float(p), 4) for p in probs]
        return result


def main():
    predictor = ChurnPredictor()

    sample_customers = [
        {
            "customerID": "1001-DEMO1",
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 2,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 70.35,
            "TotalCharges": 139.05
        },
        {
            "customerID": "1002-DEMO2",
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "Yes",
            "tenure": 60,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "Yes",
            "DeviceProtection": "Yes",
            "TechSupport": "Yes",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Two year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Bank transfer (automatic)",
            "MonthlyCharges": 85.50,
            "TotalCharges": 5120.00
        }
    ]

    scored = predictor.predict(sample_customers)
    cols = ["customerID", "Contract", "tenure", "MonthlyCharges", "churn_probability", "churn_prediction"]
    print("\nSample Predictions:")
    print(scored[cols].to_string(index=False))


if __name__ == "__main__":
    main()
