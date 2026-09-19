"""Unit tests for the churn prediction pipeline."""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.data.make_dataset import load_config, clean_telco_data, split_and_save_data
from src.features.build_features import build_preprocessor, get_features_and_target
from src.models.train import train
from src.models.predict import ChurnPredictor


@pytest.fixture
def config():
    return load_config("config/config.yaml")


@pytest.fixture
def sample_df():
    """Create a minimal dataframe mirroring Telco structure."""
    return pd.DataFrame({
        "customerID": ["001", "002", "003", "004"],
        "gender": ["Male", "Female", "Male", "Female"],
        "SeniorCitizen": [0, 0, 1, 0],
        "Partner": ["Yes", "No", "No", "Yes"],
        "Dependents": ["No", "No", "No", "Yes"],
        "tenure": [1, 24, 0, 48],
        "PhoneService": ["Yes", "Yes", "No", "Yes"],
        "MultipleLines": ["No", "Yes", "No phone service", "Yes"],
        "InternetService": ["DSL", "Fiber optic", "DSL", "No"],
        "OnlineSecurity": ["No", "Yes", "Yes", "No internet service"],
        "OnlineBackup": ["Yes", "No", "No", "No internet service"],
        "DeviceProtection": ["No", "Yes", "No", "No internet service"],
        "TechSupport": ["No", "No", "Yes", "No internet service"],
        "StreamingTV": ["No", "Yes", "No", "No internet service"],
        "StreamingMovies": ["No", "No", "Yes", "No internet service"],
        "Contract": ["Month-to-month", "One year", "Month-to-month", "Two year"],
        "PaperlessBilling": ["Yes", "No", "Yes", "No"],
        "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        "MonthlyCharges": [45.20, 89.10, 29.85, 20.05],
        "TotalCharges": [" 45.20 ", "2138.4", " ", "962.4"],
        "Churn": ["Yes", "No", "No", "No"]
    })


def test_config_structure(config):
    assert "data" in config
    assert "features" in config
    assert "model" in config
    assert "numerical" in config["features"]
    assert "categorical" in config["features"]


def test_clean_telco_data(sample_df):
    cleaned = clean_telco_data(sample_df)
    assert cleaned["TotalCharges"].dtype == np.float64
    assert cleaned.loc[cleaned["tenure"] == 0, "TotalCharges"].iloc[0] == 0.0
    assert cleaned["Churn"].tolist() == [1, 0, 0, 0]


def test_feature_preprocessor(sample_df, config):
    cleaned = clean_telco_data(sample_df)
    X, y = get_features_and_target(cleaned, config)
    preprocessor = build_preprocessor(config)
    X_trans = preprocessor.fit_transform(X)

    assert not np.isnan(X_trans).any()
    assert X_trans.shape[0] == len(sample_df)


def test_prediction_output(config):
    model_path = Path(config["model"]["output_path"])
    if not model_path.exists():
        pytest.skip("Model artifact not yet generated")

    predictor = ChurnPredictor(model_path)
    sample = pd.DataFrame([{
        "customerID": "999-TEST",
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 3,
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
        "MonthlyCharges": 75.0,
        "TotalCharges": 225.0
    }])
    scored = predictor.predict(sample)
    assert "churn_prediction" in scored.columns
    assert "churn_probability" in scored.columns
    assert 0.0 <= scored["churn_probability"].iloc[0] <= 1.0
