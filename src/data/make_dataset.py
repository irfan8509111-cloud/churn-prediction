"""Data ingestion and preprocessing script for Telco customer churn data.

Loads raw data, fixes type quirks (e.g. whitespace in TotalCharges),
converts the target column to binary, and creates stratified train/test splits.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Read project configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def clean_telco_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean data issues common to the Telco dataset.
    
    In particular:
    - TotalCharges is stored as object/string with whitespace for new accounts (tenure == 0).
    - Convert TotalCharges to float and impute tenure=0 accounts with 0.0.
    - Convert Churn from Yes/No strings to 1/0 integers.
    """
    cleaned = df.copy()

    # TotalCharges contains whitespace for new users with tenure == 0
    if "TotalCharges" in cleaned.columns:
        cleaned["TotalCharges"] = pd.to_numeric(
            cleaned["TotalCharges"].astype(str).str.strip(),
            errors="coerce"
        )
        # For customers with tenure == 0, total charges is 0
        cleaned["TotalCharges"] = cleaned["TotalCharges"].fillna(0.0)

    # Convert binary target
    if "Churn" in cleaned.columns:
        cleaned["Churn"] = (cleaned["Churn"].astype(str).str.strip().str.lower() == "yes").astype(int)

    return cleaned


def split_and_save_data(
    df: pd.DataFrame,
    config: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataset into train and test sets and save to data/processed/."""
    target_col = config["data"]["target_column"]
    test_size = config["split"]["test_size"]
    stratify = df[target_col] if config["split"]["stratify"] else None
    random_state = config["split"]["random_state"]

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=stratify,
        random_state=random_state
    )

    train_path = Path(config["data"]["train_path"])
    test_path = Path(config["data"]["test_path"])

    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Saved {len(train_df)} training rows to {train_path}")
    print(f"Saved {len(test_df)} testing rows to {test_path}")

    return train_df, test_df


def main():
    config = load_config()
    raw_path = Path(config["data"]["raw_path"])

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset not found at {raw_path}")

    print(f"Loading raw dataset from {raw_path}...")
    raw_df = pd.read_csv(raw_path)
    print(f"Loaded {raw_df.shape[0]} rows and {raw_df.shape[1]} columns.")

    cleaned_df = clean_telco_data(raw_df)
    train_df, test_df = split_and_save_data(cleaned_df, config)
    print(f"Train churn rate: {train_df['Churn'].mean():.2%}")
    print(f"Test churn rate:  {test_df['Churn'].mean():.2%}")


if __name__ == "__main__":
    main()
