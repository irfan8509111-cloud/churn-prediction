"""Feature engineering pipeline for customer churn modeling.

Uses Scikit-learn ColumnTransformer to standardize continuous variables
and one-hot encode categorical attributes without data leakage.
"""

from typing import Dict, Any, Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def build_preprocessor(config: Dict[str, Any]) -> ColumnTransformer:
    """Build ColumnTransformer based on config feature lists."""
    num_features = config["features"]["numerical"]
    cat_features = config["features"]["categorical"]

    num_pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_features),
            ("cat", cat_pipeline, cat_features)
        ],
        remainder="drop"
    )

    return preprocessor


def get_features_and_target(
    df: pd.DataFrame,
    config: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split dataframe into feature matrix X and target y."""
    target_col = config["data"]["target_column"]
    id_col = config["data"].get("id_column")

    drop_cols = [target_col]
    if id_col and id_col in df.columns:
        drop_cols.append(id_col)

    X = df.drop(columns=drop_cols)
    y = df[target_col]
    return X, y
