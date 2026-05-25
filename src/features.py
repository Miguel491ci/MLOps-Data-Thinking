import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    "gender",
    "age",
    "annual_income_k$",
    "spending_score_1_100",
]


def get_features(df: pd.DataFrame) -> pd.DataFrame:
    return df[FEATURE_COLUMNS].copy()


def scale_features(X: pd.DataFrame):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler
