#Extracción + Transformación de datos

import pandas as pd
from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH


def load_raw_data() -> pd.DataFrame:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo {RAW_DATA_PATH}. "
            "Coloca Mall_Customers.csv dentro de data/raw/"
        )

    return pd.read_csv(RAW_DATA_PATH)


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
        .str.replace("-", "_", regex=False)
    )
    return df


def preprocess_data() -> pd.DataFrame:
    df = load_raw_data()
    df = clean_column_names(df)

    if "customerid" in df.columns:
        df = df.drop(columns=["customerid"])

    expected_columns = ["gender", "age", "annual_income_k$", "spending_score_1_100"]

    missing = [col for col in expected_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"Faltan columnas esperadas: {missing}. Columnas encontradas: {list(df.columns)}"
        )

    df["gender"] = df["gender"].map({"Male": 1, "Female": 0})

    if df["gender"].isnull().any():
        raise ValueError("La columna gender tiene valores diferentes a Male/Female.")

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    return df


if __name__ == "__main__":
    processed = preprocess_data()
    print(processed.head())
    print(processed.info())