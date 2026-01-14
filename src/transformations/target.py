import pandas as pd


def convert_price_to_number(s: pd.Series) -> pd.Series:
    return s.str.replace("$", "").str.replace(",", "").astype(float)


def transform_pipeline(s: pd.Series) -> pd.Series:
    s = convert_price_to_number(s)
    return s
