import pandas as pd


def convert_price_to_number(s: pd.Series) -> pd.Series:
    return s.str.replace("$", "").str.replace(",", "").astype(float)
