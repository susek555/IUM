import numpy as np
import pandas as pd


def convert_price_to_number(s: pd.Series) -> pd.Series:
    return s.str.replace("$", "").str.replace(",", "").astype(float)


def logarithmize_price(s: pd.Series) -> pd.Series:
    return s.apply(np.log1p)


def transform_pipeline(s: pd.Series) -> pd.Series:
    s = convert_price_to_number(s)
    s = logarithmize_price(s)
    return s
