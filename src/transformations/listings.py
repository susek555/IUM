import ast
import re
from unicodedata import normalize

import contractions
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from textblob import TextBlob


def _nomralize_text(text: str) -> str:
    text = text.lower()
    out_text = normalize("NFKD", text)
    return out_text


def select_features(
    df: pd.DataFrame,
    select_features: list[str],
    target: list[str],
) -> pd.DataFrame:
    return df[select_features + target]


def add_is_luxury_attribute(df: pd.DataFrame) -> pd.DataFrame:
    def is_luxury(pt: str) -> int:
        pt = _nomralize_text(pt)
        if "loft" in pt or "villa" in pt or "boutique hotel" in pt:
            return 1
        return 0

    new_columns = df["property_type"].map(is_luxury, na_action="ignore").astype("Int64")
    df["is_luxury"] = new_columns
    return df


def aggregate_property_type(df: pd.DataFrame) -> pd.DataFrame:
    def map_proprerty_type(pt: str) -> str:
        pt = _nomralize_text(pt)
        if "rental unit" in pt:
            return "Rental unit"
        if "condo" in pt:
            return "Condo"
        if "home" in pt or "house" in pt:
            return "Home"
        if "hotel" in pt or "hostel" in pt:
            return "Hotel"
        return "Other"

    df["property_type"] = df["property_type"].map(
        map_proprerty_type, na_action="ignore"
    )
    return df


def fill_bathrooms_values_from_text(df: pd.DataFrame) -> pd.DataFrame:
    bathrooms_from_text = pd.to_numeric(
        df["bathrooms_text"].str.extract(r"(\d+\.?\d*)")[0],
        errors="coerce",
    )
    df["bathrooms"] = df["bathrooms"].fillna(bathrooms_from_text)
    return df


def add_is_bathroom_shared_attribute(df: pd.DataFrame) -> pd.DataFrame:
    def is_shared(pt: str) -> int:
        pt = pt.lower()
        if "shared" in pt:
            return 1
        return 0

    new_columns = (
        df["bathrooms_text"].map(is_shared, na_action="ignore").astype("Int64")
    )
    df["is_bathroom_shared"] = new_columns
    return df


def _convert_text_to_sentiment(text: str) -> float:
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()

    text = contractions.fix(text)  # pyright: ignore

    text = re.sub(r"[^\w\s!?]", "", text)
    text = text.lower()

    blob = TextBlob(text)
    return float(blob.sentiment.polarity)  # pyright: ignore


def convert_description_to_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    sentiment = df["description"].map(_convert_text_to_sentiment, na_action="ignore")
    df["description_sentiment"] = sentiment
    return df


def convert_neighborhood_overview_to_setiment(df: pd.DataFrame) -> pd.DataFrame:
    sentiment = df["neighborhood_overview"].map(
        _convert_text_to_sentiment, na_action="ignore"
    )
    df["neighborhood_overview_sentiment"] = sentiment
    return df


def encode_amenities_binary(df: pd.DataFrame, amenities: list[str]) -> pd.DataFrame:
    temp_amenities = df["amenities"].apply(
        lambda x: set(_nomralize_text(a) for a in ast.literal_eval(x))
        if pd.notna(x) and isinstance(x, str) and x.startswith("[")
        else set()
    )

    new_columns_data = {}

    for amenity in amenities:
        search_key = _nomralize_text(amenity)
        clean_col_name = (
            f"amenity_{search_key.strip().replace(' ', '_').replace('/', '_').lower()}"
        )
        new_columns_data[clean_col_name] = temp_amenities.apply(
            lambda tags: 1 if search_key in tags else 0
        ).astype("Int64")

    new_df = pd.DataFrame(new_columns_data, index=df.index)
    df = pd.concat([df, new_df], axis=1)

    return df


def convert_percentage_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        df[col] = df[col].str.strip("%").astype(float) / 100.0
    return df


def convert_tf_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    def map_tf(val: str) -> int:
        return 1 if val == "t" else 0

    for col in columns:
        df[col] = df[col].map(map_tf, na_action="ignore").astype("Int64")
    return df


def drop_processed_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.drop(columns=columns)
    return df


def convert_price_to_number(df: pd.DataFrame) -> pd.DataFrame:
    df["price"] = df["price"].str.replace("$", "").str.replace(",", "").astype(float)
    return df


def transform_price(df: pd.DataFrame) -> pd.DataFrame:
    df["price"] = np.log1p(df["price"])
    return df