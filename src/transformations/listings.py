import ast
import re

import contractions
import pandas as pd
from bs4 import BeautifulSoup
from haversine import haversine
from textblob import TextBlob

from src.transformations.features import AMENITIES, INITIAL_FEATURES

CENTRE_LAT = 37.9755
CENTRE_LON = 23.7349


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def add_distance_to_centre_attribute(df: pd.DataFrame):
    def _compute_distance(row):
        lat = row.get("latitude")
        lon = row.get("longitude")
        return haversine((float(lat), float(lon)), (CENTRE_LAT, CENTRE_LON))

    df["distance_to_centre"] = df.apply(_compute_distance, axis=1)


def add_is_luxury_attribute(df: pd.DataFrame):
    def is_luxury(pt: str) -> int:
        pt = _normalize_text(pt)
        if "loft" in pt or "villa" in pt or "boutique hotel" in pt:
            return 1
        return 0

    new_columns = df["property_type"].map(is_luxury, na_action="ignore").astype("Int64")
    df["is_luxury"] = new_columns


def aggregate_property_type(df: pd.DataFrame):
    def map_proprerty_type(pt: str) -> str:
        pt = _normalize_text(pt)
        if "rental unit" in pt:
            return "rental_unit"
        if "condo" in pt:
            return "condo"
        if "home" in pt or "house" in pt:
            return "home"
        if "hotel" in pt or "hostel" in pt:
            return "hotel"
        return "other"

    df["property_type"] = df["property_type"].map(
        map_proprerty_type, na_action="ignore"
    )


def fill_bathrooms_values_from_text(df: pd.DataFrame):
    bathrooms_from_text = pd.to_numeric(
        df["bathrooms_text"].str.extract(r"(\d+\.?\d*)")[0], errors="coerce"
    )
    df["bathrooms"] = df["bathrooms"].combine_first(bathrooms_from_text)


def add_is_bathroom_shared_attribute(df: pd.DataFrame):
    def is_shared(pt: str) -> int:
        pt = pt.lower()
        if "shared" in pt:
            return 1
        return 0

    new_columns = (
        df["bathrooms_text"].map(is_shared, na_action="ignore").astype("Int64")
    )
    df["is_bathroom_shared"] = new_columns


def convert_text_to_sentiment(text: str) -> float:
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()

    text = contractions.fix(text)

    text = re.sub(r"[^\w\s!?-]", "", text)
    text = text.lower()

    blob = TextBlob(text)
    return float(blob.sentiment.polarity)


def convert_description_to_sentiment(df: pd.DataFrame):
    sentiment = df["description"].map(convert_text_to_sentiment, na_action="ignore")
    df["description_sentiment"] = sentiment


def convert_neighborhood_overview_to_sentiment(df: pd.DataFrame):
    sentiment = df["neighborhood_overview"].map(
        convert_text_to_sentiment, na_action="ignore"
    )
    df["neighborhood_overview_sentiment"] = sentiment


def add_amenity_count_attribute(df: pd.DataFrame):
    df["amenity_count"] = df["amenities"].apply(
        lambda x: len([a.strip().lower() for a in ast.literal_eval(x)])
        if pd.notna(x) and x.startswith("[")
        else 0
    )


def encode_amenities_binary(df: pd.DataFrame, amenities: list[str]):
    temp_amenities = df["amenities"].apply(
        lambda x: set(_normalize_text(a) for a in ast.literal_eval(x))
        if pd.notna(x) and isinstance(x, str) and x.startswith("[")
        else set()
    )

    new_columns_data = {}

    for amenity in amenities:
        search_key = _normalize_text(amenity)
        clean_col_name = (
            f"amenity_{search_key.strip().replace(' ', '_').replace('/', '_').lower()}"
        )
        new_columns_data[clean_col_name] = temp_amenities.apply(
            lambda tags: 1 if search_key in tags else 0
        ).astype("Int64")

    new_df = pd.DataFrame(new_columns_data, index=df.index)
    df[new_df.columns] = new_df


def convert_percentage_columns(df: pd.DataFrame, columns: list[str]):
    for col in columns:
        df[col] = df[col].str.strip("%").astype(float) / 100.0


def convert_tf_columns(df: pd.DataFrame, columns: list[str]):
    def map_tf(val: str) -> int:
        return 1 if val == "t" else 0

    for col in columns:
        df[col] = df[col].map(map_tf, na_action="ignore").astype("Int64")


def transform_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    percentage_attributes = ["host_response_rate", "host_acceptance_rate"]
    tf_attributes = ["host_is_superhost", "host_identity_verified", "instant_bookable"]
    drop = [
        "longitude",
        "latitude",
        "bathrooms_text",
        "description",
        "neighborhood_overview",
        "amenities",
    ]

    df = df.loc[:, INITIAL_FEATURES].copy()
    add_distance_to_centre_attribute(df)
    add_is_luxury_attribute(df)
    aggregate_property_type(df)
    fill_bathrooms_values_from_text(df)
    add_is_bathroom_shared_attribute(df)
    add_amenity_count_attribute(df)
    encode_amenities_binary(df, AMENITIES)
    convert_description_to_sentiment(df)
    convert_neighborhood_overview_to_sentiment(df)
    convert_percentage_columns(df, percentage_attributes)
    convert_tf_columns(df, tf_attributes)
    df.drop(columns=drop, inplace=True)

    return df
