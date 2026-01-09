import numpy as np
import pandas as pd
import ast

from src.features import INITIAL_FEATURES, TARGET, AMENITIES


def select_features(
    df: pd.DataFrame,
    select_features: list[str],
    target: list[str],
) -> pd.DataFrame:
    return df[select_features + target]


def convert_price_to_number(df: pd.DataFrame) -> pd.DataFrame:
    df["price"] = df["price"].str.replace("$", "").str.replace(",", "").astype(float)
    return df


def transform_price(df: pd.DataFrame) -> pd.DataFrame:
    df["price"] = np.log1p(df["price"])
    return df


def add_is_luxury_attribute(df: pd.DataFrame) -> pd.DataFrame:
    def is_luxury(pt: str) -> int:
        pt = pt.lower()
        if "loft" in pt or "villa" in pt or "boutique hotel" in pt:
            return 1
        return 0

    new_columns = df["property_type"].map(is_luxury)
    df["is_luxury"] = new_columns
    return df


def aggregate_property_type(df: pd.DataFrame) -> pd.DataFrame:
    def map_proprerty_type(pt: str) -> str:
        pt = pt.lower()
        if "rental unit" in pt:
            return "Rental unit"
        if "condo" in pt:
            return "Condo"
        if "home" in pt or "house" in pt:
            return "Home"
        if "hotel" in pt or "hostel" in pt:
            return "Hotel"
        return "Other"

    df["property_type"] = df["property_type"].map(map_proprerty_type)
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
        if pd.isna(pt):
            return 0
        pt = pt.lower()
        if "shared" in pt:
            return 1
        return 0

    new_columns = df["bathrooms_text"].map(is_shared)
    df["is_bathroom_shared"] = new_columns
    return df


def drop_bathroom_text_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=["bathrooms_text"])
    return df


def normalize_text(text: str) -> str:
    return (
        text.lower()
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2019", "'")
        .strip()
    )


def encode_amenities_binary(df: pd.DataFrame) -> pd.DataFrame:
    temp_amenities = df["amenities"].apply(
        lambda x: set(normalize_text(a) for a in ast.literal_eval(x))
        if pd.notna(x) and isinstance(x, str) and x.startswith("[")
        else set()
    )

    new_columns_data = {}

    for amenity in AMENITIES:
        search_key = normalize_text(amenity)
        clean_col_name = (
            f"amenity_{amenity.strip().replace(' ', '_').replace('/', '_').lower()}"
        )
        new_columns_data[clean_col_name] = temp_amenities.apply(
            lambda tags: 1 if search_key in tags else 0
        )

    new_df = pd.DataFrame(new_columns_data, index=df.index)
    df = pd.concat([df, new_df], axis=1)

    df = df.drop(columns=["amenities"])
    return df


def transformation_pipeline(listings: pd.DataFrame) -> pd.DataFrame:
    listings = select_features(listings, INITIAL_FEATURES, TARGET)

    listings = add_is_luxury_attribute(listings)

    listings = fill_bathrooms_values_from_text(listings)
    listings = add_is_bathroom_shared_attribute(listings)
    listings = drop_bathroom_text_column(listings)

    listings = convert_price_to_number(listings)
    listings = transform_price(listings)

    listings = encode_amenities_binary(listings)

    return listings
