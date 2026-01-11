from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


def preprocess() -> pd.DataFrame:
    dataset = pd.read_csv("./data/processed/dataset.csv")
    features = dataset.drop(columns=["price"])
    binary_cols = [
        col
        for col in features.columns
        if features[col].dropna().nunique() <= 2
        and set(features[col].dropna().unique()).issubset({0, 1})
    ]

    fill_with_zero_columns = ["average_lead_time", "average_booking_duration"]

    num_columns = (
        features.select_dtypes(include=["number"])
        .columns.difference(binary_cols + fill_with_zero_columns)
        .tolist()
    )

    ohe_columns = ["property_type", "room_type"]

    ord_columns = ["host_response_time"]


    transformer = ColumnTransformer(
        transformers=[
            (
                "bin",
                Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))]),
                binary_cols,
            ),
            (
                "zero",
                Pipeline([("imputer"), SimpleImputer(strategy="constant", fill_value=0.0)]),
                fill_with_zero_columns,
            ),
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                num_columns,
            ),
            (
                "ohe",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                ohe_columns,
            ),
            (
                "ord",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "ord",
                            OrdinalEncoder(
                                categories=[
                                    [
                                        "a few days or more",
                                        "within a day",
                                        "within a few hours",
                                        "within an hour",
                                    ]
                                ],
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                ord_columns,
            ),
        ],
        remainder="drop",
    )
    features_array = transformer.fit_transform(features)
    features_names = transformer.get_feature_names_out()
    features_df = pd.DataFrame(
        features_array, columns=features_names, index=features.index
    )

    dataset = features_df.assign(price=dataset["price"].values)
    return dataset