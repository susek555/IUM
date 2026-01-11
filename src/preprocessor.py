import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.tranformer = None

    def fit(self, X, y=None):
        cpy = X.copy()
        features = cpy.drop(columns=["price"])

        binary_cols = [
            col
            for col in features.columns
            if features[col].dropna().nunique() <= 2
            and set(features[col].dropna().unique()).issubset({0, 1})
        ]
        zero_columns = ["average_lead_time", "average_booking_duration"]
        num_columns = (
            features.select_dtypes(include=["number"])
            .columns.difference(binary_cols + zero_columns)
            .tolist()
        )
        ohe_columns = ["property_type", "room_type"]
        ord_columns = ["host_response_time"]

        self.transformer = ColumnTransformer(
            transformers=[
                (
                    "bin",
                    Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))]),
                    binary_cols,
                ),
                (
                    "zero",
                    Pipeline(
                        [
                            (
                                "imputer",
                                SimpleImputer(strategy="constant", fill_value=0.0),
                            )
                        ]
                    ),
                    zero_columns,
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
        self.transformer.fit(X)
        return self

    def transform(self, X):
        if self.transformer is None:
            raise RuntimeError("Use fit before transform")
        features_array = self.transformer.transform(X)
        features_names = self.transformer.get_feature_names_out()
        features = pd.DataFrame(features_array, columns=features_names, index=X.index)
        dataset = features.assign(price=X["price"].values)

        return dataset

