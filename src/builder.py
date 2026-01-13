import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

import src.transformations.listings as listings_transforms
import src.transformations.sessions as sessions_transforms
import src.transformations.target as target_transforms
from src.service.app import predict_price
from src.service.model import PredictionData


class FeatureBuilder(BaseEstimator, TransformerMixin):
    def __init__(self, sessions):
        self.sessions = sessions

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        listings_features = listings_transforms.transform_pipeline(X)
        sessions_features = sessions_transforms.transform_pipeline(self.sessions)

        features = listings_features.merge(
            sessions_features, left_on="id", right_on="listing_id", how="left"
        )
        return features.drop(columns=["listing_id", "id"])


def generate_logs(listings, sessions, n):
    target = target_transforms.transform_pipeline(listings["price"])
    builder = FeatureBuilder(sessions)
    X = builder.fit_transform(listings)
    X["price"] = target
    indices = X.index.to_list()
    i = 0
    while i != n:
        idx = np.random.choice(indices)
        x = X.iloc[idx]
        if x.isna().sum() != 0:
            continue
        _ = predict_price(PredictionData(**x.to_dict()))
        i += 1


if __name__ == "__main__":
    listings = pd.read_csv("./data/data_new/listings.csv")
    sessions = pd.read_csv("./data/data_new/sessions.csv")
    generate_logs(listings, sessions, 1965)
