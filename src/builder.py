from sklearn.base import BaseEstimator, TransformerMixin

import src.transformations.listings as listings_transforms
import src.transformations.sessions as sessions_transforms


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
