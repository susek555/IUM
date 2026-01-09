# from sklearn.base import BaseEstimator, TransformerMixin

# class BathroomsImputer(BaseEstimator, TransformerMixin):
#     def fit(self, X, y=None):
#         return self

#     def transform(self, X):
#         X = X.copy()

#         extracted = (
#             X["bathrooms_text"]
#             .str.extract(r"(\d+\.?\d*)")
#             .astype(float)
#         )

#         X["bathrooms"] = X["bathrooms"].fillna(extracted)

#         X["bathrooms_text"] = X["bathrooms_text"].fillna(
#             X["bathrooms"].astype(str) + " bath"
#         )

#         return X