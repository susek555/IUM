from sklearn.base import BaseEstimator, TransformerMixin, clone
import pandas as pd

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, estimator, percent, ohe_features):
        self.estimator = estimator
        self.percent = percent
        self.ohe_features = ohe_features

        
    def fit(self, X, y):
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y)
        
        feature_names = X.columns.tolist()
        importances = pd.Series(self.estimator_.feature_importances_, index=feature_names)

        groups = {}
        for col in feature_names:
            matched = False
            for feature in self.ohe_features:
                prefix = "ohe__" + feature
                if col.startswith(prefix):
                    groups.setdefault(prefix, []).append(col)
                    matched = True
                    break
            if not matched:
                groups[col] = [col]

        group_importances = {
            g: importances[col].sum() for g, col in groups.items()
        }

        sorted_group_importances = {
            k: v for k, v in sorted(group_importances.items(), key=lambda item: item[1], reverse=True)
        }

        selected_groups = {}
        for g, v in sorted_group_importances.items():
            if (len(selected_groups.keys()) / len(group_importances.keys())) < self.percent:
                selected_groups[g] = v

        self.selected_features_ = [
            col 
            for g in selected_groups
            for col in groups[g]
        ] 
        return self

    def transform(self, X):
        return X[self.selected_features_]