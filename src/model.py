import pandas as pd
from dataclasses import dataclass

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from pathlib import Path

@dataclass
class Model:
    data: pd.DataFrame = None
    model: RandomForestRegressor = None

    def read_data_from_file(self) -> pd.DataFrame:
        data = pd.read_csv("./data/processed/preprocessed.csv")
        if "Unnamed: 0" in data.columns:
            data = data.drop(columns=["Unnamed: 0"])
        return data

    def fit(self, data: pd.DataFrame = None, n_estimators: int = 100, cv: int = 5) -> None:
        if data is None:
            data = self.read_data_from_file()
        self.data = data

        train_data = data.copy()

        X = train_data.drop(columns=["price"])
        y = train_data["price"]

        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1,
        )

        cv_scores = cross_val_score(
            self.model,
            X,
            y,
            cv=cv,
            scoring="neg_mean_squared_error",
        )

        mse_scores = -cv_scores
        print(f"Mean CV MSE: {mse_scores.mean():.2f} ± {mse_scores.std():.2f}")

        self.model.fit(X, y)

    def predict_all(self) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("Model is not trained. Call fit() before predict_all().")

        X_full = self.data.drop(columns=["price"])

        full_results = self.data.copy()
        full_results['predicted_price'] = self.model.predict(X_full)

        return full_results


if __name__ == "__main__":
    model = Model()
    model.fit(n_estimators=100, cv=5)
    results = model.predict_all()
    save_dir = Path("./data/predictions")
    save_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(save_dir / "predictions.csv")


