import ast
from collections import Counter
from typing import Counter as CounterType

import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import MultiLabelBinarizer


def _extract_amenities_list(df: pd.DataFrame) -> pd.Series:
    amenities_list = df["amenities"].apply(
        lambda x: [a.strip().lower() for a in ast.literal_eval(x)]
        if pd.notna(x) and x.startswith("[")
        else []
    )
    return amenities_list


def get_amenities_counter(df: pd.DataFrame) -> CounterType[str]:
    amenities_list = _extract_amenities_list(df)

    all_amenities = sum(amenities_list, [])
    return Counter(all_amenities)


def calc_amenities_correlation(df: pd.DataFrame, min_freq: int = 10) -> pd.Series:
    df["price"] = (
        df["price"].astype(str).str.replace(r"[$,]", "", regex=True).astype(float)
    )

    amenities_list = _extract_amenities_list(df)

    mlb = MultiLabelBinarizer(sparse_output=True)
    amenities_matrix = mlb.fit_transform(amenities_list)

    amenities_df = pd.DataFrame.sparse.from_spmatrix(
        amenities_matrix, columns=mlb.classes_, index=df.index
    )

    counts = amenities_df.sum()
    valid_cols = counts[counts >= min_freq].index
    amenities_df = amenities_df[valid_cols]

    correlation = amenities_df.sparse.to_dense().corrwith(
        df["price"], method="spearman"
    )

    return correlation.sort_values(ascending=False)


def calc_amenities_mutual_info(df: pd.DataFrame, min_freq: int = 50) -> pd.DataFrame:
    df["price"] = (
        df["price"].astype(str).str.replace(r"[$,]", "", regex=True).astype(float)
    )

    amenities_list = _extract_amenities_list(df)

    mlb = MultiLabelBinarizer(sparse_output=True)
    amenities_matrix = mlb.fit_transform(amenities_list)
    amenities_df = pd.DataFrame.sparse.from_spmatrix(
        amenities_matrix, columns=mlb.classes_, index=df.index
    )

    counts = amenities_df.sum()
    valid_cols = counts[counts >= min_freq].index
    X = amenities_df[valid_cols]
    y = df["price"]

    mi_scores = mutual_info_regression(X, y, discrete_features=True, random_state=42)

    results = []
    for idx, col in enumerate(X.columns):
        has_amenity = X[col] == 1
        mean_diff = y[has_amenity].mean() - y[~has_amenity].mean()

        results.append(
            {
                "amenity": col,
                "mutual_info": mi_scores[idx],
                "price_diff": mean_diff,
                "count": counts[col],
            }
        )

    return (
        pd.DataFrame(results)
        .sort_values(by="mutual_info", ascending=False)
        .set_index("amenity")
    )


if __name__ == "__main__":
    listings = pd.read_csv("./data/raw/listings.csv")
    corr = calc_amenities_correlation(listings, min_freq=50)

    print("Top 10 udogodnień podbijających cenę:")
    print(corr.head(10))

    print("\nTop 10 udogodnień w tanich ofertach:")
    print(corr.tail(10))

    print(f"Średnia wartość korelacji: {corr.mean():.4f}")

    mi_df = calc_amenities_mutual_info(listings, min_freq=50)

    print("Top 10 najważniejszych udogodnień (wg Mutual Information):")
    print(mi_df.head(10))
