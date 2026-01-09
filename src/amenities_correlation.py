import pandas as pd
import ast
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_selection import mutual_info_regression

def get_unique_amenities(df: pd.DataFrame) -> set[str]:
    amenities_series = df["amenities"].dropna().apply(
        lambda x: [a.strip().lower() for a in ast.literal_eval(x)]
        if x.startswith("[")
        else []
    )

    unique_amenities = set()
    for amenities_list in amenities_series:
        unique_amenities.update(amenities_list)

    return unique_amenities

def calc_amenities_correlation(path: str, min_freq: int = 10) -> pd.Series:
    listings = pd.read_csv(path)

    listings["price"] = (
        listings["price"].astype(str).str.replace(r"[$,]", "", regex=True).astype(float)
    )

    amenities_list = listings["amenities"].apply(
        lambda x: [a.strip().lower() for a in ast.literal_eval(x)]
        if pd.notna(x) and x.startswith("[")
        else []
    )

    mlb = MultiLabelBinarizer(sparse_output=True)
    amenities_matrix = mlb.fit_transform(amenities_list)

    amenities_df = pd.DataFrame.sparse.from_spmatrix(
        amenities_matrix, columns=mlb.classes_, index=listings.index
    )

    counts = amenities_df.sum()
    valid_cols = counts[counts >= min_freq].index
    amenities_df = amenities_df[valid_cols]

    correlation = amenities_df.sparse.to_dense().corrwith(
        listings["price"], method="spearman"
    )

    return correlation.sort_values(ascending=False)


def calc_amenities_mutual_info(path: str, min_freq: int = 50) -> pd.DataFrame:
    listings = pd.read_csv(path)

    listings["price"] = (
        listings["price"].astype(str).str.replace(r"[$,]", "", regex=True).astype(float)
    )

    amenities_list = listings["amenities"].apply(
        lambda x: [a.strip().lower() for a in ast.literal_eval(x)]
        if pd.notna(x) and x.startswith("[")
        else []
    )

    mlb = MultiLabelBinarizer(sparse_output=False)
    amenities_matrix = mlb.fit_transform(amenities_list)
    amenities_df = pd.DataFrame(amenities_matrix, columns=mlb.classes_, index=listings.index)

    counts = amenities_df.sum()
    valid_cols = counts[counts >= min_freq].index
    X = amenities_df[valid_cols]
    y = listings["price"]

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
    corr = calc_amenities_correlation("./data/raw/listings.csv", min_freq=50)

    print("Top 10 udogodnień podbijających cenę:")
    print(corr.head(10))

    print("\nTop 10 udogodnień w tanich ofertach:")
    print(corr.tail(10))

    print(f"Średnia wartość korelacji: {corr.mean():.4f}")

    mi_df = calc_amenities_mutual_info("./data/raw/listings.csv", min_freq=50)

    print("Top 10 najważniejszych udogodnień (wg Mutual Information):")
    print(mi_df.head(10))
