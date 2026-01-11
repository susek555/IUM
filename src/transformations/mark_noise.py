import pandas as pd


def mark_noise_listings(
    sdf: pd.DataFrame, ldf: pd.DataFrame, features: pd.DataFrame
) -> pd.DataFrame:
    ldf["host_since"] = pd.to_datetime(ldf["host_since"], errors="coerce")
    today = pd.to_datetime(sdf["timestamp"]).max()
    is_mature = (ldf["host_since"] + pd.Timedelta(days=180)) <= today
    mature_ids = ldf.loc[is_mature, "id"]
    features["is_mature"] = features["id"].isin(mature_ids)

    features["is_active"] = (
        (features["listing_views_ltm"] > 30) |
        (features["conversion_rate_ltm"] > 0.005)
    )

    features["is_training_sample"] = (features["is_mature"] & features["is_active"]).astype(int)

    features.drop(columns=["is_mature", "is_active"], inplace=True)
    features.drop(columns=["host_since"], inplace=True)
