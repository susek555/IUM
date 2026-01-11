from pathlib import Path

import pandas as pd

import src.transformations.listings as listings_transforms
import src.transformations.sessions as sessions_transforms
import src.transformations.target as target_transforms
import src.transformations.mark_noise as mark_noise


def main() -> None:
    listings = pd.read_csv("./data/raw/listings.csv")
    sessions = pd.read_csv("./data/raw/sessions.csv")
    target = listings["price"]

    listings_features = listings_transforms.transform_pipeline(listings)
    sessions_features = sessions_transforms.transform_pipeline(sessions)
    transformed_target = target_transforms.transform_pipeline(target)

    features = listings_features.merge(
        sessions_features, left_on="id", right_on="listing_id", how="left"
    )
    mark_noise.mark_noise_listings(sessions, listings_features, features)
    features.drop(columns=["listing_id", "id"], inplace=True)

    dataset = features.assign(price=transformed_target.values)
    save_dir = Path("./data/processed")
    save_dir.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(save_dir / "dataset.csv")


if __name__ == "__main__":
    main()
