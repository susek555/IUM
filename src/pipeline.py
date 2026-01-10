import pandas as pd

import src.transformations.listings as listings_transforms
import src.transformations.sessions as sessions_transforms
import src.transformations.target as target_transforms


def transformation_pipeline(
    listings: pd.DataFrame, sessions: pd.DataFrame, target: pd.Series
) -> pd.DataFrame:
    listings_features = listings_transforms.transform_pipeline(listings)
    sessions_features = sessions_transforms.transform_pipeline(sessions)
    transformed_target = target_transforms.transform_pipeline(target)

    features = listings_features.merge(
        sessions_features, left_on="id", right_on="listing_id", how="left"
    )
    features.drop(columns=["listing_id", "id"], inplace=True)

    dataset = features.assign(price=transformed_target.values)

    return dataset
