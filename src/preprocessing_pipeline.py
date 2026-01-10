import pandas as pd
import src.transformations.listings as listings_transforms
import src.transformations.sessions as sessions_transforms
from src.features import TARGET, INITIAL_FEATURES, AMENITIES


def transformation_pipeline(
    listings: pd.DataFrame, sessions: pd.DataFrame
) -> pd.DataFrame:
    # LISTINGS TRANSFORMATIONS
    listings = listings_transforms.select_features(listings, INITIAL_FEATURES, TARGET)

    listings = listings_transforms.add_is_luxury_attribute(listings)
    listings = listings_transforms.aggregate_property_type(listings)

    listings = listings_transforms.fill_bathrooms_values_from_text(listings)
    listings = listings_transforms.add_is_bathroom_shared_attribute(listings)

    listings = listings_transforms.encode_amenities_binary(listings, AMENITIES)

    listings = listings_transforms.convert_description_to_sentiment(listings)
    listings = listings_transforms.convert_neighborhood_overview_to_setiment(listings)

    listings = listings_transforms.convert_percentage_columns(
        listings, ["host_response_rate", "host_acceptance_rate"]
    )
    listings = listings_transforms.convert_tf_columns(
        listings, ["host_is_superhost", "host_identity_verified", "instant_bookable"]
    )

    listings = listings_transforms.drop_processed_columns(
        listings,
        ["bathrooms_text", "description", "neighbourhood_overview", "amenities"],
    )

    listings = listings_transforms.convert_price_to_number(listings)
    listings = listings_transforms.transform_price(listings)

    # SESSIONS TRANSFORMATIONS
    sessions = sessions_transforms.drop_browse_listings(sessions)
    sessions, newest_timestamp = sessions_transforms.get_newest_timestamp(sessions)
    sessions = sessions_transforms.drop_records_older_than_one_year(
        sessions, newest_timestamp
    )

    listings_sessions = sessions_transforms.get_views_last(sessions)
    listings_sessions = sessions_transforms.get_unique_viewers_last(
        sessions, listings_sessions
    )
    listings_sessions = sessions_transforms.get_conversion_rate(
        sessions, listings_sessions
    )
    listings_sessions = sessions_transforms.get_average_lead_time(
        sessions, listings_sessions
    )
    listings_sessions = sessions_transforms.get_average_booking_duration(
        sessions, listings_sessions
    )

    # MERGE LISTINGS WITH SESSIONS STATS
    listings = listings.merge(
        listings_sessions, left_on="id", right_on="listing_id", how="left"
    )
    listings.drop(columns=["listing_id"], inplace=True)
    listings.drop(columns=["id"], inplace=True)

    return listings
