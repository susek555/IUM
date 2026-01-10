import pandas as pd
import numpy as np
from datetime import datetime


def drop_browse_listings(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["action"] != "browse_listings"]


def get_newest_timestamp(df: pd.DataFrame) -> tuple[pd.DataFrame, datetime]:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["booking_date"] = pd.to_datetime(df["booking_date"], errors="coerce")
    newest_timestamp = df["timestamp"].max()
    return df, newest_timestamp


def drop_records_older_than_one_year(
    df: pd.DataFrame, newest_timestamp: datetime
) -> pd.DataFrame:
    threshold = newest_timestamp - pd.Timedelta(days=365)
    return df[df["timestamp"] >= threshold]


def get_views_last(df: pd.DataFrame) -> pd.DataFrame:
    views_series = df.loc[
        (df["action"] == "view_listing") & (df["listing_id"].notna()), "listing_id"
    ]

    counts = views_series.value_counts()
    all_unique_listings = df["listing_id"].dropna().unique()

    result_df = counts.reindex(all_unique_listings, fill_value=0).reset_index()
    result_df.columns = ["listing_id", "listing_views_last_1y"]
    result_df["listing_views_last_1y"] = result_df["listing_views_last_1y"].astype(int)
    return result_df


def get_unique_viewers_last(
    df: pd.DataFrame, listings_df: pd.DataFrame
) -> pd.DataFrame:
    unique_counts = (
        df.loc[
            (df["action"] == "view_listing") & (df["listing_id"].notna()),
            ["listing_id", "user_id"],
        ]
        .drop_duplicates()["listing_id"]
        .value_counts()
    )

    listings_df["unique_viewers_last_1y"] = (
        listings_df["listing_id"].map(unique_counts).fillna(0).astype(int)
    )

    return listings_df

def get_conversion_rate(df: pd.DataFrame, listings_df: pd.DataFrame) -> pd.DataFrame:
    booking_counts = df.loc[
        df["action"] == "book_listing",
        "listing_id"
    ].value_counts()

    num_bookings = listings_df["listing_id"].map(booking_counts).fillna(0)

    listings_df["conversion_rate_last_1y"] = num_bookings / listings_df["listing_views_last_1y"]
    listings_df["conversion_rate_last_1y"] = listings_df["conversion_rate_last_1y"].fillna(0.0)
    listings_df["conversion_rate_last_1y"] = listings_df["conversion_rate_last_1y"].replace([np.inf, -np.inf], 0.0)

    return listings_df

def get_average_lead_time(df: pd.DataFrame, listings_df: pd.DataFrame) -> pd.DataFrame:
    bookings = df.loc[
        (df["action"] == "book_listing") &
        (df["booking_date"].notna()),
        ["listing_id", "timestamp", "booking_date"]
    ].copy()

    bookings["lead_time_days"] = (
        bookings["booking_date"] - bookings["timestamp"].dt.normalize()
    ).dt.days

    avg_lead_time = bookings.groupby("listing_id")["lead_time_days"].mean()

    listings_df["average_lead_time"] = (
        listings_df["listing_id"]
        .map(avg_lead_time)
        .fillna(0)
    )

    return listings_df


def get_average_booking_duration(df: pd.DataFrame, listings_df: pd.DataFrame) -> pd.DataFrame:
    durations = df.loc[
        (df["action"] == "book_listing") &
        (df["booking_duration"].notna()),
        ["listing_id", "booking_duration"]
    ]

    avg_duration = durations.groupby("listing_id")["booking_duration"].mean()

    listings_df["average_booking_duration"] = (
        listings_df["listing_id"]
        .map(avg_duration)
        .fillna(0)
    )

    return listings_df