import numpy as np
import pandas as pd


def drop_browse_listings(df: pd.DataFrame) -> pd.DataFrame:
    df[df["action"] != "browse_listings"]


def convert_timestamps_to_dates(df: pd.DataFrame) -> pd.DataFrame:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["booking_date"] = pd.to_datetime(df["booking_date"], errors="coerce")


def drop_records_older_than_one_year(
    df: pd.DataFrame,
) -> pd.DataFrame:
    threshold = df["timestamp"].max() - pd.Timedelta(days=365)
    df[df["timestamp"] >= threshold]


def get_views_last(df: pd.DataFrame) -> pd.DataFrame:
    views_series = df.loc[
        (df["action"] == "view_listing") & (df["listing_id"].notna()), "listing_id"
    ]

    counts = views_series.value_counts()
    all_unique_listings = df["listing_id"].dropna().unique()

    result_df = counts.reindex(all_unique_listings, fill_value=0).reset_index()
    result_df.columns = ["listing_id", "listing_views_ltm"]
    result_df["listing_views_ltm"] = result_df["listing_views_ltm"].astype(int)
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

    listings_df["unique_viewers_ltm"] = (
        listings_df["listing_id"].map(unique_counts).fillna(0).astype(int)
    )


def get_conversion_rate(df: pd.DataFrame, listings_df: pd.DataFrame) -> pd.DataFrame:
    booking_counts = df.loc[df["action"] == "book_listing", "listing_id"].value_counts()
    num_bookings = listings_df["listing_id"].map(booking_counts).fillna(0)
    views = listings_df["listing_views_ltm"]

    listings_df["conversion_rate_ltm"] = np.where(
        views > 0, num_bookings / views, np.nan
    )


def get_average_lead_time(df: pd.DataFrame, listings_df: pd.DataFrame) -> pd.DataFrame:
    bookings = df.loc[
        (df["action"] == "book_listing") & (df["booking_date"].notna()),
        ["listing_id", "timestamp", "booking_date"],
    ].copy()

    bookings["lead_time_days"] = (
        bookings["booking_date"] - bookings["timestamp"].dt.normalize()  # pyright: ignore
    ).dt.days

    avg_lead_time = bookings.groupby("listing_id")["lead_time_days"].mean()

    listings_df["average_lead_time"] = listings_df["listing_id"].map(
        avg_lead_time, na_action="ignore"
    )


def get_average_booking_duration(
    df: pd.DataFrame, listings_df: pd.DataFrame
) -> pd.DataFrame:
    durations = df.loc[
        (df["action"] == "book_listing") & (df["booking_duration"].notna()),
        ["listing_id", "booking_duration"],
    ]

    avg_duration = durations.groupby("listing_id")["booking_duration"].mean()

    listings_df["average_booking_duration"] = listings_df["listing_id"].map(
        avg_duration, na_action="ignore"
    )


def transform_pipeline(sdf: pd.DataFrame) -> pd.DataFrame:
    sdf.copy()
    drop_browse_listings(sdf)
    convert_timestamps_to_dates(sdf)
    drop_records_older_than_one_year(sdf)

    ldf = get_views_last(sdf)
    get_unique_viewers_last(sdf, ldf)
    get_conversion_rate(sdf, ldf)
    get_average_lead_time(sdf, ldf)
    get_average_booking_duration(sdf, ldf)
    return ldf
