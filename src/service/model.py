from typing import Literal

from pydantic import BaseModel, confloat, conint

Binary = conint(ge=0, le=1)
Percentage = confloat(ge=0.0, le=1.0)
ReviewScore = confloat(ge=0.0, le=5.0)
SentimentScore = confloat(ge=-1.0, le=1.0)

type PropertyType = Literal["rental_unit", "condo", "hotel", "home", "other"]
type RoomType = Literal["Entire home/apt", "Private room", "Hotel room", "Shared room"]
type HostResponseTime = Literal[
    "a few days or more", "witihin a day", "within a few hours", "within an hour"
]


class PredictionData(BaseModel):
    property_type: PropertyType
    room_type: RoomType
    accommodates: int
    bathrooms: int
    bedrooms: int
    beds: int
    host_response_time: HostResponseTime
    host_response_rate: Percentage
    host_acceptance_rate: Percentage
    host_is_superhost: Binary
    host_identity_verifed: Binary
    review_scores_rating: Percentage
    number_of_reviews: int
    minimum_nights: int
    maximum_nights: int
    instant_bookable: Binary
    distance_to_centre: float
    is_luxury: Binary
    is_bathroom_shared: Binary
    amenity_dishwasher: Binary
    amenity_iron: Binary
    amenity_toaster: Binary
    amenity_oven: Binary
    amenity_kitchen: Binary
    amenity_microwave: Binary
    amenity_crib: Binary
    amenity_dinning_table: Binary
    amenity_free_dryer_in_unit: Binary
    amenity_pack_n_playtravel_crib: Binary
    amenity_count: int
    description_sentiment: SentimentScore
    neighborhood_overview_sentiment: SentimentScore
    listings_views_ltm: int
    conversion_rate_ltm: Percentage
    average_lead_time: float
    average_booking_duration: float
    price: float
