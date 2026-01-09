_accomodation_features = [
    "property_type",
    "room_type",
    "accommodates",
    "bathrooms",
    "bathrooms_text",
    "bedrooms",
    "beds",
    "amenities",
]

_text_features = [
    "description",
    "neighborhood_overview",
]

_location_features = [
    "latitude",
    "longitude",
]

_host_trust_features = [
    "host_response_time",
    "host_response_rate",
    "host_acceptance_rate",
    "host_is_superhost",
    "host_identity_verified",
    "review_scores_rating",
    "number_of_reviews",
]

_availability_features = [
    "minimum_nights",
    "maximum_nights",
    "instant_bookable",
]

INITIAL_FEATURES = (
    _accomodation_features
    + _text_features
    + _location_features
    + _host_trust_features
    + _availability_features
)

AMENITIES = [
    "dishwasher",
    "iron",
    "toaster",
    "oven",
    "kitchen",
    "microwave",
    "crib",
    "dining table",
    "Free dryer \u2013 In unit",
    "Pack \u2019n play/Travel crib",
]

TARGET = ["price"]
