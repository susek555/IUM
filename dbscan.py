import csv
import numpy as np
from dataclasses import dataclass
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt


@dataclass
class Listing:
    id: int
    coords: tuple[float, float]
    region_label: int = -1


def read_coordinates() -> tuple[list[Listing], np.ndarray]:
    """
    Reads a listings.csv file and returns a NumPy array of coordinates.
    Return format:
    - A list of Listing objects containing id and coordinates.
    - A NumPy array of shape (n, 2) where n is the number of listings, and each row contains [latitude, longitude].
    """
    listings = []
    coords = []
    with open("./data/listings.csv", 'r', newline='', encoding='utf-8') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            listing = Listing(
                id=int(row['id']),
                coords=(float(row['latitude']), float(row['longitude']))
            )
            listings.append(listing)
            coords.append([listing.coords[0], listing.coords[1]])

    return listings, np.array(coords)


def group_listings_by_region(eps_meters: float, min_samples: int) -> list[Listing]:
    """
    Groups listings by region using DBSCAN clustering algorithm.
    Returns a list of Listing objects with updated region_label.
    """

    listings, coords = read_coordinates()
    coords_rad = np.radians(coords)

    EARTH_RADIUS_METERS = 6371000
    eps = eps_meters / EARTH_RADIUS_METERS

    db = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric='haversine'
    )
    labels = db.fit_predict(coords_rad)

    for listing, label in zip(listings, labels):
        listing.region_label = label

    return listings

def visualize_regions(listings: list[Listing]) -> None:
    lats = np.array([l.coords[0] for l in listings])
    lons = np.array([l.coords[1] for l in listings])
    labels = np.array([l.region_label for l in listings])

    unique_labels = set(labels)

    plt.figure(figsize=(10, 8))

    for label in unique_labels:
        mask = labels == label

        if label == -1:
            plt.scatter(
                lons[mask],
                lats[mask],
                c='lightgray',
                s=10,
                label='Noise'
            )
        else:
            plt.scatter(
                lons[mask],
                lats[mask],
                s=20,
                label=f'Cluster {label}'
            )

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("DBSCAN – klastry geolokalizacyjne")
    plt.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    listings = group_listings_by_region(eps_meters=120, min_samples=5)
    visualize_regions(listings)