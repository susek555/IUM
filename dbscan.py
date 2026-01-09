import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import contextily as ctx
from sklearn.cluster import HDBSCAN
from scipy.spatial import ConvexHull

def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, usecols=['id', 'latitude', 'longitude'])
    return df

def group_listings_by_region(df: pd.DataFrame, min_cluster_size: int = 15) -> pd.DataFrame:
    coords_rad = np.radians(df[['latitude', 'longitude']].to_numpy())

    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=1,
        metric='haversine',
        cluster_selection_method='eom'
    )

    df['cluster'] = clusterer.fit_predict(coords_rad)

    num_clusters = df['cluster'].nunique() - (1 if -1 in df['cluster'].values else 0)
    print(f"Znaleziono {num_clusters} klastrów.")

    return df

def visualize_regions(df: pd.DataFrame, min_cluster_size: int) -> None:
# def visualize_regions(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14, 14))
    noise = df[df['cluster'] == -1]
    ax.scatter(
        noise['longitude'],
        noise['latitude'],
        c='black',
        s=15,
        alpha=0.6,
        label='Szum (Noise)',
        zorder=1
    )

    unique_labels = sorted([l for l in df['cluster'].unique() if l != -1])

    cmap = plt.get_cmap('tab20')

    for i, label in enumerate(unique_labels):
        color = cmap(i % 20)

        cluster_data = df[df['cluster'] == label]
        points = cluster_data[['longitude', 'latitude']].values

        ax.scatter(
            points[:, 0],
            points[:, 1],
            c=[color],
            s=25,
            label=f'Region {label}' if i < 10 else None,
            zorder=3
        )

        if len(points) >= 3:
            hull = ConvexHull(points)

            hull_points = points[hull.vertices]

            ax.fill(
                hull_points[:, 0],
                hull_points[:, 1],
                color=color,
                alpha=0.2,
                zorder=2
            )

            hull_loop = np.vstack((hull_points, hull_points[0]))
            ax.plot(
                hull_loop[:, 0],
                hull_loop[:, 1],
                color=color,
                linestyle='--',
                linewidth=1,
                zorder=2
            )

    try:
        ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.CartoDB.Positron)
    except Exception:
        print("Brak połączenia z mapą online.")

    plt.title(f"Analiza regionów: {len(unique_labels)} wykrytych stref (Szum na czarno)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    plt.tight_layout()
    # plt.show()
    plt.savefig(f"./results/regions_analysis_min_cluster_{min_cluster_size}.png", dpi=300)


if __name__ == "__main__":
    df_listings = load_data("./data/listings.csv")

    for i in range(10, 25):
        df_listings = group_listings_by_region(df_listings, min_cluster_size=i)
        visualize_regions(df_listings, min_cluster_size=i)

        print(f"Wyniki analizy zapisane dla min_cluster_size={i}.\n")