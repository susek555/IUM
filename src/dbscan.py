import contextily as ctx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from sklearn.cluster import HDBSCAN


def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, usecols=["id", "latitude", "longitude"])
    return df


def group_listings_by_region(
    df: pd.DataFrame, min_cluster_size: int = 15, min_samples=3
) -> tuple[pd.DataFrame, float]:
    coords_rad = np.radians(df[["latitude", "longitude"]].to_numpy())

    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="haversine",
        cluster_selection_method="eom",
        copy=False,
    )

    df["cluster"] = clusterer.fit_predict(coords_rad)

    num_clusters = df["cluster"].nunique() - (1 if -1 in df["cluster"].values else 0)
    print(f"Dla min_cluster_size={min_cluster_size} znaleziono {num_clusters} klastrów")

    return df


def aggregate_results(
    df: pd.DataFrame,
    min_cluster_sizes: list[int],
    min_samples = 3
) -> list[tuple[int, pd.DataFrame]]:
    return [
        (i, group_listings_by_region(df.copy(), min_cluster_size=i, min_samples=min_samples))
        for i in min_cluster_sizes
    ]


def visualize_regions(ax: np.ndarray, df: pd.DataFrame, title: str) -> None:
    noise = df[df["cluster"] == -1]
    ax.scatter(
        noise["longitude"],
        noise["latitude"],
        c="black",
        s=15,
        alpha=0.6,
        label="Szum (Noise)",
        zorder=1,
    )

    unique_labels = sorted([label for label in df["cluster"].unique() if label != -1])

    cmap = plt.get_cmap("tab20")

    for i, label in enumerate(unique_labels):
        color = cmap(i % 20)

        cluster_data = df[df["cluster"] == label]
        points = cluster_data[["longitude", "latitude"]].values

        ax.scatter(
            points[:, 0],
            points[:, 1],
            c=[color],
            s=25,
            label=f"Region {label}" if i < 10 else None,
            zorder=3,
        )

        if len(points) >= 3:
            hull = ConvexHull(points)

            hull_points = points[hull.vertices]

            ax.fill(
                hull_points[:, 0], hull_points[:, 1], color=color, alpha=0.2, zorder=2
            )

            hull_loop = np.vstack((hull_points, hull_points[0]))
            ax.plot(
                hull_loop[:, 0],
                hull_loop[:, 1],
                color=color,
                linestyle="--",
                linewidth=1,
                zorder=2,
            )

    try:
        ctx.add_basemap(ax, crs="EPSG:4326", source=ctx.providers.CartoDB.Positron)
    except Exception:
        print("Brak połączenia z mapą online.")

    ax.set_title(title)


def visualize_regions_grid(
    evals: list[tuple[int, float, pd.DataFrame]],
    ncols: int = 3,
    figsize_per_plot=(5, 4),
) -> None:
    n = len(evals)
    ncols = max(1, ncols)
    nrows = (n + ncols - 1) // ncols

    _, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows),
        squeeze=False,
    )

    for idx, (mcs, df_copy) in enumerate(evals):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row][col]

        visualize_regions(ax, df_copy, title=f"min_cluster_size={mcs}")

    for idx in range(n, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row][col].axis("off")

    plt.tight_layout()
    plt.show()
