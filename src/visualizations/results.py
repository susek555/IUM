import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import contextily as ctx
from scipy.spatial import ConvexHull
from matplotlib.colors import Normalize
from pathlib import Path


def visualize_results(differences: np.ndarray, clusters: np.ndarray) -> None:
    results = pd.DataFrame({"difference": differences, "cluster_id": clusters})

    plt.figure(figsize=(20, 10))

    results.boxplot(
        column="difference",
        by="cluster_id",
        grid=False,
        showfliers=False,
    )

    plt.axhline(0, linestyle="--", linewidth=1)
    plt.title("Price differences grouped by clusters")
    plt.suptitle("")
    plt.xlabel("cluster_id")
    plt.ylabel("price - predicted_price")

    plt.tight_layout()
    plt.show()


def save_price_spreads(
    differences: np.ndarray, clusters: np.ndarray, listings: pd.DataFrame
) -> None:
    dataset = pd.read_csv("./data/processed/dataset.csv")
    results = pd.read_csv("./data/predictions/predictions.csv")

    dataset["predicted_price"] = results["predicted_price"]
    dataset["price_spread"] = dataset["price"] - dataset["predicted_price"]

    save_dir = Path("./data/analysis")
    save_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        dataset[["latitude", "longitude", "cluster_id", "price_spread"]]
    ).to_csv(save_dir / "price_spreads.csv", index=False)


def visualize_map(
    differences: np.ndarray, clusters: np.ndarray, listings: pd.DataFrame
) -> None:
    results = pd.DataFrame({"price_spread": differences, "cluster_id": clusters})
    results = pd.concat([results, listings[["longitude", "latitude"]]], axis=1)

    fig, ax = plt.subplots(figsize=(12, 10))
    cluster_stats = results.groupby("cluster_id")["price_spread"].median().reset_index()

    v_abs = max(
        abs(cluster_stats["price_spread"].min()),
        abs(cluster_stats["price_spread"].max()),
    )
    norm = Normalize(vmin=-v_abs, vmax=v_abs)
    cmap = plt.get_cmap("RdYlGn")

    noise = results[results["cluster_id"] == -1]
    if not noise.empty:
        ax.scatter(
            noise["longitude"], noise["latitude"], c="black", s=5, alpha=0.3, zorder=1
        )

    unique_clusters = [c for c in results["cluster_id"].unique() if c != -1]

    for cluster_id in unique_clusters:
        median_val = cluster_stats.loc[
            cluster_stats["cluster_id"] == cluster_id, "price_spread"
        ].values[0]
        color = cmap(norm(median_val))

        cluster_data = results[results["cluster_id"] == cluster_id]
        points = cluster_data[["longitude", "latitude"]].values

        ax.scatter(
            points[:, 0],
            points[:, 1],
            c=[color],
            s=30,
            edgecolors="black",
            linewidths=0.2,
            zorder=3,
        )

        if len(points) >= 3:
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]

            ax.fill(
                hull_points[:, 0], hull_points[:, 1], color=color, alpha=0.3, zorder=2
            )

            hull_loop = np.vstack((hull_points, hull_points[0]))
            ax.plot(hull_loop[:, 0], hull_loop[:, 1], color=color, lw=1.5, zorder=2)

    try:
        ctx.add_basemap(ax, crs="EPSG:4326", source=ctx.providers.CartoDB.Positron)
    except Exception:
        pass

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Median price differences")

    ax.set_title("Price differences grouped by regions")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    save_price_spreads()
    visualize_results()
    visualize_map()
