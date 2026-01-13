import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import contextily as ctx
from scipy.spatial import ConvexHull
import matplotlib.patheffects as patheffects
from matplotlib.colors import Normalize
from pathlib import Path


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


def visualize_results(
    differences: np.ndarray, clusters: np.ndarray, ax=None, title=None
) -> None:
    results = pd.DataFrame({"difference": differences, "cluster_id": clusters})

    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 10))

    results.boxplot(
        column="difference", by="cluster_id", grid=False, showfliers=False, ax=ax
    )

    ax.axhline(0, linestyle="--", color="red", linewidth=1, alpha=0.6)
    ax.set_title(title if title else "Price differences grouped by clusters")
    ax.set_xlabel("cluster_id")
    ax.set_ylabel("price - predicted_price")

    if ax.get_figure():
        ax.get_figure().suptitle("")


def visualize_results_compare(
    diffs1: np.ndarray,
    diffs2: np.ndarray,
    clusters: np.ndarray,
    label1: str = "Linear",
    label2: str = "Random Forest",
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True)

    visualize_results(diffs1, clusters, ax=axes[0], title=label1)
    visualize_results(diffs2, clusters, ax=axes[1], title=label2)

    plt.tight_layout()
    plt.show()


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

        centroid = points.mean(axis=0)
        ax.text(
            centroid[0],
            centroid[1],
            str(cluster_id),
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="black",
            zorder=4,
            path_effects=[patheffects.withStroke(linewidth=3, foreground="white")],
        )

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
