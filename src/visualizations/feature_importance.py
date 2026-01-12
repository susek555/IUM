import pandas as pd
import matplotlib.pyplot as plt

def visualize_feature_importance_tables(pipe):
    importances = pipe.named_steps['regressor'].feature_importances_
    feature_names = pipe.named_steps['preprocessor'].transformer.get_feature_names_out()

    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False)

    top_10 = df.head(10).copy()
    top_10['importance'] = top_10['importance'].round(4)

    bottom_10 = df.tail(10).copy()
    bottom_10['importance'] = bottom_10['importance'].round(4)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    for i, (data, title) in enumerate([
        (top_10, "Most Valuable Features (Top 10)"),
        (bottom_10, "Least Valuable Features (Bottom 10)")
    ]):
        ax = axes[i]
        ax.axis('off')
        ax.set_title(title, fontweight="bold", pad=20)

        table = ax.table(
            cellText=data.values,
            colLabels=data.columns,
            loc='center',
            cellLoc='left'
        )

        # Stylizacja tabeli
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

    plt.tight_layout()
    plt.show()

def plot_importance_distribution(pipe):
    importances = pipe.named_steps['regressor'].feature_importances_
    feature_names = pipe.named_steps['preprocessor'].transformer.get_feature_names_out()

    df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    df = df.sort_values('importance', ascending=False).reset_index(drop=True)

    plt.figure(figsize=(12, 6))

    plt.plot(df.index, df['importance'], marker='o', linestyle='-', color='#2c3e50', markersize=4)

    plt.yscale('log')

    cutoff = 0.005
    plt.axhline(y=cutoff, color='red', linestyle='--', alpha=0.6, label=f'Cutoff ({cutoff})')

    num_above = len(df[df['importance'] > cutoff])

    plt.title("Attribute importance distribution", fontsize=14, pad=15)
    plt.xlabel("Attributes ranking (from most important)", fontsize=12)
    plt.ylabel("Imp[ortance (log scale)", fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()

    plt.annotate(f'Important features: {num_above}',
                 xy=(num_above, cutoff), xytext=(num_above + 5, cutoff * 5),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

    plt.tight_layout()
    plt.show()
