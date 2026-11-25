"""
visualization.py - Member 1 (common code)

Shared plots used across every member's figures (E and G marks).
Every function draws the figure, saves it when save_path is given
(figures live in reports/figures/ per config.FIGURES_DIR) and shows
it when it is not.
"""

import matplotlib

matplotlib.use("Agg")  # headless-safe; swap to an interactive backend in notebooks if wanted

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")


def _finish(fig, save_path: str = None) -> None:
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[visualization] saved {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame, save_path: str = None) -> None:
    """Correlation heatmap, saved in reports/figures/"""
    numeric = df.select_dtypes(include=[np.number])
    corr = numeric.corr()
    size = max(8, min(0.6 * len(corr.columns) + 4, 20))
    fig, ax = plt.subplots(figsize=(size, size * 0.8))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=len(corr.columns) <= 12,
                fmt=".2f", square=True, linewidths=0.5, ax=ax)
    ax.set_title("Correlation Heatmap")
    _finish(fig, save_path)


def plot_distribution(df: pd.DataFrame, column: str, save_path: str = None) -> None:
    """Column distribution (histogram/boxplot)"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if pd.api.types.is_numeric_dtype(df[column]):
        sns.histplot(df[column], kde=True, ax=axes[0], color="#2c7fb0")
        sns.boxplot(x=df[column], ax=axes[1], color="#7fcdbb")
        axes[0].set_ylabel("count")
    else:
        counts = df[column].value_counts().head(20)
        sns.barplot(x=counts.values, y=counts.index, ax=axes[0], color="#2c7fb0")
        axes[0].set_ylabel(column)
        counts.plot(kind="pie", autopct="%1.1f%%", ax=axes[1])
        axes[1].set_ylabel("")
    fig.suptitle(f"Distribution of {column}")
    _finish(fig, save_path)


def plot_predicted_vs_actual(y_true, y_pred, title: str, save_path: str = None) -> None:
    """Unified predicted-vs-actual plot used by all tasks (E)"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.4, s=18, color="#2c7fb0", label="predictions")
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", label="perfect prediction")
    ax.set_xlabel("actual")
    ax.set_ylabel("predicted")
    ax.set_title(title)
    ax.legend()
    _finish(fig, save_path)
