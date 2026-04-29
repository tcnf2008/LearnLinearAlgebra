"""用矩阵分解演示推荐系统中的评分预测。

运行方式：
    .venv/bin/python topics/09-applications/05-recommender-matrix-factorization/demo.py

脚本会对一个稀疏用户-物品评分矩阵做 ALS 分解，输出预测评分和推荐结果，并保存可视化图。
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


USERS = ["Xiao Lin", "A Zhou", "Mia", "Noah", "Chen", "Iris"]
ITEMS = ["Sci-Fi", "Action", "Comedy", "Romance", "Doc", "Anime", "Mystery"]
RATINGS = np.array(
    [
        [5, 4, np.nan, 2, np.nan, 3, 4],
        [4, 5, 3, np.nan, 1, np.nan, 5],
        [np.nan, 2, 5, 4, 3, 5, np.nan],
        [1, np.nan, 4, 5, 4, 4, 2],
        [2, 1, np.nan, 4, 5, 3, np.nan],
        [4, np.nan, 3, np.nan, 2, 5, 4],
    ],
    dtype=float,
)


def observed_entries(ratings: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    users, items = np.where(~np.isnan(ratings))
    values = ratings[users, items]
    return users, items, values


def initialize_items(item_count: int, rank: int) -> np.ndarray:
    factors = np.zeros((item_count, rank))
    for item in range(item_count):
        for factor in range(rank):
            factors[item, factor] = (
                0.35 * np.sin((item + 1) * (factor + 1) * 1.17)
                + 0.22 * np.cos((item + 2) * (factor + 2) * 0.71)
            )
    return factors


def ridge_solve(design: np.ndarray, values: np.ndarray, rank: int, regularization: float) -> np.ndarray:
    if len(values) == 0:
        return np.zeros(rank)
    lhs = design.T @ design + regularization * np.eye(rank)
    rhs = design.T @ values
    return np.linalg.solve(lhs, rhs)


def als_factorize(
    ratings: np.ndarray,
    rank: int = 2,
    regularization: float = 0.18,
    iterations: int = 30,
) -> tuple[float, np.ndarray, np.ndarray, list[float]]:
    """用 ALS 分解中心化评分矩阵。"""
    user_count, item_count = ratings.shape
    user_ids, item_ids, values = observed_entries(ratings)
    mean_rating = float(values.mean())
    centered = values - mean_rating

    user_factors = np.zeros((user_count, rank))
    item_factors = initialize_items(item_count, rank)
    losses: list[float] = []

    for _ in range(iterations):
        for user in range(user_count):
            mask = user_ids == user
            design = item_factors[item_ids[mask]]
            user_factors[user] = ridge_solve(design, centered[mask], rank, regularization)

        for item in range(item_count):
            mask = item_ids == item
            design = user_factors[user_ids[mask]]
            item_factors[item] = ridge_solve(design, centered[mask], rank, regularization)

        predictions = mean_rating + user_factors @ item_factors.T
        errors = values - predictions[user_ids, item_ids]
        loss = float(errors @ errors + regularization * (np.sum(user_factors**2) + np.sum(item_factors**2)))
        losses.append(loss)

    return mean_rating, user_factors, item_factors, losses


def main() -> None:
    rank = 2
    regularization = 0.18
    iterations = 30
    selected_user = 0

    mean_rating, user_factors, item_factors, losses = als_factorize(
        RATINGS,
        rank=rank,
        regularization=regularization,
        iterations=iterations,
    )
    predictions = np.clip(mean_rating + user_factors @ item_factors.T, 1, 5)
    user_ids, item_ids, values = observed_entries(RATINGS)
    errors = values - predictions[user_ids, item_ids]
    rmse = float(np.sqrt(np.mean(errors**2)))

    missing = np.isnan(RATINGS[selected_user])
    recommendation_order = np.argsort(-predictions[selected_user, missing])
    missing_items = np.where(missing)[0]

    print("Recommender matrix factorization demo")
    print(f"rank={rank}, regularization={regularization}, iterations={iterations}")
    print(f"global mean rating={mean_rating:.3f}")
    print(f"observed RMSE={rmse:.4f}")
    print(f"user factor for {USERS[selected_user]}: {np.round(user_factors[selected_user], 4)}")
    print("recommendations:")
    for order_index in recommendation_order:
        item = missing_items[order_index]
        print(f"  {ITEMS[item]}: predicted rating {predictions[selected_user, item]:.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)

    observed_plot = np.where(np.isnan(RATINGS), predictions, RATINGS)
    im = axes[0].imshow(observed_plot, cmap="YlGnBu", vmin=1, vmax=5)
    axes[0].set_title("observed ratings + predictions")
    axes[0].set_xticks(range(len(ITEMS)), ITEMS, rotation=35, ha="right")
    axes[0].set_yticks(range(len(USERS)), USERS)
    for user in range(RATINGS.shape[0]):
        for item in range(RATINGS.shape[1]):
            label = f"{observed_plot[user, item]:.1f}" if np.isnan(RATINGS[user, item]) else f"{int(RATINGS[user, item])}"
            color = "black" if np.isnan(RATINGS[user, item]) else "white"
            axes[0].text(item, user, label, ha="center", va="center", color=color, fontsize=8)
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    axes[1].scatter(user_factors[:, 0], user_factors[:, 1], color="#2563eb", label="users")
    axes[1].scatter(item_factors[:, 0], item_factors[:, 1], color="#059669", marker="s", label="items")
    for index, name in enumerate(USERS):
        axes[1].text(user_factors[index, 0], user_factors[index, 1], f" {name}", fontsize=8)
    for index, name in enumerate(ITEMS):
        axes[1].text(item_factors[index, 0], item_factors[index, 1], f" {name}", fontsize=8)
    axes[1].axhline(0, color="#94a3b8", linewidth=1)
    axes[1].axvline(0, color="#94a3b8", linewidth=1)
    axes[1].set_title("latent factor space")
    axes[1].set_xlabel("factor 1")
    axes[1].set_ylabel("factor 2")
    axes[1].legend()

    axes[2].plot(losses, color="#2563eb")
    axes[2].set_title("ALS objective")
    axes[2].set_xlabel("iteration")
    axes[2].set_ylabel("regularized loss")
    axes[2].grid(True, alpha=0.25)

    output_path = Path(__file__).with_name("recommender_matrix_factorization_demo.png")
    fig.savefig(output_path, dpi=160)
    print(f"saved figure: {output_path}")


if __name__ == "__main__":
    main()
