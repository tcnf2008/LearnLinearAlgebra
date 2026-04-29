"""用 PCA 演示三维数据降维和重建。

运行方式：
    .venv/bin/python topics/09-applications/04-pca-dimensionality-reduction/demo.py

脚本会生成一个接近二维平面的三维数据集，计算 PCA 主成分、解释方差和重建误差，并保存可视化图。
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def make_data(sample_count: int = 100, noise: float = 0.18) -> np.ndarray:
    """生成一个主要落在二维平面附近的三维点云。"""
    t = np.linspace(-1.0, 1.0, sample_count)
    u = -2.2 + 4.4 * t
    v = 1.55 * np.sin(2.2 * np.pi * (t + 1) / 2)

    rng = np.random.default_rng(7)
    eps = rng.normal(0.0, noise, size=(sample_count, 3))
    x = 1.02 * u + 0.28 * v + eps[:, 0]
    y = 0.48 * u + 0.92 * v + eps[:, 1]
    z = -0.38 * u + 0.65 * v + eps[:, 2]
    return np.column_stack([x, y, z])


def pca(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回均值、按方差降序排列的特征值和主成分向量。"""
    mean = data.mean(axis=0)
    centered = data - mean
    covariance = centered.T @ centered / (len(data) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    return mean, eigenvalues[order], eigenvectors[:, order]


def project_and_reconstruct(
    data: np.ndarray,
    mean: np.ndarray,
    components: np.ndarray,
    component_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    centered = data - mean
    selected = components[:, :component_count]
    scores = centered @ selected
    reconstructed = scores @ selected.T + mean
    return scores, reconstructed


def main() -> None:
    data = make_data()
    mean, eigenvalues, components = pca(data)
    explained_ratio = eigenvalues / eigenvalues.sum()
    component_counts = [1, 2, 3]

    print("PCA dimensionality reduction demo")
    print(f"data shape: {data.shape}")
    print(f"mean: {np.round(mean, 4)}")
    print(f"eigenvalues: {np.round(eigenvalues, 4)}")
    print(f"explained variance ratio: {np.round(explained_ratio, 4)}")
    print(f"PC1 vector: {np.round(components[:, 0], 4)}")

    reconstructions: dict[int, np.ndarray] = {}
    scores_2d = None
    for count in component_counts:
        scores, reconstructed = project_and_reconstruct(data, mean, components, count)
        rmse = float(np.sqrt(np.mean((data - reconstructed) ** 2)))
        kept = float(explained_ratio[:count].sum())
        reconstructions[count] = reconstructed
        if count == 2:
            scores_2d = scores
        print(f"k={count} | explained={kept:.1%} | reconstruction RMSE={rmse:.4f}")

    fig = plt.figure(figsize=(14, 4.8), constrained_layout=True)
    ax3d = fig.add_subplot(1, 3, 1, projection="3d")
    ax2d = fig.add_subplot(1, 3, 2)
    axbar = fig.add_subplot(1, 3, 3)

    ax3d.scatter(data[:, 0], data[:, 1], data[:, 2], s=28, color="#2563eb", alpha=0.78, label="original")
    ax3d.scatter(
        reconstructions[2][:, 0],
        reconstructions[2][:, 1],
        reconstructions[2][:, 2],
        s=18,
        color="#059669",
        alpha=0.7,
        label="rank-2 reconstruction",
    )
    for index, (vector, color, label) in enumerate(
        zip(components.T, ["#e11d48", "#d97706", "#7c3aed"], ["PC1", "PC2", "PC3"])
    ):
        end = mean + vector * np.sqrt(eigenvalues[index]) * 1.7
        ax3d.plot([mean[0], end[0]], [mean[1], end[1]], [mean[2], end[2]], color=color, linewidth=3, label=label)
    ax3d.set_title("3D data and PCA axes")
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")
    ax3d.legend(loc="upper left", fontsize=8)

    if scores_2d is not None:
        ax2d.scatter(scores_2d[:, 0], scores_2d[:, 1], color="#2563eb", alpha=0.78)
    ax2d.axhline(0, color="#94a3b8", linewidth=1)
    ax2d.axvline(0, color="#94a3b8", linewidth=1)
    ax2d.set_title("2D PCA coordinates")
    ax2d.set_xlabel("PC1 score")
    ax2d.set_ylabel("PC2 score")

    axbar.bar(["PC1", "PC2", "PC3"], explained_ratio, color=["#2563eb", "#2563eb", "#cbd5e1"])
    axbar.plot(["PC1", "PC2", "PC3"], np.cumsum(explained_ratio), color="#059669", marker="o")
    axbar.set_ylim(0, 1.05)
    axbar.set_title("explained variance")
    axbar.set_ylabel("ratio")

    output_path = Path(__file__).with_name("pca_dimensionality_reduction_demo.png")
    fig.savefig(output_path, dpi=160)
    print(f"saved figure: {output_path}")


if __name__ == "__main__":
    main()
