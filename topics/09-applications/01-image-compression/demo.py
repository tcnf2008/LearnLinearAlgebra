"""用 SVD 演示灰度图像的低秩压缩。

运行方式：
    .venv/bin/python topics/09-applications/01-image-compression/demo.py

脚本会生成一个合成灰度图，分别用不同 rank 的截断 SVD 重建，并保存对比图。
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def make_demo_image(size: int = 96) -> np.ndarray:
    """生成一个不依赖外部图片文件的灰度图矩阵。"""
    y, x = np.mgrid[0:size, 0:size] / (size - 1)

    smooth_background = 0.22 + 0.35 * (1 - y) + 0.18 * x
    hill = np.where(y > 0.58 + 0.10 * np.sin(2.4 * np.pi * x), 0.18, 0.0)
    sun = 0.28 * np.exp(-0.5 * (((x - 0.74) / 0.11) ** 2 + ((y - 0.25) / 0.11) ** 2))
    shadow = -0.18 * np.exp(-0.5 * (((x - 0.38) / 0.24) ** 2 + ((y - 0.68) / 0.09) ** 2))
    ridge = 0.12 * np.sin(7 * np.pi * (x + 0.18 * y))

    image = smooth_background + hill + sun + shadow + ridge
    return np.clip(image, 0.0, 1.0)


def truncated_svd(image: np.ndarray, rank: int) -> np.ndarray:
    """保留前 rank 个奇异值并重建图像。"""
    u, singular_values, vt = np.linalg.svd(image, full_matrices=False)
    return (u[:, :rank] * singular_values[:rank]) @ vt[:rank, :]


def storage_count(shape: tuple[int, int], rank: int) -> int:
    rows, cols = shape
    return rank * (rows + cols + 1)


def main() -> None:
    image = make_demo_image()
    u, singular_values, vt = np.linalg.svd(image, full_matrices=False)
    ranks = [2, 6, 12, 24]
    original_storage = image.size
    total_energy = np.sum(singular_values**2)

    fig, axes = plt.subplots(2, len(ranks) + 1, figsize=(14, 7), constrained_layout=True)

    axes[0, 0].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("original")
    axes[0, 0].axis("off")

    axes[1, 0].plot(singular_values, color="#2563eb")
    axes[1, 0].set_title("singular values")
    axes[1, 0].set_xlabel("i")
    axes[1, 0].set_ylabel("sigma")

    print("SVD image compression demo")
    print(f"image shape: {image.shape}, original storage: {original_storage} numbers")

    for col, rank in enumerate(ranks, start=1):
        approx = (u[:, :rank] * singular_values[:rank]) @ vt[:rank, :]
        approx = np.clip(approx, 0.0, 1.0)
        residual = image - approx
        rmse = float(np.sqrt(np.mean(residual**2)))
        kept_energy = float(np.sum(singular_values[:rank] ** 2) / total_energy)
        compressed_storage = storage_count(image.shape, rank)

        axes[0, col].imshow(approx, cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(f"rank {rank}\nenergy {kept_energy:.1%}")
        axes[0, col].axis("off")

        limit = max(float(np.max(np.abs(residual))), 1e-6)
        axes[1, col].imshow(residual, cmap="coolwarm", vmin=-limit, vmax=limit)
        axes[1, col].set_title(f"residual\nRMSE {rmse:.3f}")
        axes[1, col].axis("off")

        print(
            f"rank={rank:>2} | storage={compressed_storage:>5}/{original_storage} "
            f"| energy={kept_energy:>6.1%} | RMSE={rmse:.4f}"
        )

    output_path = Path(__file__).with_name("image_compression_demo.png")
    fig.suptitle("Image compression with truncated SVD", fontsize=15)
    fig.savefig(output_path, dpi=160)
    print(f"saved figure: {output_path}")


if __name__ == "__main__":
    main()
