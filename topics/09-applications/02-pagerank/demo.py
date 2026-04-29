"""用 PageRank 演示有向图上的稳定分布。

运行方式：
    .venv/bin/python topics/09-applications/02-pagerank/demo.py

脚本会构造一个小型网页链接图，比较不同阻尼因子的 PageRank 排名和收敛曲线。
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LABELS = ["A", "B", "C", "D", "E", "F"]
EDGES = [
    (0, 1),
    (0, 2),
    (1, 2),
    (1, 3),
    (2, 0),
    (2, 3),
    (2, 4),
    (3, 4),
    (4, 3),
    (4, 5),
    (5, 2),
]
POSITIONS = np.array(
    [
        [0.10, 0.76],
        [0.42, 0.90],
        [0.82, 0.72],
        [0.76, 0.20],
        [0.38, 0.12],
        [0.08, 0.30],
    ]
)


def transition_matrix(node_count: int, edges: list[tuple[int, int]]) -> np.ndarray:
    """构造列随机转移矩阵，死链列均匀分配给所有节点。"""
    matrix = np.zeros((node_count, node_count), dtype=float)
    out_degree = np.zeros(node_count, dtype=int)

    for source, _target in edges:
        out_degree[source] += 1

    for source in range(node_count):
        if out_degree[source] == 0:
            matrix[:, source] = 1.0 / node_count

    for source, target in edges:
        matrix[target, source] += 1.0 / out_degree[source]

    return matrix


def pagerank(
    matrix: np.ndarray,
    alpha: float = 0.85,
    iterations: int = 80,
    tolerance: float = 1e-12,
) -> tuple[np.ndarray, list[float]]:
    """用幂迭代计算 PageRank。"""
    n = matrix.shape[0]
    google_matrix = alpha * matrix + (1.0 - alpha) / n
    rank = np.full(n, 1.0 / n)
    deltas: list[float] = []

    for _ in range(iterations):
        next_rank = google_matrix @ rank
        next_rank = next_rank / next_rank.sum()
        delta = float(np.abs(next_rank - rank).sum())
        deltas.append(delta)
        rank = next_rank
        if delta < tolerance:
            break

    return rank, deltas


def draw_graph(ax: plt.Axes, ranks: np.ndarray) -> None:
    ax.set_title("link graph, node size = PageRank")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    for source, target in EDGES:
        start = POSITIONS[source]
        end = POSITIONS[target]
        direction = end - start
        start = start + 0.07 * direction
        end = end - 0.10 * direction
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={
                "arrowstyle": "->",
                "color": "#64748b",
                "lw": 1.2 + 8 * ranks[source],
                "alpha": 0.75,
            },
        )

    max_rank = float(ranks.max())
    for index, (x, y) in enumerate(POSITIONS):
        size = 1100 * (0.45 + ranks[index] / max_rank)
        ax.scatter([x], [y], s=size, color="#2563eb", edgecolor="#1e40af", linewidth=1.6, zorder=3)
        ax.text(x, y + 0.012, LABELS[index], color="white", ha="center", va="center", weight="bold", zorder=4)
        ax.text(x, y - 0.055, f"{ranks[index]:.1%}", color="white", ha="center", va="center", fontsize=8, zorder=4)


def main() -> None:
    matrix = transition_matrix(len(LABELS), EDGES)
    alphas = [0.55, 0.85, 0.95]
    results: dict[float, tuple[np.ndarray, list[float]]] = {}

    print("PageRank demo")
    print("edges:", ", ".join(f"{LABELS[s]}->{LABELS[t]}" for s, t in EDGES))

    for alpha in alphas:
        ranks, deltas = pagerank(matrix, alpha=alpha)
        results[alpha] = (ranks, deltas)
        order = np.argsort(-ranks)
        ranking = " > ".join(f"{LABELS[index]}({ranks[index]:.3f})" for index in order)
        print(f"alpha={alpha:.2f} | iterations={len(deltas):>2} | ranking: {ranking}")

    chosen_alpha = 0.85
    chosen_ranks, chosen_deltas = results[chosen_alpha]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)
    draw_graph(axes[0], chosen_ranks)

    order = np.argsort(chosen_ranks)
    axes[1].barh([LABELS[i] for i in order], chosen_ranks[order], color="#2563eb")
    axes[1].set_title("PageRank at alpha=0.85")
    axes[1].set_xlabel("rank probability")

    for alpha, (_ranks, deltas) in results.items():
        axes[2].semilogy(range(1, len(deltas) + 1), deltas, marker="o", label=f"alpha={alpha:.2f}")
    axes[2].set_title("power iteration convergence")
    axes[2].set_xlabel("iteration")
    axes[2].set_ylabel("L1 change")
    axes[2].legend()
    axes[2].grid(True, alpha=0.25)

    output_path = Path(__file__).with_name("pagerank_demo.png")
    fig.savefig(output_path, dpi=160)
    print(f"saved figure: {output_path}")


if __name__ == "__main__":
    main()
