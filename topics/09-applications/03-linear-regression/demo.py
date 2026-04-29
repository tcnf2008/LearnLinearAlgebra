"""用线性代数演示一元线性回归。

运行方式：
    .venv/bin/python topics/09-applications/03-linear-regression/demo.py

脚本会比较正规方程、numpy.linalg.lstsq 和梯度下降得到的参数，并保存残差和损失曲线图。
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def make_data() -> tuple[np.ndarray, np.ndarray]:
    """构造一个带轻微异常点的一元回归数据集。"""
    x = np.linspace(-2.8, 2.8, 15)
    noise = np.array([0.10, -0.12, 0.04, 0.18, -0.10, 0.07, -0.04, 0.02, 0.08, -0.16, 0.06, 0.10, -0.08, 0.04, 0.12])
    y = 0.55 + 0.78 * x + noise
    y[-1] += 1.4
    return x, y


def design_matrix(x: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones_like(x), x])


def sse(x_matrix: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    residual = y - x_matrix @ theta
    return float(residual @ residual)


def normal_equation(x_matrix: np.ndarray, y: np.ndarray) -> np.ndarray:
    """用正规方程求解。solve 比显式求逆更稳。"""
    return np.linalg.solve(x_matrix.T @ x_matrix, x_matrix.T @ y)


def gradient_descent(
    x_matrix: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.035,
    steps: int = 1800,
) -> tuple[np.ndarray, list[float]]:
    theta = np.zeros(x_matrix.shape[1])
    losses: list[float] = []
    n = len(y)

    for _ in range(steps):
        residual = x_matrix @ theta - y
        gradient = (2.0 / n) * x_matrix.T @ residual
        theta -= learning_rate * gradient
        losses.append(sse(x_matrix, y, theta))

    return theta, losses


def main() -> None:
    x, y = make_data()
    x_matrix = design_matrix(x)

    theta_normal = normal_equation(x_matrix, y)
    theta_lstsq, residuals, rank, singular_values = np.linalg.lstsq(x_matrix, y, rcond=None)
    theta_gd, losses = gradient_descent(x_matrix, y)

    y_hat = x_matrix @ theta_lstsq
    residual = y - y_hat
    rmse = float(np.sqrt(np.mean(residual**2)))
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - sse(x_matrix, y, theta_lstsq) / sst
    orthogonality = x_matrix.T @ residual

    print("Linear regression demo")
    print(f"theta by normal equation: intercept={theta_normal[0]:.4f}, slope={theta_normal[1]:.4f}")
    print(f"theta by numpy.linalg.lstsq: intercept={theta_lstsq[0]:.4f}, slope={theta_lstsq[1]:.4f}")
    print(f"theta by gradient descent: intercept={theta_gd[0]:.4f}, slope={theta_gd[1]:.4f}")
    print(f"rank(X)={rank}, singular values={np.round(singular_values, 4)}")
    print(f"RMSE={rmse:.4f}, R^2={r2:.4f}")
    print(f"X^T residual={np.round(orthogonality, 10)}")
    if residuals.size:
        print(f"numpy residual sum of squares={residuals[0]:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True)

    axes[0].scatter(x, y, color="#2563eb", label="data")
    line_x = np.linspace(x.min() - 0.3, x.max() + 0.3, 100)
    line_y = theta_lstsq[0] + theta_lstsq[1] * line_x
    axes[0].plot(line_x, line_y, color="#e11d48", label="least squares")
    for xi, yi, pred in zip(x, y, y_hat):
        axes[0].plot([xi, xi], [yi, pred], color="#e11d48", alpha=0.45)
    axes[0].set_title("least squares fit")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].legend()

    axes[1].bar(range(len(residual)), residual, color=np.where(residual >= 0, "#d97706", "#2563eb"))
    axes[1].axhline(0, color="#172033", linewidth=1)
    axes[1].set_title("residuals")
    axes[1].set_xlabel("sample index")
    axes[1].set_ylabel("y - X theta")

    axes[2].plot(losses, color="#059669")
    axes[2].axhline(sse(x_matrix, y, theta_lstsq), color="#e11d48", linestyle="--", label="least squares SSE")
    axes[2].set_title("gradient descent loss")
    axes[2].set_xlabel("step")
    axes[2].set_ylabel("SSE")
    axes[2].legend()

    output_path = Path(__file__).with_name("linear_regression_demo.png")
    fig.savefig(output_path, dpi=160)
    print(f"saved figure: {output_path}")


if __name__ == "__main__":
    main()
