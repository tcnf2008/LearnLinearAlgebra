# LearnLinearAlgebra

一个用于系统学习 Linear Algebra（线性代数）的自学项目。

本项目不只整理知识点，也为每个核心概念配套直观的 HTML 交互页面，必要时再补充 Python 可视化或数值实验。原则是：**一个知识点优先做成一个完整的 `index.html` 学习页**，把知识讲解、互动呈现、练习题和参考答案放在同一页面里。

## 组织原则

每个知识点都是一个独立学习单元：

```text
topics/
  01-vectors-and-linear-combinations/
    index.html             # 主学习页面：讲解 + 互动 + 练习 + 答案
    demo.py                # 可选：只有需要数值实验或算法实验时才创建
```

这种结构的目标是降低切换成本：读到一个概念时，可以立刻在同一页面操作、观察、练习和对答案。

## Python 虚拟环境

Python demo 是可选项，不是每个知识点都必须创建。只有当 HTML 页面不足以清楚展示数值计算、算法过程或真实应用时，才在知识点目录中添加 `demo.py`。

所有 Python demo 都应在项目根目录的虚拟环境中执行。

首次使用时创建虚拟环境并安装依赖：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt -i https://pypi.org/simple
```

运行某个可选 Python demo：

```bash
.venv/bin/python topics/<topic-folder>/demo.py
```

## 学习路线

### 1. 向量与线性组合

目录：`topics/01-vectors-and-linear-combinations/`

核心问题：

- 什么是向量？
- 向量加法和数乘为什么是线性代数的基本操作？
- 什么是线性组合、span、基和坐标？

建议演示：

- `index.html`：二维向量拖拽，展示向量加法、数乘、线性组合、span、练习和答案。
- 本章暂不需要 Python 脚本，HTML 交互已经足够表达核心直觉。

### 2. 矩阵作为线性变换

目录：`topics/02-matrices-as-linear-transformations/`

核心问题：

- 矩阵为什么可以看作函数？
- 矩阵乘向量的几何意义是什么？
- 矩阵乘法为什么表示变换的复合？

建议演示：

- `index.html`：拖动 2x2 矩阵参数，观察网格如何旋转、缩放、剪切和翻转。
- 可选 `demo.py`：对点云或网格应用矩阵变换，比较不同矩阵的效果。

### 3. 线性方程组与解空间

目录：`topics/03-linear-systems-and-solution-spaces/`

核心问题：

- 线性方程组到底在求什么？
- 无解、唯一解、无穷多解分别对应什么几何图像？
- 高斯消元为什么有效？

建议演示：

- `index.html`：二维直线或三维平面的交点变化，并配套练习。
- 可选 `demo.py`：实现一个小型高斯消元过程，并和 `numpy.linalg.solve` 对比。

### 4. 子空间、秩与零空间

目录：`topics/04-subspaces-rank-and-nullspace/`

核心问题：

- 什么是子空间？
- 列空间、零空间分别描述了矩阵的什么性质？
- 秩和维数之间有什么关系？

建议演示：

- `index.html`：展示矩阵如何把二维平面压成平面、直线或点。
- 可选 `demo.py`：计算 rank、null space，并可视化列空间和零空间。

### 5. 行列式

目录：`topics/05-determinants/`

核心问题：

- 行列式为什么表示面积或体积缩放因子？
- 行列式为 0 为什么表示不可逆？
- 负行列式意味着什么？

建议演示：

- `index.html`：单位正方形经 2x2 矩阵变换后的面积变化。
- 可选 `demo.py`：随机生成矩阵，验证行列式和面积缩放的关系。

### 6. 特征值与特征向量

目录：`topics/06-eigenvalues-and-eigenvectors/`

核心问题：

- 什么方向经过线性变换后仍然不改变方向？
- 特征值表示什么缩放关系？
- 对角化为什么能简化复杂变换？

建议演示：

- `index.html`：在二维变换中拖动向量，高亮特征方向。
- 可选 `demo.py`：实现幂迭代，观察主特征向量如何出现。

### 7. 正交、投影与最小二乘

目录：`topics/07-orthogonality-projection-and-least-squares/`

核心问题：

- 点积如何表达长度、角度和相似度？
- 投影为什么是“最近”的近似？
- 最小二乘为什么能解决超定方程？

建议演示：

- `index.html`：向量投影到直线或平面，并配套最小二乘直觉练习。
- 可选 `demo.py`：用最小二乘做一元或多元线性回归。

### 8. SVD 与 PCA

目录：`topics/08-svd-and-pca/`

核心问题：

- SVD 如何把任意矩阵拆成旋转、缩放、旋转？
- 奇异值表示什么？
- PCA 为什么能降维？

建议演示：

- `index.html`：单位圆经过矩阵变换成为椭圆，展示主轴方向。
- 可选 `demo.py`：图片压缩或二维数据 PCA 降维。

### 9. 应用专题

目录：`topics/09-applications/`

可选主题：

- 图像压缩：`topics/09-applications/01-image-compression/`
- PageRank：`topics/09-applications/02-pagerank/`
- 线性回归：`topics/09-applications/03-linear-regression/`
- PCA 降维：`topics/09-applications/04-pca-dimensionality-reduction/`
- 推荐系统中的矩阵分解：`topics/09-applications/05-recommender-matrix-factorization/`
- 神经网络中的矩阵运算基础

已实现专题：

- `01-image-compression/index.html`：把灰度图看作矩阵，用截断 SVD 展示低秩压缩、残差和存储量取舍。
- `01-image-compression/demo.py`：用 `numpy.linalg.svd` 生成不同 rank 的重建图和误差统计。
- `02-pagerank/index.html`：把网页链接图转成随机转移矩阵，展示阻尼因子、幂迭代和稳定排名向量。
- `02-pagerank/demo.py`：用 `numpy` 计算不同阻尼因子的 PageRank，并画出排序和收敛曲线。
- `03-linear-regression/index.html`：把一元线性回归写成 `Xθ≈y`，展示残差平方、损失地形、正规方程和投影正交条件。
- `03-linear-regression/demo.py`：比较正规方程、`numpy.linalg.lstsq` 和梯度下降的回归参数与误差。
- `04-pca-dimensionality-reduction/index.html`：把三维点云降到一维或二维，展示主成分、解释方差、投影坐标和重建误差。
- `04-pca-dimensionality-reduction/demo.py`：用协方差特征分解计算 PCA，并比较不同保留维度的解释方差和 RMSE。
- `05-recommender-matrix-factorization/index.html`：把稀疏评分矩阵分解成用户因子和物品因子，展示 ALS、正则化、评分补全和推荐排序。
- `05-recommender-matrix-factorization/demo.py`：用 `numpy` 实现小型 ALS 矩阵分解，并输出观测 RMSE、预测矩阵和推荐结果。

## 每个知识点的推荐文件

```text
index.html
```

用于说明：

- 这个知识点要解决什么问题
- 需要掌握的核心概念和公式
- 互动演示
- 概念题、手算题、互动实验题、思考题
- 参考答案要点

```text
demo.py
```

可选文件。只有在需要数值验证、真实数据、算法过程或 HTML 不适合表达时才创建，例如高斯消元、最小二乘、SVD、PCA。

## 第一阶段目标

先完成下面 4 个知识点，每个知识点至少包含一个完整的 `index.html`：

1. 向量与线性组合
2. 矩阵作为线性变换
3. 线性方程组与解空间
4. 特征值与特征向量

完成这 4 个模块后，再扩展行列式、正交投影、SVD 和应用专题。
