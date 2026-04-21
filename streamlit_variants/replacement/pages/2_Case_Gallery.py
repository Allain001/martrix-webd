from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


CASES = {
    "剪切直觉": {
        "matrix": np.array([[1.0, 0.8], [0.0, 1.0]]),
        "story": "适合解释为什么图形会明显倾斜，但面积并没有被立刻压扁。",
        "focus": "讲 shape change，而不是先讲公式。",
    },
    "旋转与缩放": {
        "matrix": np.array([[1.2, -0.6], [0.6, 1.2]]),
        "story": "适合同时讲旋转、面积缩放和特征值。",
        "focus": "这是最像产品演示页封面效果的一个案例。",
    },
    "镜像翻转": {
        "matrix": np.array([[1.0, 0.0], [0.0, -1.0]]),
        "story": "最快讲清 det(A) 为什么会变成负数。",
        "focus": "重点解释方向翻转。",
    },
    "方程组入口": {
        "matrix": np.array([[2.0, 1.0], [1.0, 3.0]]),
        "story": "把图形直觉连接到 Ax = b 的唯一解。",
        "focus": "适合作为二维模块和方程求解模块之间的桥。",
    },
}


def plot_case(matrix: np.ndarray) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(6.6, 6.6))
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    transformed = square @ matrix.T
    axis.plot(*np.vstack([square, square[0]]).T, "--", linewidth=1.8, label="原始方格")
    axis.plot(*np.vstack([transformed, transformed[0]]).T, "-", linewidth=2.2, label="变换后方格")
    basis = np.eye(2)
    transformed_basis = basis @ matrix.T
    colors = ["#5ba0ff", "#f0a85a"]
    for idx, vector in enumerate(transformed_basis):
        axis.arrow(0, 0, vector[0], vector[1], color=colors[idx], head_width=0.08, linewidth=2.5, length_includes_head=True)
    axis.axhline(0, color="black", linewidth=1.0, alpha=0.35)
    axis.axvline(0, color="black", linewidth=1.0, alpha=0.35)
    axis.grid(True, linestyle="--", alpha=0.25)
    axis.set_aspect("equal", "box")
    axis.legend()
    return figure


st.set_page_config(page_title="MatrixVis | 案例走廊", layout="wide")
st.title("案例走廊")
st.write("保留参考仓库的模块式浏览体验，但把内容换成更适合 MatrixVis 的教学场景。")

selected_case = st.sidebar.selectbox("选择案例", list(CASES.keys()))
case = CASES[selected_case]
determinant = float(np.linalg.det(case["matrix"]))
rank = int(np.linalg.matrix_rank(case["matrix"]))

left, right = st.columns([3, 2])
with left:
    st.subheader(f"{selected_case} | 图形预览")
    st.pyplot(plot_case(case["matrix"]), width="stretch")

with right:
    st.subheader("讲解摘要")
    st.metric("det(A)", f"{determinant:.4f}")
    st.metric("rank(A)", str(rank))
    st.markdown("### 教学故事")
    st.write(case["story"])
    st.markdown("### 演示重点")
    st.write(case["focus"])
    st.markdown("### 当前矩阵")
    st.latex(
        r"""
        A =
        \begin{bmatrix}
        %.3f & %.3f \\
        %.3f & %.3f
        \end{bmatrix}
        """
        % (
            case["matrix"][0, 0],
            case["matrix"][0, 1],
            case["matrix"][1, 0],
            case["matrix"][1, 1],
        )
    )

st.markdown("---")
st.markdown(
    """
    #### 答辩提示
    - 先让评委看图形变化，再讲行列式和秩。  
    - 如果时间紧，优先讲“剪切直觉”和“镜像翻转”两个场景。  
    """
)
