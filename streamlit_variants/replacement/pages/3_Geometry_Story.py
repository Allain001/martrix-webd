from __future__ import annotations

import numpy as np
import streamlit as st


def determinant_story(value: float) -> str:
    if abs(value) < 0.1:
        return "面积几乎被压扁，说明这个变换接近塌缩。"
    if value < 0:
        return "方向发生翻转，所以观众会看到镜像感。"
    return "方向保持不变，主要表现为旋转、缩放或剪切。"


st.set_page_config(page_title="martrixvis | 几何讲解板", layout="wide")
st.title("几何讲解板")
st.write("这一页不强调复杂交互，而是专门服务于讲解：把 det、eigen、rank 讲成可读的几何故事。")

st.sidebar.header("自定义矩阵")
a11 = st.sidebar.number_input("a11", value=1.2, step=0.1)
a12 = st.sidebar.number_input("a12", value=0.4, step=0.1)
a21 = st.sidebar.number_input("a21", value=-0.3, step=0.1)
a22 = st.sidebar.number_input("a22", value=1.1, step=0.1)

matrix = np.array([[a11, a12], [a21, a22]], dtype=float)
determinant = float(np.linalg.det(matrix))
rank = int(np.linalg.matrix_rank(matrix))
eigenvalues, _ = np.linalg.eig(matrix)

card1, card2, card3 = st.columns(3)
card1.metric("det(A)", f"{determinant:.4f}")
card2.metric("rank(A)", str(rank))
card3.metric("面积缩放", f"{determinant:.4f}x")

left, right = st.columns([2, 2])
with left:
    st.subheader("一句话讲法")
    st.info(determinant_story(determinant))
    st.markdown("### 当前矩阵")
    st.latex(
        r"""
        A =
        \begin{bmatrix}
        %.3f & %.3f \\
        %.3f & %.3f
        \end{bmatrix}
        """
        % (matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1])
    )

with right:
    st.subheader("特征值与讲解")
    if np.all(np.isreal(eigenvalues)):
        st.latex(r"\lambda_1 = %.3f,\quad \lambda_2 = %.3f" % (np.real(eigenvalues[0]), np.real(eigenvalues[1])))
        st.write("如果某个方向对应特征向量，那么它在变换后不会偏转，只会伸缩。")
    else:
        st.write("当前矩阵的特征值为复数，因此不会出现实平面里的稳定特征方向。")
    st.markdown(
        """
        ### 给答辩用的三句脚本
        1. 这个矩阵首先决定了面积如何被缩放。  
        2. 如果 det(A) 为负，图形方向会翻转。  
        3. 特征值和特征方向告诉我们，哪些方向在变换后仍保持方向不变。  
        """
    )
