from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


def generate_random_cluster(n_points: int = 30, seed: int = 0) -> np.ndarray:
    cov = np.array([[1.0, 0.55], [0.55, 1.45]])
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal((0.0, 0.0), cov, size=n_points)


def rotation_matrix(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def make_rotational_svd(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u_raw, singular_values, vt_raw = np.linalg.svd(matrix)
    sigma = np.diag(singular_values)
    v_raw = vt_raw.T
    reflect = np.diag([1.0, -1.0])

    u_rot = u_raw.copy()
    v_rot = v_raw.copy()
    sigma_signed = sigma.copy()

    if np.linalg.det(u_rot) < 0 and np.linalg.det(v_rot) < 0:
        u_rot = u_rot @ reflect
        v_rot = v_rot @ reflect
    elif np.linalg.det(u_rot) < 0:
        u_rot = u_rot @ reflect
        sigma_signed = reflect @ sigma_signed
    elif np.linalg.det(v_rot) < 0:
        v_rot = v_rot @ reflect
        sigma_signed = sigma_signed @ reflect

    return u_rot, sigma_signed, v_rot


def svd_path_transform(t: float, v_rot: np.ndarray, sigma_signed: np.ndarray, u_rot: np.ndarray) -> np.ndarray:
    theta_v = float(np.arctan2(v_rot[1, 0], v_rot[0, 0]))
    theta_u = float(np.arctan2(u_rot[1, 0], u_rot[0, 0]))

    if t <= 1.0:
        return rotation_matrix(theta_v * t)
    if t <= 2.0:
        alpha = t - 1.0
        sigma_t = np.diag(
            [
                1.0 + alpha * (sigma_signed[0, 0] - 1.0),
                1.0 + alpha * (sigma_signed[1, 1] - 1.0),
            ]
        )
        return rotation_matrix(theta_v) @ sigma_t
    return rotation_matrix(theta_v) @ sigma_signed @ rotation_matrix(-theta_u * (t - 2.0))


def build_square(points: np.ndarray) -> np.ndarray:
    min_x, max_x = points[:, 0].min(), points[:, 0].max()
    min_y, max_y = points[:, 1].min(), points[:, 1].max()
    center_x = 0.5 * (min_x + max_x)
    center_y = 0.5 * (min_y + max_y)
    half_side = 0.65 * max(max_x - min_x, max_y - min_y) + 1e-6
    return np.array(
        [
            [center_x - half_side, center_y - half_side],
            [center_x + half_side, center_y - half_side],
            [center_x + half_side, center_y + half_side],
            [center_x - half_side, center_y + half_side],
        ]
    )


def draw_scene(points: np.ndarray, transformed: np.ndarray, square: np.ndarray, square_transformed: np.ndarray, matrix: np.ndarray, show_arrows: bool, title: str) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(7, 7))
    axis.scatter(points[:, 0], points[:, 1], s=38, alpha=0.88, label="原始点云")
    axis.scatter(transformed[:, 0], transformed[:, 1], s=56, marker="x", alpha=0.92, label="变换后点云")

    axis.plot(*np.vstack([square, square[0]]).T, "--", linewidth=1.6, label="原始边界")
    axis.plot(*np.vstack([square_transformed, square_transformed[0]]).T, "--", linewidth=1.9, label="变换后边界")

    if show_arrows:
        for source, target in zip(points, transformed):
            axis.arrow(
                source[0],
                source[1],
                target[0] - source[0],
                target[1] - source[1],
                linewidth=0.9,
                alpha=0.25,
                head_width=0.07,
                length_includes_head=True,
            )

    eigvals, eigvecs = np.linalg.eig(matrix)
    if np.all(np.isreal(eigvals)):
        for index in range(2):
            vec = np.real(eigvecs[:, index])
            vec = vec / np.linalg.norm(vec)
            axis.arrow(
                0,
                0,
                vec[0] * 2.0 * abs(float(np.real(eigvals[index]))),
                vec[1] * 2.0 * abs(float(np.real(eigvals[index]))),
                head_width=0.12,
                linewidth=2.1,
                alpha=0.9,
                label=f"特征方向 {index + 1}",
            )

    axis.axhline(0, color="black", linewidth=1.0, alpha=0.4)
    axis.axvline(0, color="black", linewidth=1.0, alpha=0.4)
    axis.grid(True, linestyle="--", alpha=0.26)
    axis.set_aspect("equal", "box")
    axis.set_title(title)
    axis.legend(loc="upper left", fontsize="small")
    return figure


st.set_page_config(page_title="martrixvis | 二维矩阵实验室", layout="wide")
st.title("二维矩阵实验室")
st.write("用图形方式解释 2×2 矩阵如何对整个平面施加动作。")

st.sidebar.header("随机点云")
seed = st.sidebar.slider("随机种子", 0, 100, 0, 1)
st.sidebar.markdown("---")
st.sidebar.header("矩阵 A")
mode = st.sidebar.radio("定义方式", ["旋转 + 缩放", "对称矩阵", "手动输入"])

if mode == "旋转 + 缩放":
    theta_deg = st.sidebar.slider("旋转角度（度）", -180, 180, 28)
    scale_x = st.sidebar.slider("x 方向缩放", 0.1, 3.0, 1.8, 0.1)
    scale_y = st.sidebar.slider("y 方向缩放", 0.1, 3.0, 0.7, 0.1)
    matrix = rotation_matrix(np.deg2rad(theta_deg)) @ np.array([[scale_x, 0.0], [0.0, scale_y]])
elif mode == "对称矩阵":
    col1, col2 = st.sidebar.columns(2)
    with col1:
        a11 = st.number_input("a11", value=1.8, step=0.1)
        a12 = st.number_input("a12", value=0.6, step=0.1)
    with col2:
        st.markdown("a21 = a12")
        st.text(f"{a12:.3f}")
        a22 = st.number_input("a22", value=1.4, step=0.1)
    matrix = np.array([[a11, a12], [a12, a22]])
else:
    col1, col2 = st.sidebar.columns(2)
    with col1:
        a11 = st.number_input("a11", value=1.2, step=0.1, key="m11")
        a21 = st.number_input("a21", value=-0.3, step=0.1, key="m21")
    with col2:
        a12 = st.number_input("a12", value=0.4, step=0.1, key="m12")
        a22 = st.number_input("a22", value=1.1, step=0.1, key="m22")
    matrix = np.array([[a11, a12], [a21, a22]])

show_arrows = st.sidebar.checkbox("显示映射箭头", value=True)
t = st.sidebar.slider("分解路径", 0.0, 3.0, 3.0, 0.01)

points = generate_random_cluster(seed=seed)
square = build_square(points)
u_rot, sigma_signed, v_rot = make_rotational_svd(matrix)
transform = svd_path_transform(t, v_rot, sigma_signed, u_rot)

transformed_points = points @ transform.T
transformed_square = square @ transform.T
determinant = float(np.linalg.det(matrix))
eigenvalues, _ = np.linalg.eig(matrix)

phase_label = "阶段 1：旋转" if t <= 1.0 else "阶段 2：拉伸" if t <= 2.0 else "阶段 3：组合完成"

plot_col, info_col = st.columns([3, 2])
with plot_col:
    st.subheader("图形舞台")
    figure = draw_scene(points, transformed_points, square, transformed_square, matrix, show_arrows, f"martrixvis 变换视图 | {phase_label}")
    st.pyplot(figure, width="stretch")

with info_col:
    st.subheader("讲解面板")
    st.metric("det(A)", f"{determinant:.4f}")
    st.metric("面积缩放", f"{determinant:.4f}x")
    st.metric("当前阶段", phase_label)
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
    st.markdown("### 特征值")
    if np.all(np.isreal(eigenvalues)):
        st.latex(r"\lambda_1 = %.3f,\quad \lambda_2 = %.3f" % (np.real(eigenvalues[0]), np.real(eigenvalues[1])))
    else:
        st.write("当前特征值为复数，因此不会出现平面内的实特征方向。")
    st.markdown(
        """
        ### 讲法建议
        - 先看点云和边界如何整体变形。
        - 再解释 `det(A)` 是面积缩放和方向翻转的核心指标。
        - 最后用特征方向说明“哪些方向只伸缩、不偏转”。
        """
    )
