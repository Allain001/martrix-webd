from __future__ import annotations

import os

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

    det_u = np.linalg.det(u_rot)
    det_v = np.linalg.det(v_rot)

    if det_u < 0 and det_v < 0:
        u_rot = u_rot @ reflect
        v_rot = v_rot @ reflect
    elif det_u < 0:
        u_rot = u_rot @ reflect
        sigma_signed = reflect @ sigma_signed
    elif det_v < 0:
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
        start = rotation_matrix(theta_v)
        sigma_t = np.diag(
            [
                1.0 + alpha * (sigma_signed[0, 0] - 1.0),
                1.0 + alpha * (sigma_signed[1, 1] - 1.0),
            ]
        )
        return start @ sigma_t

    alpha = t - 2.0
    return rotation_matrix(theta_v) @ sigma_signed @ rotation_matrix(-theta_u * alpha)


def build_square(points: np.ndarray) -> np.ndarray:
    min_x, max_x = points[:, 0].min(), points[:, 0].max()
    min_y, max_y = points[:, 1].min(), points[:, 1].max()
    center_x = 0.5 * (min_x + max_x)
    center_y = 0.5 * (min_y + max_y)
    half_span = 0.5 * max(max_x - min_x, max_y - min_y)
    half_side = half_span * 1.25 + 1e-6
    return np.array(
        [
            [center_x - half_side, center_y - half_side],
            [center_x + half_side, center_y - half_side],
            [center_x + half_side, center_y + half_side],
            [center_x - half_side, center_y + half_side],
        ]
    )


def draw_scene(points: np.ndarray, transformed: np.ndarray, square: np.ndarray, square_transformed: np.ndarray, matrix: np.ndarray, show_arrows: bool, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(points[:, 0], points[:, 1], s=38, alpha=0.88, label="原始点云")
    ax.scatter(transformed[:, 0], transformed[:, 1], s=56, marker="x", alpha=0.92, label="变换后点云")

    square_closed = np.vstack([square, square[0]])
    square_transformed_closed = np.vstack([square_transformed, square_transformed[0]])
    ax.plot(square_closed[:, 0], square_closed[:, 1], "--", linewidth=1.6, label="原始边界")
    ax.plot(square_transformed_closed[:, 0], square_transformed_closed[:, 1], "--", linewidth=1.9, label="变换后边界")

    if show_arrows:
        for source, target in zip(points, transformed):
            ax.arrow(
                source[0],
                source[1],
                target[0] - source[0],
                target[1] - source[1],
                linewidth=0.9,
                alpha=0.28,
                head_width=0.07,
                length_includes_head=True,
            )

    eigvals, eigvecs = np.linalg.eig(matrix)
    if np.all(np.isreal(eigvals)):
        for index in range(2):
            vec = np.real(eigvecs[:, index])
            vec = vec / np.linalg.norm(vec)
            length = 2.0 * abs(float(np.real(eigvals[index])))
            ax.arrow(
                0,
                0,
                vec[0] * length,
                vec[1] * length,
                head_width=0.12,
                linewidth=2.2,
                alpha=0.92,
                label=f"特征方向 {index + 1}",
            )

    ax.axhline(0, color="black", linewidth=1.0, alpha=0.45)
    ax.axvline(0, color="black", linewidth=1.0, alpha=0.45)
    ax.grid(True, linestyle="--", alpha=0.26)
    ax.set_aspect("equal", "box")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize="small")
    return fig


def save_gif_hint() -> None:
    if os.path.exists("svd2d_animation.gif"):
        st.success("检测到现有动画文件：svd2d_animation.gif")
        st.image("svd2d_animation.gif")
    else:
        st.info("如果你想做答辩素材，可以继续保留原仓库里的 GIF 导出流程。")


def main() -> None:
    st.set_page_config(page_title="MatrixVis | 二维矩阵变换实验室", layout="wide")

    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.3rem; padding-bottom: 2rem;}
        .hero {
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 24px;
            padding: 24px 28px;
            background: linear-gradient(135deg, rgba(18,23,32,0.96), rgba(10,13,18,0.94));
            box-shadow: 0 18px 60px rgba(0,0,0,0.28);
            margin-bottom: 1rem;
        }
        .hero h1 {color: #f5efe4; margin-bottom: 0.6rem;}
        .hero p {color: #bab2a6; line-height: 1.75; max-width: 860px;}
        </style>
        <div class="hero">
            <h1>二维矩阵变换实验室</h1>
            <p>
                用这个页面讲清楚矩阵不是一堆数字，而是对整个平面的动作。
                你可以修改 A，拖动分解路径滑块，观察点云、边界框和特征方向如何一起变化。
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.header("随机点云")
    seed = st.sidebar.slider("随机种子", 0, 100, 0, 1)

    st.sidebar.markdown("---")
    st.sidebar.header("矩阵 A")
    mode = st.sidebar.radio("定义方式", ["旋转 + 缩放", "对称矩阵", "手动输入"])

    if mode == "旋转 + 缩放":
        theta_deg = st.sidebar.slider("旋转角度（度）", -180, 180, 28)
        scale_x = st.sidebar.slider("x 方向缩放", 0.1, 3.0, 1.8, 0.1)
        scale_y = st.sidebar.slider("y 方向缩放", 0.1, 3.0, 0.7, 0.1)
        theta = np.deg2rad(theta_deg)
        matrix = rotation_matrix(theta) @ np.array([[scale_x, 0.0], [0.0, scale_y]])
    elif mode == "对称矩阵":
        c1, c2 = st.sidebar.columns(2)
        with c1:
            a11 = st.number_input("a11", value=1.8, step=0.1)
            a12 = st.number_input("a12", value=0.6, step=0.1)
        with c2:
            st.markdown("a21 = a12")
            st.text(f"{a12:.3f}")
            a22 = st.number_input("a22", value=1.4, step=0.1)
        matrix = np.array([[a11, a12], [a12, a22]])
    else:
        c1, c2 = st.sidebar.columns(2)
        with c1:
            a11 = st.number_input("a11", value=1.2, step=0.1, key="m11")
            a21 = st.number_input("a21", value=-0.3, step=0.1, key="m21")
        with c2:
            a12 = st.number_input("a12", value=0.4, step=0.1, key="m12")
            a22 = st.number_input("a22", value=1.1, step=0.1, key="m22")
        matrix = np.array([[a11, a12], [a21, a22]])

    show_arrows = st.sidebar.checkbox("显示原始点到目标点的箭头", value=True)
    t = st.sidebar.slider("分解路径滑块", 0.0, 3.0, 3.0, 0.01)

    points = generate_random_cluster(seed=seed)
    square = build_square(points)
    u_rot, sigma_signed, v_rot = make_rotational_svd(matrix)
    transform = svd_path_transform(t, v_rot, sigma_signed, u_rot)

    transformed_points = points @ transform.T
    transformed_square = square @ transform.T
    determinant = float(np.linalg.det(matrix))
    eigvals, _ = np.linalg.eig(matrix)

    if t <= 1.0:
        stage_label = "阶段 1：旋转展开"
    elif t <= 2.0:
        stage_label = "阶段 2：拉伸展开"
    else:
        stage_label = "阶段 3：逼近完整变换"

    col_plot, col_info = st.columns([3, 2])

    with col_plot:
        st.subheader("图形舞台")
        figure = draw_scene(
            points,
            transformed_points,
            square,
            transformed_square,
            matrix,
            show_arrows,
            title=f"MatrixVis 变换视图 | {stage_label}",
        )
        st.pyplot(figure, width="stretch")

    with col_info:
        st.subheader("讲解摘要")
        st.metric("det(A)", f"{determinant:.4f}")
        st.metric("面积缩放", f"{determinant:.4f}x")
        st.metric("当前阶段", stage_label)
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
        if np.all(np.isreal(eigvals)):
            st.latex(
                r"\lambda_1 = %.3f,\quad \lambda_2 = %.3f"
                % (np.real(eigvals[0]), np.real(eigvals[1]))
            )
        else:
            st.write("当前特征值为复数，说明实特征方向不会直接显示在平面内。")
        st.markdown(
            """
            ### 讲法建议
            - 先看边界框如何旋转、拉伸或翻折。
            - 再解释 det(A) 决定了面积缩放与方向是否翻转。
            - 最后把特征值与特征方向作为“哪些方向保持不偏转”的补充解释。
            """
        )

    st.markdown("---")
    st.subheader("答辩素材")
    save_gif_hint()


if __name__ == "__main__":
    main()
