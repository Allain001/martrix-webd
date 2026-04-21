"""
二维矩阵变换页。
保留原仓库的交互逻辑，只将讲解口径与界面文案改成中文。
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import streamlit as st
from mpl_zh import configure_matplotlib_fonts


configure_matplotlib_fonts()


def generate_random_cluster(n_points=30, mean=(0.0, 0.0), cov=None, seed=0):
    """生成二维高斯点云。"""
    if cov is None:
        cov = np.array([[1.0, 0.6], [0.6, 1.5]])
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(mean, cov, size=n_points)


def apply_linear_transform(points, matrix):
    """对二维点云施加线性变换，采用行向量约定 y = x A^T。"""
    return points @ matrix.T


def rotation_matrix(theta):
    """二维旋转矩阵。"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def angle_from_rot(rot):
    """从二维旋转矩阵中提取旋转角。"""
    return np.arctan2(rot[1, 0], rot[0, 0])


def make_rotational_svd(matrix):
    """
    将 SVD 调整成更适合几何讲解的旋转-伸缩形式：
    A = U_rot * Sigma_signed * V_rot^T
    """
    u_raw, singular_values, vt_raw = np.linalg.svd(matrix)
    sigma = np.diag(singular_values)
    v_raw = vt_raw.T

    u_rot = u_raw.copy()
    v_rot = v_raw.copy()
    sigma_signed = sigma.copy()
    reflection = np.diag([1.0, -1.0])

    det_u = np.linalg.det(u_rot)
    det_v = np.linalg.det(v_rot)

    if det_u < 0 and det_v < 0:
        u_rot = u_rot @ reflection
        v_rot = v_rot @ reflection
    elif det_u < 0 and det_v >= 0:
        u_rot = u_rot @ reflection
        sigma_signed = reflection @ sigma_signed
    elif det_u >= 0 and det_v < 0:
        v_rot = v_rot @ reflection
        sigma_signed = sigma_signed @ reflection

    return u_rot, sigma_signed, v_rot, singular_values, vt_raw


def svd_path_transform(t, v_rot, sigma_signed, u_rot, theta_v_rad, theta_u_rad):
    """沿着 SVD 路径计算当前时刻的变换矩阵。"""
    s1_signed = sigma_signed[0, 0]
    s2_signed = sigma_signed[1, 1]

    if t <= 1.0:
        alpha = t
        phi = alpha * theta_v_rad
        return rotation_matrix(phi)
    if t <= 2.0:
        alpha = t - 1.0
        rv = rotation_matrix(theta_v_rad)
        s1_t = 1.0 + alpha * (s1_signed - 1.0)
        s2_t = 1.0 + alpha * (s2_signed - 1.0)
        s_t = np.array([[s1_t, 0.0], [0.0, s2_t]])
        return rv @ s_t

    alpha = t - 2.0
    rv = rotation_matrix(theta_v_rad)
    s_final = sigma_signed
    psi = alpha * theta_u_rad
    ru_t = rotation_matrix(-psi)
    return rv @ s_final @ ru_t


def plot_overlay(
    points,
    points_transformed,
    matrix,
    corners,
    corners_transformed,
    show_point_arrows=True,
    show_original=True,
    show_transformed=True,
    draw_eigs=True,
    title_suffix="",
    xlim=None,
    ylim=None,
):
    fig, ax = plt.subplots(figsize=(7, 7))
    draw_frame(
        ax,
        points,
        points_transformed,
        matrix,
        corners,
        corners_transformed,
        show_point_arrows=show_point_arrows,
        show_original=show_original,
        show_transformed=show_transformed,
        draw_eigs=draw_eigs,
        title_suffix=title_suffix,
        xlim=xlim,
        ylim=ylim,
    )
    plt.tight_layout()
    return fig


def draw_frame(
    ax,
    points,
    points_transformed,
    matrix,
    corners,
    corners_transformed,
    show_point_arrows=True,
    show_original=True,
    show_transformed=True,
    draw_eigs=True,
    title_suffix="",
    xlim=None,
    ylim=None,
):
    ax.clear()

    if show_original:
        ax.scatter(points[:, 0], points[:, 1], s=40, alpha=0.9, label="原始点云")
        sq_x = np.append(corners[:, 0], corners[0, 0])
        sq_y = np.append(corners[:, 1], corners[0, 1])
        ax.plot(sq_x, sq_y, "k--", linewidth=1.5, label="原始方框")

    if show_transformed:
        ax.scatter(points_transformed[:, 0], points_transformed[:, 1], s=70, marker="x", alpha=0.9, label="变换后点云")
        sq_tx = np.append(corners_transformed[:, 0], corners_transformed[0, 0])
        sq_ty = np.append(corners_transformed[:, 1], corners_transformed[0, 1])
        ax.plot(sq_tx, sq_ty, "r--", linewidth=1.5, label="变换后方框")

    if show_point_arrows and show_original and show_transformed:
        for (x, y), (xp, yp) in zip(points, points_transformed):
            ax.arrow(
                x, y, xp - x, yp - y,
                head_width=0.08,
                length_includes_head=True,
                linewidth=1.0,
                alpha=0.4,
            )

    ax.axhline(0, color="black", linewidth=1, alpha=0.5)
    ax.axvline(0, color="black", linewidth=1, alpha=0.5)

    if draw_eigs:
        evals, evecs = np.linalg.eig(matrix)
        if np.all(np.isreal(evals)):
            all_x = np.concatenate([points[:, 0], points_transformed[:, 0], corners[:, 0], corners_transformed[:, 0], [0.0]])
            all_y = np.concatenate([points[:, 1], points_transformed[:, 1], corners[:, 1], corners_transformed[:, 1], [0.0]])
            max_rad = np.max(np.sqrt(all_x**2 + all_y**2)) or 1.0
            max_abs_lambda = np.max(np.abs(evals)) or 1.0
            base_scale = 0.7 * max_rad / max_abs_lambda

            for i in range(2):
                lam = float(np.real(evals[i]))
                vec = np.real(evecs[:, i])
                vec = vec / np.linalg.norm(vec)
                length = base_scale * abs(lam)
                ax.arrow(
                    0, 0,
                    length * vec[0], length * vec[1],
                    head_width=0.12,
                    length_includes_head=True,
                    linewidth=2.0,
                    alpha=0.9,
                    label=f"特征向量 {i + 1} (λ={lam:.2f})",
                )

    if xlim is None or ylim is None:
        all_x = np.concatenate([points[:, 0], points_transformed[:, 0], corners[:, 0], corners_transformed[:, 0]])
        all_y = np.concatenate([points[:, 1], points_transformed[:, 1], corners[:, 1], corners_transformed[:, 1]])

        x_min, x_max = all_x.min(), all_x.max()
        y_min, y_max = all_y.min(), all_y.max()
        x_mid = 0.5 * (x_min + x_max)
        y_mid = 0.5 * (y_min + y_max)
        span = max(x_max - x_min, y_max - y_min)
        half = 0.5 * span * 1.1
        xlim = (x_mid - half, x_mid + half)
        ylim = (y_mid - half, y_mid + half)

    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_aspect("equal", "box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    base_title = "二维线性变换：点云、方框与特征方向"
    ax.set_title(f"{base_title} {title_suffix}" if title_suffix else base_title)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper left", fontsize="small")


def create_animation_gif(
    filename,
    points,
    corners,
    matrix,
    u_rot,
    sigma_signed,
    v_rot,
    theta_v_rad,
    theta_u_rad,
    xlim,
    ylim,
    show_arrows=True,
    n_frames=120,
    fps=30,
):
    """生成沿着 SVD 路径变化的 GIF 动画。"""
    fig, ax = plt.subplots(figsize=(7, 7))
    writer = PillowWriter(fps=fps)

    with writer.saving(fig, filename, dpi=150):
        for i in range(n_frames):
            t = 3.0 * i / (n_frames - 1)
            current = svd_path_transform(t, v_rot, sigma_signed, u_rot, theta_v_rad, theta_u_rad)
            pts_m = points @ current
            crn_m = corners @ current

            draw_frame(
                ax,
                points,
                pts_m,
                matrix,
                corners,
                crn_m,
                show_point_arrows=show_arrows,
                show_original=True,
                show_transformed=True,
                draw_eigs=(i == n_frames - 1),
                title_suffix=f"(t={t:.2f})",
                xlim=xlim,
                ylim=ylim,
            )
            writer.grab_frame()

    plt.close(fig)


def main():
    st.set_page_config(page_title="martrixvis | 二维变换", layout="wide")

    st.title("二维线性变换与特征向量")
    st.write(
        """
        这个页面展示 **2×2 矩阵** 如何作用在二维点云和参考方框上，
        并通过一条 **旋转 → 拉伸 → 旋转** 的 SVD 路径，把矩阵动作解释成连续可见的几何过程：

        1. 先经过 $\\tilde V$
        2. 再经过带符号的对角伸缩 $\\tilde\\Sigma$
        3. 最后经过 $\\tilde U^T$

        最终得到的整体效果与矩阵 $A$ 完全一致。
        """
    )

    st.sidebar.header("随机点云")
    seed = st.sidebar.slider("随机种子", 0, 100, 0, 1)

    st.sidebar.markdown("---")
    st.sidebar.header("变换矩阵 A")
    mode = st.sidebar.radio("矩阵输入方式", ["角度 + 缩放", "对称矩阵", "手动输入矩阵"])

    if mode == "角度 + 缩放":
        theta_deg = st.sidebar.slider("旋转角度（度）", -180, 180, 30)
        scale_x = st.sidebar.slider("x 方向缩放", 0.1, 3.0, 2.0, 0.1)
        scale_y = st.sidebar.slider("y 方向缩放", 0.1, 3.0, 0.5, 0.1)
        theta = np.deg2rad(theta_deg)
        rot = rotation_matrix(theta)
        scale = np.array([[scale_x, 0.0], [0.0, scale_y]])
        matrix = rot @ scale
    elif mode == "对称矩阵":
        st.sidebar.write("输入对称矩阵 A 的条目：")
        c1, c2 = st.sidebar.columns(2)
        with c1:
            a11 = st.number_input("a11", value=2.0, step=0.1, key="sym_a11")
            a12 = st.number_input("a12", value=0.5, step=0.1, key="sym_a12")
        with c2:
            st.markdown("a21（与 a12 对称）")
            st.text(f"{a12:.3f}")
            a22 = st.number_input("a22", value=2.0, step=0.1, key="sym_a22")
        matrix = np.array([[a11, a12], [a12, a22]])
    else:
        st.sidebar.write("输入 2×2 矩阵 A 的四个条目：")
        c1, c2 = st.sidebar.columns(2)
        with c1:
            a11 = st.number_input("a11", value=2.0, step=0.1, key="man_a11")
            a21 = st.number_input("a21", value=0.5, step=0.1, key="man_a21")
        with c2:
            a12 = st.number_input("a12", value=1.0, step=0.1, key="man_a12")
            a22 = st.number_input("a22", value=2.0, step=0.1, key="man_a22")
        matrix = np.array([[a11, a12], [a21, a22]])

    show_arrows = st.sidebar.checkbox("显示原始点到变换点的箭头", value=True)

    u_rot, sigma_signed, v_rot, singular_values, _ = make_rotational_svd(matrix)
    theta_u_rad = angle_from_rot(u_rot)
    theta_v_rad = angle_from_rot(v_rot)
    theta_u_deg = np.degrees(theta_u_rad)
    theta_v_deg = np.degrees(theta_v_rad)

    random_points = generate_random_cluster(n_points=30, seed=seed)
    min_x, max_x = random_points[:, 0].min(), random_points[:, 0].max()
    min_y, max_y = random_points[:, 1].min(), random_points[:, 1].max()
    cx = 0.5 * (min_x + max_x)
    cy = 0.5 * (min_y + max_y)
    half_span = 0.5 * max(max_x - min_x, max_y - min_y)
    half_side = half_span * 1.3 + 1e-6

    corners = np.array([
        [cx - half_side, cy - half_side],
        [cx + half_side, cy - half_side],
        [cx + half_side, cy + half_side],
        [cx - half_side, cy + half_side],
    ])

    points = np.vstack([random_points, corners])
    final_transform = v_rot @ sigma_signed @ u_rot.T
    points_a = points @ final_transform
    corners_a = corners @ final_transform

    all_x = np.concatenate([points[:, 0], points_a[:, 0], corners[:, 0], corners_a[:, 0]])
    all_y = np.concatenate([points[:, 1], points_a[:, 1], corners[:, 1], corners_a[:, 1]])
    x_min, x_max = all_x.min(), all_x.max()
    y_min, y_max = all_y.min(), all_y.max()
    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)
    span = max(x_max - x_min, y_max - y_min)
    half = 0.5 * span * 1.1
    xlim = (x_mid - half, x_mid + half)
    ylim = (y_mid - half, y_mid + half)

    st.sidebar.markdown("---")
    t = st.sidebar.slider(
        "SVD 路径（0=单位阵，1=完成 V 旋转，2=完成带符号伸缩，3=完整矩阵 A）",
        min_value=0.0,
        max_value=3.0,
        value=3.0,
        step=0.01,
    )

    current = svd_path_transform(t, v_rot, sigma_signed, u_rot, theta_v_rad, theta_u_rad)
    points_m = points @ current
    corners_m = corners @ current

    eigenvalues, _ = np.linalg.eig(matrix)
    st.subheader("当前矩阵 A")
    st.latex(
        r"""
        A =
        \begin{bmatrix}
        %.3f & %.3f \\
        %.3f & %.3f
        \end{bmatrix}
        """ % (matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1])
    )

    st.subheader("矩阵 A 的特征值")
    if np.all(np.isreal(eigenvalues)):
        st.latex(
            r"""
            \lambda_1 = %.3f,\quad
            \lambda_2 = %.3f
            """ % (np.real(eigenvalues[0]), np.real(eigenvalues[1]))
        )
    else:
        st.write("特征值为复数，因此平面中不会出现对应的实特征向量方向。")

    col1, col2 = st.columns([3, 2])

    if t <= 1.0:
        suffix = "（阶段 1：逐步完成 V 旋转）"
    elif t <= 2.0:
        suffix = "（阶段 2：逐步完成带符号伸缩）"
    else:
        suffix = "（阶段 3：逐步完成 U^T 旋转，逼近完整 A）"

    with col1:
        st.subheader("图形视图（由滑块控制）")
        fig = plot_overlay(
            points,
            points_m,
            matrix,
            corners,
            corners_m,
            show_point_arrows=show_arrows,
            show_original=True,
            show_transformed=True,
            draw_eigs=(t >= 2.99),
            title_suffix=suffix,
            xlim=xlim,
            ylim=ylim,
        )
        st.pyplot(fig, width="stretch")

    with col2:
        st.subheader("旋转 + 伸缩的 SVD 解释")
        st.markdown(
            r"""
我们把矩阵写成更适合几何讲解的形式：

$$
A = \tilde U \,\tilde\Sigma\, \tilde V^T
$$

其中：
- $\tilde U$ 与 $\tilde V$ 是**纯旋转矩阵**（行列式为 1）
- $\tilde\Sigma$ 是可能带符号的对角矩阵，用来表达拉伸以及必要时的翻转
"""
        )

        st.latex(
            r"""
            \Sigma =
            \begin{bmatrix}
            %.3f & 0 \\
            0 & %.3f
            \end{bmatrix}
            """ % (singular_values[0], singular_values[1])
        )

        st.latex(
            r"""
            \tilde U \approx
            \begin{bmatrix}
            %.3f & %.3f \\
            %.3f & %.3f
            \end{bmatrix}
            """ % (u_rot[0, 0], u_rot[0, 1], u_rot[1, 0], u_rot[1, 1])
        )

        v_t_display = v_rot.T
        st.latex(
            r"""
            \tilde V^T \approx
            \begin{bmatrix}
            %.3f & %.3f \\
            %.3f & %.3f
            \end{bmatrix}
            """ % (v_t_display[0, 0], v_t_display[0, 1], v_t_display[1, 0], v_t_display[1, 1])
        )

        st.markdown(
            r"""
对于二维纯旋转矩阵
$$
R(\theta)=
\begin{bmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{bmatrix},
$$
第一列正好就是 $\bigl[\cos\theta,\ \sin\theta\bigr]^T$，
所以我们可以直接从 $\tilde U$ 与 $\tilde V$ 的第一列读取它们对应的近似旋转角。
"""
        )

        st.latex(
            r"""
            \theta_{\tilde U} \approx %.1f^\circ,\quad
            \cos(\theta_{\tilde U}) \approx \tilde U_{11} = %.3f,\quad
            \sin(\theta_{\tilde U}) \approx \tilde U_{21} = %.3f
            """ % (theta_u_deg, u_rot[0, 0], u_rot[1, 0])
        )

        st.latex(
            r"""
            \theta_{\tilde V} \approx %.1f^\circ,\quad
            \cos(\theta_{\tilde V}) \approx \tilde V_{11} = %.3f,\quad
            \sin(\theta_{\tilde V}) \approx \tilde V_{21} = %.3f
            """ % (theta_v_deg, v_rot[0, 0], v_rot[1, 0])
        )

    st.markdown("---")
    st.caption("建议慢慢拖动滑块讲解：单位阵 → V 旋转 → 带符号伸缩 → U^T 旋转 → 完整矩阵动作。")

    st.markdown("## 从 SVD 路径生成 GIF 动画")

    if st.button("生成 GIF 动画（svd2d_animation.gif）"):
        with st.spinner("正在生成 GIF 动画，请稍候..."):
            try:
                create_animation_gif(
                    filename="svd2d_animation.gif",
                    points=points,
                    corners=corners,
                    matrix=matrix,
                    u_rot=u_rot,
                    sigma_signed=sigma_signed,
                    v_rot=v_rot,
                    theta_v_rad=theta_v_rad,
                    theta_u_rad=theta_u_rad,
                    xlim=xlim,
                    ylim=ylim,
                    show_arrows=show_arrows,
                    n_frames=120,
                    fps=30,
                )
                st.success("动画已保存为 svd2d_animation.gif")
            except Exception as exc:
                st.error(f"生成动画失败：{exc}")

    if os.path.exists("svd2d_animation.gif"):
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.image("svd2d_animation.gif")


if __name__ == "__main__":
    main()
