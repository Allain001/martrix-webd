# pages/5_PCA_Demo.py
# Streamlit PCA demo page: 2D demo, 3D demo, and PCA<->SVD bridge
#
# Dependencies:
#   streamlit
#   numpy
#   plotly
#
# Put this file into your Streamlit app's /pages folder.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st


# -----------------------------
# Core PCA via SVD (no sklearn)
# -----------------------------
@dataclass
class PCAResult:
    mean: np.ndarray          # (d,)
    U: np.ndarray             # (n, r)
    s: np.ndarray             # (r,)
    Vt: np.ndarray            # (r, d)
    Xc: np.ndarray            # (n, d) centered
    scores: np.ndarray        # (n, r) = Xc @ V


def pca_via_svd(X: np.ndarray) -> PCAResult:
    """PCA on rows-as-samples data matrix X (n_samples, n_features)."""
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0)
    Xc = X - mean

    # economy SVD: Xc = U diag(s) Vt
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    V = Vt.T
    scores = Xc @ V
    return PCAResult(mean=mean, U=U, s=s, Vt=Vt, Xc=Xc, scores=scores)


def reconstruct_from_components(p: PCAResult, k: int) -> np.ndarray:
    """Reconstruct X using first k principal components."""
    k = int(np.clip(k, 0, p.s.shape[0]))
    if k == 0:
        return np.tile(p.mean, (p.Xc.shape[0], 1))
    V = p.Vt.T
    Zk = p.scores[:, :k]
    Xhat = Zk @ V[:, :k].T + p.mean
    return Xhat


def reconstruct_from_selected_singulars(p: PCAResult, selected: List[int]) -> np.ndarray:
    """Reconstruct X using selected singular components (bridge tab)."""
    if not selected:
        return np.tile(p.mean, (p.Xc.shape[0], 1))
    sel = np.array(sorted(set(int(i) for i in selected)), dtype=int)
    sel = sel[(sel >= 0) & (sel < p.s.shape[0])]
    if sel.size == 0:
        return np.tile(p.mean, (p.Xc.shape[0], 1))
    # Xc_hat = U[:,sel] diag(s[sel]) Vt[sel,:]
    Xc_hat = (p.U[:, sel] * p.s[sel]) @ p.Vt[sel, :]
    return Xc_hat + p.mean


def explained_variance_ratio(p: PCAResult) -> np.ndarray:
    """Explained variance ratio using singular values: var_i ∝ s_i^2/(n-1)."""
    n = p.Xc.shape[0]
    if n <= 1:
        return np.zeros_like(p.s)
    vals = (p.s ** 2) / (n - 1)
    total = vals.sum() if vals.sum() > 0 else 1.0
    return vals / total


def mse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.mean((a - b) ** 2))


# -----------------------------
# Data generators (2D / 3D)
# -----------------------------
def gen_2d_ellipse(
    n: int,
    angle_deg: float,
    scale_major: float,
    scale_minor: float,
    noise: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # base gaussian
    X = rng.normal(size=(n, 2))
    X[:, 0] *= scale_major
    X[:, 1] *= scale_minor

    # rotate
    th = math.radians(angle_deg)
    R = np.array([[math.cos(th), -math.sin(th)],
                  [math.sin(th),  math.cos(th)]], dtype=float)
    X = X @ R.T

    # add noise
    X += rng.normal(scale=noise, size=X.shape)
    return X


def add_outliers(X: np.ndarray, n_outliers: int, spread: float, seed: int) -> np.ndarray:
    if n_outliers <= 0:
        return X
    rng = np.random.default_rng(seed + 999)
    d = X.shape[1]
    out = rng.normal(size=(n_outliers, d)) * spread
    return np.vstack([X, out])


def gen_3d_pancake(
    n: int,
    tilt_deg_x: float,
    tilt_deg_y: float,
    scale1: float,
    scale2: float,
    thickness: float,
    noise: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    # generate points mostly on a plane (z small)
    X = rng.normal(size=(n, 3))
    X[:, 0] *= scale1
    X[:, 1] *= scale2
    X[:, 2] *= thickness

    # tilt plane by rotations about x and y
    ax = math.radians(tilt_deg_x)
    ay = math.radians(tilt_deg_y)

    Rx = np.array([[1, 0, 0],
                   [0, math.cos(ax), -math.sin(ax)],
                   [0, math.sin(ax),  math.cos(ax)]], dtype=float)
    Ry = np.array([[ math.cos(ay), 0, math.sin(ay)],
                   [0, 1, 0],
                   [-math.sin(ay), 0, math.cos(ay)]], dtype=float)
    R = Ry @ Rx
    X = X @ R.T

    X += rng.normal(scale=noise, size=X.shape)
    return X


# -----------------------------
# Plot helpers
# -----------------------------
def fig_2d_pca_scene(
    X: np.ndarray,
    Xhat: np.ndarray,
    p: PCAResult,
    k: int,
    show_shadows: bool,
) -> go.Figure:
    mean = p.mean
    V = p.Vt.T

    fig = go.Figure()

    # original points
    fig.add_trace(go.Scatter(
        x=X[:, 0], y=X[:, 1],
        mode="markers",
        name="Original",
        marker=dict(size=6, opacity=0.8),
    ))

    # reconstructed points (only meaningful if k < 2)
    if k < X.shape[1]:
        fig.add_trace(go.Scatter(
            x=Xhat[:, 0], y=Xhat[:, 1],
            mode="markers",
            name=f"Reconstruction (k={k})",
            marker=dict(
                size=9,
                symbol="x",
                color="orange",
                line=dict(width=2),
                opacity=0.9,
                ),
            ))  

    # mean point
    fig.add_trace(go.Scatter(
        x=[mean[0]], y=[mean[1]],
        mode="markers+text",
        name="Mean",
        text=["mean"],
        textposition="top center",
        marker=dict(size=10),
    ))

    # PC axes arrows (scaled by singular values for visual length)
    # make them visible even with small s
    s = p.s
    scale = (np.max(np.linalg.norm(X - mean, axis=1)) + 1e-9) * 0.9
    for i in range(2):
        vec = V[:, i]
        length = (s[i] / (s[0] + 1e-9)) * scale
        end = mean + vec * length
        fig.add_trace(go.Scatter(
            x=[mean[0], end[0]],
            y=[mean[1], end[1]],
            mode="lines",
            name=f"PC{i+1}",
            line=dict(width=4),
        ))

    # shadows: connect each point to its projection (k=1)
    if show_shadows and k == 1:
        # projection onto PC1 line: Xhat is the projected reconstruction
        for i in range(min(X.shape[0], 250)):  # cap to keep plot fast
            fig.add_trace(go.Scatter(
                x=[X[i, 0], Xhat[i, 0]],
                y=[X[i, 1], Xhat[i, 1]],
                mode="lines",
                showlegend=False,
                line=dict(width=1),
                opacity=0.35,
            ))

    fig.update_layout(
        height=650,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="x",
        yaxis_title="y",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)  # equal aspect
    return fig


def fig_3d_pca_scene(
    X: np.ndarray,
    Xhat: np.ndarray,
    p: PCAResult,
    k: int,
    show_plane: bool,
    show_vectors: bool,
) -> go.Figure:
    mean = p.mean
    V = p.Vt.T
    s = p.s

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=X[:, 0], y=X[:, 1], z=X[:, 2],
        mode="markers",
        name="Original",
        marker=dict(size=3, opacity=0.75),
    ))

    if k < X.shape[1]:
        fig.add_trace(go.Scatter3d(
            x=Xhat[:, 0], y=Xhat[:, 1], z=Xhat[:, 2],
            mode="markers",
            name=f"Reconstruction (k={k})",
            marker=dict(
                size=5,
                symbol="x",
                color="orange",
                line=dict(width=3),
                opacity=0.9,
                ),
        ))

    # PC vectors
    if show_vectors:
        scale = (np.max(np.linalg.norm(X - mean, axis=1)) + 1e-9) * 0.9
        for i in range(3):
            vec = V[:, i]
            length = (s[i] / (s[0] + 1e-9)) * scale
            end = mean + vec * length
            fig.add_trace(go.Scatter3d(
                x=[mean[0], end[0]],
                y=[mean[1], end[1]],
                z=[mean[2], end[2]],
                mode="lines",
                name=f"PC{i+1}",
                line=dict(width=6),
            ))

    # PCA plane spanned by PC1 and PC2
    if show_plane and p.s.shape[0] >= 2:
        v1 = V[:, 0]
        v2 = V[:, 1]
        # grid in plane coordinates
        grid = np.linspace(-1, 1, 15)
        a, b = np.meshgrid(grid, grid)
        # scale plane size
        plane_scale = np.max(np.linalg.norm(X - mean, axis=1)) * 0.9
        P = mean.reshape(1, 1, 3) + (a[..., None] * v1 + b[..., None] * v2) * plane_scale
        fig.add_trace(go.Surface(
            x=P[:, :, 0], y=P[:, :, 1], z=P[:, :, 2],
            name="PC1-PC2 plane",
            showscale=False,
            opacity=0.25,
        ))

    fig.update_layout(
        height=700,
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def fig_sigma_and_evr(p: PCAResult) -> Tuple[go.Figure, go.Figure]:
    sig = p.s
    evr = explained_variance_ratio(p)
    xs = list(range(1, len(sig) + 1))

    f1 = go.Figure()
    f1.add_trace(go.Bar(x=xs, y=sig, name="σ"))
    f1.update_layout(
        height=320, margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="component index",
        yaxis_title="singular value σ",
    )

    f2 = go.Figure()
    f2.add_trace(go.Bar(x=xs, y=evr, name="explained variance ratio"))
    f2.update_layout(
        height=320, margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="component index",
        yaxis_title="explained variance ratio",
    )

    return f1, f2


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="PCA 演示（二维 / 三维 / SVD 桥接）", layout="wide")

st.title("PCA 演示")
st.caption("把 PCA 看成“旋转 → 投影 → 重建”的过程，并用中心化数据上的 SVD 来实现。")

tabs = st.tabs(["二维：旋转与投影", "三维：平面 / 直线投影", "桥接：PCA = 中心化数据的 SVD"])

# -----------------------------
# Tab 1: 2D
# -----------------------------
with tabs[0]:
    left, right = st.columns([1, 2], gap="large")

    with left:
        st.subheader("二维数据")
        n = st.slider("点的数量", 50, 800, 250, 10)
        angle = st.slider("主轴倾斜角（度）", 0, 89, 45, 1)
        scale_major = st.slider("主方向尺度", 0.5, 6.0, 3.0, 0.1)
        scale_minor = st.slider("次方向尺度", 0.1, 3.0, 0.8, 0.1)
        noise = st.slider("噪声强度", 0.0, 1.5, 0.50, 0.1)
        seed = st.number_input("随机种子", value=7, step=1)

        st.divider()
        add_out = st.checkbox("加入离群点", value=False)
        out_n = st.slider("离群点数量", 0, 80, 12, 1, disabled=not add_out)
        out_spread = st.slider("离群点分布范围", 1.0, 20.0, 10.0, 0.5, disabled=not add_out)

        st.divider()
        k = st.slider("保留 k 个主成分", 0, 2, 1, 1)
        show_shadows = st.checkbox("显示投影阴影（k=1）", value=True)

    X = gen_2d_ellipse(
        n=n,
        angle_deg=float(angle),
        scale_major=float(scale_major),
        scale_minor=float(scale_minor),
        noise=float(noise),
        seed=int(seed),
    )
    if add_out:
        X = add_outliers(X, n_outliers=int(out_n), spread=float(out_spread), seed=int(seed))

    p = pca_via_svd(X)
    Xhat = reconstruct_from_components(p, k=k)
    err = mse(X, Xhat)

    with right:
        st.subheader("可视化结果")
        st.plotly_chart(fig_2d_pca_scene(X=X, Xhat=Xhat, p=p, k=k, show_shadows=show_shadows), use_container_width=True)
        evr = explained_variance_ratio(p)
        st.write(f"**重建均方误差：** {err:.6f}")
        st.write(f"**解释方差比：** 主成分 1 为 {evr[0]:.3f}，主成分 2 为 {evr[1]:.3f}")

        with st.expander("讲解提示"):
            st.markdown(
                """
- 点云被拉伸成斜椭圆，说明两个维度存在相关性。
- PCA 会找到一组新的正交坐标轴，也就是主成分方向。
- 只保留 PC1，就等于把所有点投影到最佳拟合直线上。
- 丢掉 PC2 会增大重建误差，但往往也能滤掉部分噪声。
                """.strip()
            )

# -----------------------------
# Tab 2: 3D
# -----------------------------
with tabs[1]:
    left, right = st.columns([1, 2], gap="large")

    with left:
        st.subheader("三维数据")
        n3 = st.slider("点的数量", 80, 2000, 200, 20)
        tilt_x = st.slider("绕 x 轴倾斜（度）", 0, 89, 25, 1)
        tilt_y = st.slider("绕 y 轴倾斜（度）", 0, 89, 35, 1)
        scale1 = st.slider("方向 1 的扩展尺度", 0.5, 8.0, 4.0, 0.1)
        scale2 = st.slider("方向 2 的扩展尺度", 0.5, 8.0, 2.5, 0.1)
        thickness = st.slider("薄厚方向尺度", 0.01, 2.0, 1.5, 0.01)
        noise3 = st.slider("噪声强度", 0.0, 1.5, 0.85, 0.05)
        seed3 = st.number_input("随机种子", value=11, step=1, key="seed3")

        st.divider()
        k3 = st.slider("保留 k 个主成分", 0, 3, 2, 1)
        show_plane = st.checkbox("显示 PC1-PC2 平面", value=True)
        show_vectors = st.checkbox("显示主成分方向", value=True)

    X3 = gen_3d_pancake(
        n=int(n3),
        tilt_deg_x=float(tilt_x),
        tilt_deg_y=float(tilt_y),
        scale1=float(scale1),
        scale2=float(scale2),
        thickness=float(thickness),
        noise=float(noise3),
        seed=int(seed3),
    )
    p3 = pca_via_svd(X3)
    X3hat = reconstruct_from_components(p3, k=k3)
    err3 = mse(X3, X3hat)
    evr3 = explained_variance_ratio(p3)

    with right:
        st.subheader("可视化结果")
        st.plotly_chart(
            fig_3d_pca_scene(X=X3, Xhat=X3hat, p=p3, k=k3, show_plane=show_plane, show_vectors=show_vectors),
            use_container_width=True,
        )
        st.write(f"**重建均方误差：** {err3:.6f}")
        st.write(
            f"**解释方差比：** 主成分 1 为 {evr3[0]:.3f}，主成分 2 为 {evr3[1]:.3f}，主成分 3 为 {evr3[2]:.3f}"
        )

        with st.expander("讲解提示"):
            st.markdown(
                """
- 当 k=2 时，相当于把点云投影到最佳拟合平面，同时削弱最薄的噪声方向。
- 当 k=1 时，所有点会进一步压缩到一条最佳拟合直线。
- 当 k=3 时，表示保留全部信息。
                """.strip()
            )

# -----------------------------
# Tab 3: Bridge (PCA = SVD)
# -----------------------------
with tabs[2]:
    st.subheader("PCA 本质上是中心化数据的 SVD")

    top = st.columns([1, 1, 1], gap="large")

    with top[0]:
        st.markdown("**选择桥接视图使用的数据集**")
        bridge_choice = st.radio(
            "数据集",
            ["使用当前二维数据", "使用当前三维数据"],
            label_visibility="collapsed",
        )
        if bridge_choice == "使用当前二维数据":
            pb = p
            Xb = X
        else:
            pb = p3
            Xb = X3

        st.markdown("**选择保留哪些奇异分量**")
        r = pb.s.shape[0]
        default_sel = list(range(min(r, 2)))  # default keep first 2
        selected = st.multiselect(
            "保留分量",
            options=list(range(r)),
            default=default_sel,
            format_func=lambda i: f"分量 {i+1}",
        )
        Xbhat = reconstruct_from_selected_singulars(pb, selected)
        errb = mse(Xb, Xbhat)

        st.write(f"**重建均方误差：** {errb:.6f}")
        st.caption("这里展示的就是对中心化矩阵 X 进行低秩重建，只保留你选中的奇异分量。")

    with top[1]:
        f_sigma, f_evr = fig_sigma_and_evr(pb)
        st.plotly_chart(f_sigma, use_container_width=True)

    with top[2]:
        f_sigma, f_evr = fig_sigma_and_evr(pb)
        st.plotly_chart(f_evr, use_container_width=True)

    st.divider()

    st.markdown("中心化数据 $X_c$ 的 SVD 为：")
    st.latex(r"X_c = U \Sigma V^\top")
    
    st.markdown(
        r"**主成分方向** 就是矩阵 $V$ 的列向量，而每个主成分捕获的方差与 "
        r"$\sigma_i^2$ 成正比（严格来说还会除以 $n-1$）。"
    )
    
    st.markdown(
        r"如果把某个奇异值 $\sigma_i$ 对应的分量关掉，就相当于删除了这部分结构信息，重建质量也会下降。"
)

