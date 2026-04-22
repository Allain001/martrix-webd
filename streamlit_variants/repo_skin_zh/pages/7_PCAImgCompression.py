# pages/6_Eigenfaces_PCA.py
# PCA Demo with two modes:
#
# Mode A) Built-in sample (faces): classic eigenfaces PCA on a face dataset (grayscale).
#   - Built-in sample path DISPLAY is relative: \assets\faces
#   - PCA uses ORIGINAL dataset pixels (no resizing before PCA)
#   - Display scale is display-only and keeps ratio (default 2x)
#
# Mode B) User image (RGB): PCA on pixels in RGB space (pixels are samples, RGB are features).
#   - Dataset source is either Built-in sample OR user image (no separate upload section)
#   - Show original + RGB channels in a 2x2 panel grid
#   - Show eigen images (PC score maps) in a 2x2 grid too (PC1, PC2, PC3, EVR plot)
#   - Display resize slider WORKS:
#       slider controls desired scale; we prevent cropping by capping render width to PANEL_MAX_W
#   - Reconstruct color image using k = 1,2,3 components and show difference image
#
# Dependencies: streamlit, numpy, plotly, pillow

from __future__ import annotations

import os
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image


# -----------------------------
# Repo root finder (portable)
# -----------------------------
def find_repo_root() -> Path:
    """
    Try to locate repo root robustly:
    1) If current working directory contains assets/faces, use it.
    2) Else walk upward from this file until we find assets/faces.
    3) Else fall back to a known absolute path.
    """
    cwd = Path.cwd()
    if (cwd / "assets" / "faces").is_dir():
        return cwd

    here = Path(__file__).resolve()
    for parent in [here.parent] + list(here.parents):
        if (parent / "assets" / "faces").is_dir():
            return parent

    return Path(r"D:\Visualizer2026")


REPO_ROOT = find_repo_root()
DEFAULT_FACES_DIR = str((REPO_ROOT / "assets" / "faces").resolve())
BUILTIN_FACES_DISPLAY = r"\assets\faces"


# -----------------------------
# PCA helpers
# -----------------------------
@dataclass
class PCAResult:
    mean: np.ndarray          # (p,)
    U: np.ndarray             # (n, r)
    s: np.ndarray             # (r,)
    Vt: np.ndarray            # (r, p)
    Xc: np.ndarray            # (n, p)
    scores: np.ndarray        # (n, r)


def pca_via_svd(X: np.ndarray) -> PCAResult:
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0)
    Xc = X - mean
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    scores = Xc @ Vt.T
    return PCAResult(mean=mean, U=U, s=s, Vt=Vt, Xc=Xc, scores=scores)


def explained_variance_ratio(p: PCAResult) -> np.ndarray:
    n = p.Xc.shape[0]
    if n <= 1:
        return np.zeros_like(p.s)
    vals = (p.s ** 2) / (n - 1)
    total = vals.sum() if vals.sum() > 0 else 1.0
    return vals / total


def reconstruct_one_vector(p: PCAResult, x: np.ndarray, k: int) -> np.ndarray:
    k = int(np.clip(k, 0, p.s.shape[0]))
    xc = x - p.mean
    if k == 0:
        return p.mean.copy()
    V = p.Vt.T
    z = xc @ V[:, :k]
    xhat = z @ V[:, :k].T + p.mean
    return xhat


def reconstruct_matrix(p: PCAResult, X: np.ndarray, k: int) -> np.ndarray:
    k = int(np.clip(k, 0, p.s.shape[0]))
    Xc = X - p.mean
    if k == 0:
        return np.tile(p.mean, (X.shape[0], 1))
    V = p.Vt.T
    Z = Xc @ V[:, :k]
    Xhat = Z @ V[:, :k].T + p.mean
    return Xhat


# -----------------------------
# Face dataset IO (no resize)
# -----------------------------
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".pgm", ".tif", ".tiff", ".webp"}


def list_images_recursive(root: str) -> List[str]:
    paths: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext in IMG_EXTS:
                paths.append(os.path.join(dirpath, fn))
    paths.sort()
    return paths


def load_gray_vec_no_resize(path: str) -> Tuple[np.ndarray, Tuple[int, int]]:
    img = Image.open(path).convert("L")
    w, h = img.size
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.flatten(), (w, h)


def vec_to_gray_img(vec: np.ndarray, size_wh: Tuple[int, int]) -> Image.Image:
    w, h = size_wh
    arr = vec.reshape((h, w))
    arr = np.clip(arr, 0.0, 1.0)
    arr_u8 = (arr * 255.0).astype(np.uint8)
    return Image.fromarray(arr_u8, mode="L")


def normalize_for_display(vec: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    vmin = float(np.min(vec))
    vmax = float(np.max(vec))
    if vmax - vmin < eps:
        return np.zeros_like(vec)
    return (vec - vmin) / (vmax - vmin)


# -----------------------------
# Plots
# -----------------------------
def fig_evr(evr: np.ndarray, cumulative: bool = True, height: int = 320) -> go.Figure:
    xs = np.arange(1, len(evr) + 1)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=xs, y=evr, name="EVR"))
    if cumulative:
        fig.add_trace(go.Scatter(x=xs, y=np.cumsum(evr), mode="lines+markers", name="Cumulative"))
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="component",
        yaxis_title="explained variance",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


# -----------------------------
# Streamlit UI
# -----------------------------
uploaded = None

st.set_page_config(page_title="PCA 图像压缩", layout="wide")
st.title("PCA 图像压缩")
st.caption("内置样例使用特征脸数据集；上传图片时会对 RGB 像素做 PCA，并展示主成分得分图与重建结果。")

with st.sidebar:
    st.subheader("数据来源")
    mode = st.radio("选择模式", ["内置样例（人脸）", "用户图片（RGB）"], index=0)

    st.divider()
    st.subheader("显示缩放（保持比例）")
    # For faces: default 2x; for user image: default 1.0 (panel cap handles large images)
    display_scale = st.slider(
        "Scale",
        min_value=0.1,
        max_value=6.0,
        value=2.0 if mode == "内置样例（人脸）" else 1.0,
        step=0.1,
    )

    if mode == "内置样例（人脸）":
        st.divider()
        st.subheader("内置样例路径")
        st.text_input("内置路径", value=BUILTIN_FACES_DISPLAY, disabled=True)

        st.divider()
        st.subheader("加载设置")
        max_imgs = st.slider("最多加载图像数", 20, 200, 100, 10)
        seed = st.number_input("打乱随机种子", value=7, step=1)
        shuffle = st.checkbox("打乱图像顺序", value=True)

        st.divider()
        st.subheader("特征脸视图")
        eig_cols = st.slider("特征脸每行列数", 2, 8, 4, 1)
        show_eigs = st.slider("展示多少张特征脸", 4, 64, 16, 4)

        st.divider()
        st.subheader("重建设置")
        k = st.slider("保留 k 个主成分", 0, 200, 25, 1)

    else:
        st.divider()
        st.subheader("用户图像输入")
        uploaded = st.file_uploader("上传彩色图像（png/jpg/webp）", type=["png", "jpg", "jpeg", "webp"])

        st.divider()
        st.subheader("RGB PCA 重建")
        k_rgb = st.slider("保留 k 个主成分（RGB）", 1, 3, 2, 1)
        show_evr_rgb = st.checkbox("显示解释方差", value=True)

        st.divider()
        st.subheader("自动适配面板")
        PANEL_MAX_W = st.slider("面板最大宽度（像素）", 220, 520, 360, 10)
        st.caption("不会裁切图像：在 2x2 网格中，图像宽度会被限制在这个上限内。")


# -----------------------------
# Mode A: Built-in eigenfaces
# -----------------------------
if mode == "内置样例（人脸）":
    root = DEFAULT_FACES_DIR
    if not root or not os.path.isdir(root):
        st.warning(f"未找到内置数据集目录：{root}")
        st.stop()

    paths_all = list_images_recursive(root)
    if len(paths_all) == 0:
        st.warning(f"在该目录下未找到图像：{root}")
        st.stop()

    rng = np.random.default_rng(int(seed))
    idx = np.arange(len(paths_all))
    if shuffle:
        rng.shuffle(idx)
    idx = idx[: int(max_imgs)]
    paths = [paths_all[i] for i in idx]

    @st.cache_data(show_spinner=False)
    def load_pack_no_resize(paths_in: List[str], root_dir: str) -> Tuple[np.ndarray, List[str], Tuple[int, int]]:
        X_list: List[np.ndarray] = []
        labels: List[str] = []
        base_size: Optional[Tuple[int, int]] = None

        for pth in paths_in:
            vec, sz = load_gray_vec_no_resize(pth)
            if base_size is None:
                base_size = sz
            if sz != base_size:
                raise ValueError(
                    f"Image size mismatch: {pth} is {sz}, expected {base_size}. "
                    "PCA requires all images have the same resolution."
                )
            X_list.append(vec)
            labels.append(os.path.relpath(pth, start=root_dir))

        if base_size is None or len(X_list) == 0:
            raise ValueError("No readable images loaded.")
        X = np.vstack(X_list).astype(np.float32)
        return X, labels, base_size

    try:
        X, labels, size_wh = load_pack_no_resize(paths, root)
    except Exception as e:
        st.error(str(e))
        st.stop()

    n, _p = X.shape
    pca = pca_via_svd(X)
    evr = explained_variance_ratio(pca)

    base_w, base_h = size_wh
    display_w = max(32, int(round(base_w * float(display_scale))))
    st.sidebar.caption(f"Dataset size: {base_w}×{base_h}")
    st.sidebar.caption(f"Display width: {display_w}px")

    left, right = st.columns([1.0, 1.8], gap="large")

    with left:
        st.subheader("均值人脸")
        st.image(vec_to_gray_img(pca.mean, size_wh), caption=f"均值人脸（n={n}）", width=display_w)
        st.subheader("解释方差（前 50 项）")
        st.plotly_chart(fig_evr(evr[: min(len(evr), 50)]), use_container_width=True)

    with right:
        st.subheader("特征脸")
        num_show = int(min(show_eigs, pca.Vt.shape[0]))
        rows = int(math.ceil(num_show / int(eig_cols)))

        k0 = 0
        for _ in range(rows):
            cols = st.columns(int(eig_cols))
            for c in cols:
                if k0 >= num_show:
                    break
                v = pca.Vt[k0, :]
                v_disp = normalize_for_display(v)
                c.image(
                    vec_to_gray_img(v_disp, size_wh),
                    caption=f"主成分 {k0+1}  解释方差比 {evr[k0]:.3f}",
                    width=display_w
                )
                k0 += 1

    st.divider()
    st.subheader("使用 k 个特征脸进行重建")

    sel = st.slider("选择数据集中的人脸编号", 0, n - 1, 0, 1)
    x = X[sel, :]
    xhat = reconstruct_one_vector(pca, x, k=int(k))

    diff = np.abs(x - xhat)
    diff_disp = normalize_for_display(diff)

    c1, c2, c3 = st.columns([1, 1, 1], gap="large")
    c1.image(vec_to_gray_img(x, size_wh), caption=f"原始人脸（编号 {sel}）", use_container_width=True)
    c2.image(vec_to_gray_img(xhat, size_wh), caption=f"重建人脸（k={int(k)}）", use_container_width=True)
    c3.image(vec_to_gray_img(diff_disp, size_wh), caption="差异图（绝对误差，已重缩放）", use_container_width=True)

    mse_val = float(np.mean((x - xhat) ** 2))
    st.write(f"**重建均方误差：** {mse_val:.6f}")
    st.stop()


# -----------------------------
# Mode B: User image (RGB) PCA on pixels
# -----------------------------
if uploaded is None:
    st.info("请先在侧边栏上传彩色图像，再进行 RGB 像素级 PCA。")
    st.stop()
else:
    try:
        img_rgb = Image.open(uploaded).convert("RGB")
    except Exception as e:
        st.error(f"图像打开失败：{e}")
        st.stop()
    else:
        arr = np.asarray(img_rgb, dtype=np.uint8)  # H x W x 3
        H, W, _ = arr.shape

        # Split channels
        r_ch = arr[:, :, 0]
        g_ch = arr[:, :, 1]
        b_ch = arr[:, :, 2]

        # Display sizing (NO CROPPING) + slider works:
        # - slider controls desired_w
        # - we cap actual rendered width to PANEL_MAX_W so it fits in 2×2 panels
        panel_w = int(PANEL_MAX_W)
        desired_w = int(round(W * float(display_scale)))
        display_w_user = max(64, min(desired_w, panel_w))
        effective_scale = display_w_user / float(W)

        st.sidebar.caption(f"Image size: {W}×{H}")
        st.sidebar.caption(f"Panel cap: {panel_w}px")
        st.sidebar.caption(f"Desired scale: {float(display_scale):.2f}  Effective scale: {effective_scale:.3f}")
        st.sidebar.caption(f"Panel display width: {display_w_user}px")

        st.subheader("用户图像与 RGB 通道")
        row1 = st.columns(2, gap="large")
        row2 = st.columns(2, gap="large")
        row1[0].image(img_rgb, caption="原始彩色图像", width=display_w_user)
        row1[1].image(Image.fromarray(r_ch, mode="L"), caption="R 通道（灰度）", width=display_w_user)
        row2[0].image(Image.fromarray(g_ch, mode="L"), caption="G 通道（灰度）", width=display_w_user)
        row2[1].image(Image.fromarray(b_ch, mode="L"), caption="B 通道（灰度）", width=display_w_user)

        # PCA on pixels: X shape (N,3), N = H*W
        Xpix = (arr.reshape(-1, 3).astype(np.float32)) / 255.0
        pca_rgb = pca_via_svd(Xpix)
        evr_rgb = explained_variance_ratio(pca_rgb)

        scores = pca_rgb.scores  # (N,3)
        pc1 = normalize_for_display(scores[:, 0]).reshape(H, W)
        pc2 = normalize_for_display(scores[:, 1]).reshape(H, W)
        pc3 = normalize_for_display(scores[:, 2]).reshape(H, W)

        st.divider()
        st.subheader("特征图像（主成分得分图）")
        erow1 = st.columns(2, gap="large")
        erow2 = st.columns(2, gap="large")

        erow1[0].image(
            Image.fromarray((pc1 * 255).astype(np.uint8), mode="L"),
            caption=f"主成分 1 得分图（解释方差比 {evr_rgb[0]:.3f}）",
            width=display_w_user
        )
        erow1[1].image(
            Image.fromarray((pc2 * 255).astype(np.uint8), mode="L"),
            caption=f"主成分 2 得分图（解释方差比 {evr_rgb[1]:.3f}）",
            width=display_w_user
        )
        erow2[0].image(
            Image.fromarray((pc3 * 255).astype(np.uint8), mode="L"),
            caption=f"主成分 3 得分图（解释方差比 {evr_rgb[2]:.3f}）",
            width=display_w_user
        )

        if show_evr_rgb:
            with erow2[1]:
                st.plotly_chart(fig_evr(evr_rgb, height=260), use_container_width=True)
        else:
            erow2[1].empty()

        # Reconstruct using k components (1..3), then reshape to image
        Xhat = reconstruct_matrix(pca_rgb, Xpix, k=int(k_rgb))
        arr_hat = np.clip(Xhat.reshape(H, W, 3) * 255.0, 0, 255).astype(np.uint8)

        # Difference image (magnitude across channels)
        diff = (arr.astype(np.int16) - arr_hat.astype(np.int16))
        diff_mag = np.sqrt(np.sum(diff.astype(np.float32) ** 2, axis=2))  # HxW
        diff_disp = normalize_for_display(diff_mag.flatten()).reshape(H, W)
        diff_u8 = (diff_disp * 255).astype(np.uint8)

        st.divider()
        st.subheader(f"使用前 {int(k_rgb)} 个主成分重建的彩色图像")
        rrow1 = st.columns(2, gap="large")
        rrow2 = st.columns(2, gap="large")

        rrow1[0].image(img_rgb, caption="原始彩色图像", width=display_w_user)
        rrow1[1].image(Image.fromarray(arr_hat, mode="RGB"), caption=f"重建结果（k={int(k_rgb)}）", width=display_w_user)
        rrow2[0].image(Image.fromarray(diff_u8, mode="L"), caption="差异图（RGB 误差幅值，已重缩放）", width=display_w_user)
        rrow2[1].empty()

        mse_rgb = float(np.mean((Xpix - Xhat) ** 2))
        st.write(f"**RGB 像素空间重建均方误差：** {mse_rgb:.6f}")

        with st.expander("这里发生了什么？"):
            st.markdown(
                """
- 这里的 PCA 作用在 **像素 RGB 向量** 上。
- 每个像素都是一个样本，特征维度是 3（R、G、B），所以矩阵 **X 的尺寸是 (H·W)×3**。
- 页面展示的“特征图像”其实是 **主成分得分图**，也就是每个像素投影到主成分后的取值，再重新排回 H×W。
- 当 k=1..3 时，系统会用低维颜色表示去逼近原图，同时给出误差图辅助观察。
                """.strip()
            )
