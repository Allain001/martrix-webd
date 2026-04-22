import io
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(page_title="SVD 图像压缩（灰度）", layout="wide")

st.title("SVD 图像压缩（单通道灰度图）")
st.write(
    "上传图像后，系统会先转成灰度矩阵，再用前 k 个奇异值完成重建。"
    " 你可以直观看到压缩率、能量保留和图像质量之间的关系。"
)

# -----------------------------
# Helpers
# -----------------------------
def pil_to_float_gray(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL image to grayscale float32 array in [0, 255]."""
    gray = pil_img.convert("L")
    return np.asarray(gray).astype(np.float32)

def safe_uint8(arr: np.ndarray) -> np.ndarray:
    """Clip to [0, 255] and convert to uint8 for display/export."""
    x = np.clip(arr, 0, 255)
    return x.astype(np.uint8)

def make_sample_image(size: int = 256) -> np.ndarray:
    """Synthetic grayscale image (gradient + shapes) for demo."""
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    grad = (x / (size - 1)) * 180.0 + (y / (size - 1)) * 60.0
    img = grad.copy()
    img[40:120, 30:190] += 40.0
    img[140:220, 80:140] -= 60.0
    cx, cy = size * 0.72, size * 0.35
    r2 = (x - cx) ** 2 + (y - cy) ** 2
    img += 50.0 * np.exp(-r2 / (2 * (size * 0.08) ** 2))
    return np.clip(img, 0, 255)

def resize_pil_keep_aspect(pil_img: Image.Image, max_side: int) -> Image.Image:
    w, h = pil_img.size
    if max(w, h) <= max_side:
        return pil_img
    scale = max_side / float(max(w, h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return pil_img.resize((new_w, new_h), Image.Resampling.BICUBIC)

@st.cache_data(show_spinner=False)
def compute_svd(A: np.ndarray):
    # For images this can be heavy, so keep full_matrices=False
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    return U, s, Vt

def reconstruct(U: np.ndarray, s: np.ndarray, Vt: np.ndarray, k: int) -> np.ndarray:
    # Efficient truncated SVD reconstruction
    Uk = U[:, :k]
    sk = s[:k]
    Vtk = Vt[:k, :]
    return (Uk * sk) @ Vtk

def compression_stats(m: int, n: int, k: int):
    """
    Original parameters: m*n
    Truncated SVD parameters: k*(m + n + 1)  (U: m*k, V: n*k, s: k)
    """
    original = m * n
    approx = k * (m + n + 1)
    ratio = approx / original if original > 0 else 1.0
    return ratio, original, approx

def energy_kept(s: np.ndarray, k: int) -> float:
    denom = float(np.sum(s ** 2))
    if denom == 0:
        return 0.0
    return float(np.sum(s[:k] ** 2) / denom)

def diff_image(A: np.ndarray, Ak: np.ndarray) -> np.ndarray:
    """Return a displayable absolute-difference image in uint8."""
    d = np.abs(A - Ak)
    dmax = float(d.max())
    if dmax <= 1e-12:
        return np.zeros_like(safe_uint8(A))
    d_norm = 255.0 * (d / dmax)
    return safe_uint8(d_norm)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("输入设置")
use_sample = st.sidebar.checkbox("使用内置样例图", value=True)
max_side = st.sidebar.slider("最长边缩放上限（加速用）", 64, 2048, 512, 32)

uploaded_file = None
if not use_sample:
    uploaded_file = st.sidebar.file_uploader(
        "上传图像（PNG/JPG/TIF/BMP）。彩色图会自动转成灰度图。",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
    )

# k slider will be defined after we know matrix size

# -----------------------------
# Load image
# -----------------------------
src_label = ""
uploaded_pil_resized = None
uploaded_is_color = False

if use_sample:
    A = make_sample_image(size=max_side)
    src_label = f"内置灰度样例图（{A.shape[1]}x{A.shape[0]}）"
else:
    if uploaded_file is None:
        st.info("请先上传图像，或勾选“使用内置样例图”。")
        st.stop()

    uploaded_bytes = uploaded_file.read()
    pil_img = Image.open(io.BytesIO(uploaded_bytes))

    # Detect if original is color-ish (RGB/RGBA/etc.)
    uploaded_is_color = pil_img.mode not in ["L", "I;16", "I", "F"]
    uploaded_pil_resized = resize_pil_keep_aspect(pil_img, max_side)

    A = pil_to_float_gray(uploaded_pil_resized)
    src_label = f"上传图像已缩放到 {A.shape[1]}x{A.shape[0]}（SVD 使用的是对应灰度矩阵）"

m, n = A.shape
k_max = min(m, n)
default_k = min(30, k_max)
k = st.slider("保留的奇异值个数 k", 1, k_max, default_k)

# -----------------------------
# Compute SVD and reconstruct
# -----------------------------
with st.spinner("Computing SVD and reconstruction..."):
    U, s, Vt = compute_svd(A)
    Ak = reconstruct(U, s, Vt, k)

A_u8 = safe_uint8(A)
Ak_u8 = safe_uint8(Ak)
D_u8 = diff_image(A, Ak)

ratio, original_params, approx_params = compression_stats(m, n, k)
if ratio > 1.0:
    st.warning(
        f"当前 k={k} 时并没有真正压缩参数量。截断 SVD 仍需约 {ratio*100:.1f}% 的参数，"
        f"已经接近或超过原始矩阵大小，建议减小 k。"
    )
ek = energy_kept(s, k)


# -----------------------------
# Display
# -----------------------------
if (not use_sample) and uploaded_is_color:
    # 2x2 layout for user color image
    r1c1, r1c2 = st.columns(2)
    r2c1, r2c2 = st.columns(2)

    # Top-left: color image
    with r1c1:
        st.subheader("原始彩色图")
        st.image(uploaded_pil_resized, caption="上传图像（缩放预览）", clamp=True)

    # Top-right: intentionally empty
    with r1c2:
        st.empty()

    # Bottom-left: grayscale used for SVD
    with r2c1:
        st.subheader("参与 SVD 的灰度图")
        st.caption(src_label)
        st.image(A_u8, caption="矩阵 A", clamp=True)

    # Bottom-right: compressed result
    with r2c2:
        st.subheader(f"压缩重建结果（k = {k}）")
        st.caption(f"能量保留：{ek*100:.2f}%   |   参数占比：{ratio*100:.2f}%")
        st.image(Ak_u8, caption="重建矩阵 A_k", clamp=True)

else:
    # Original 2-column layout (sample image or grayscale upload)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("参与 SVD 的灰度图")
        st.caption(src_label)
        st.image(A_u8, caption="矩阵 A", clamp=True)

    with col2:
        st.subheader(f"压缩重建结果（k = {k}）")
        st.caption(f"能量保留：{ek*100:.2f}%   |   参数占比：{ratio*100:.2f}%")
        st.image(Ak_u8, caption="重建矩阵 A_k", clamp=True)



# Difference directly under the two panels
st.subheader("绝对误差图 |A − A_k|（已缩放便于观察）")
st.image(D_u8, clamp=True)

# Metrics
st.markdown("---")
c1, c2, c3, c4 = st.columns(4)
c1.metric("矩阵尺寸", f"{m} x {n}")
c2.metric("原始参数量", f"{original_params:,}")
c3.metric("截断 SVD 参数量", f"{approx_params:,}")
c4.metric("参数占比", f"{ratio*100:.2f}%")

# Singular values plot
st.subheader("奇异值曲线（已排序）")
st.line_chart(s)

# Download compressed grayscale image
st.subheader("下载重建结果")
out = Image.fromarray(Ak_u8, mode="L")
buf = io.BytesIO()
out.save(buf, format="PNG")
st.download_button(
    "下载压缩后的灰度图（PNG）",
    data=buf.getvalue(),
    file_name=f"svd_compressed_k{k}.png",
    mime="image/png",
)
