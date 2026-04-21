# -*- coding: utf-8 -*-

import base64
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

st.markdown(
    """
    <meta name="google-site-verification" content="8SxGZI_P4Z2GvLa6Sm_MNW3uJJXfHvKNMsegkjj1YQ8" />
    """,
    unsafe_allow_html=True,
)

st.set_page_config(page_title="martrixvis", layout="wide")
st.title("martrixvis")

st.markdown(
    """
    <div style="font-size:32px; font-weight:650; margin-top:-10px; margin-left: 200px">
              -- 线性代数交互可视化
    </div>
    """,
    unsafe_allow_html=True,
)

col_text, col_logo = st.columns([6, 1])

with col_text:
    st.markdown(
        """
        <div style="margin-top:20px; margin-left:250px; font-size:22px; line-height:1.6;">
            <div style="font-weight:600;">线性代数工作空间</div>
            <div style="margin-left:40px;">  矩阵变换、几何解释、SVD、PCA 与最小二乘</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_logo:
    st.markdown("<div style='padding-left:200px;'></div>", unsafe_allow_html=True)


st.markdown(
    """
### 关于本站

- 二维与三维矩阵变换
- SVD 与 PCA 的几何直觉
- 维度之间的投影与提升
- 图像压缩
- 最小二乘与 Ax = b
"""
)

BASE_DIR = Path(__file__).parent
VIDEOS_DIR = BASE_DIR / "videos"


@st.cache_data(show_spinner=False)
def load_gif_b64(path_str: str, file_mtime: float) -> str:
    with open(path_str, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def autoplay_gif_panel(gif_path: Path, title: str, height: int = 400) -> None:
    if gif_path.exists():
        st.markdown(title)
        b64 = load_gif_b64(str(gif_path), gif_path.stat().st_mtime)
        gif_html = (
            f'<img src="data:image/gif;base64,{b64}" '
            'style="width: 100%; max-width: 520px; height: auto; margin: 0 auto; display: block;" />'
        )
        components.html(gif_html, height=height)
    else:
        st.info(f"{gif_path.as_posix()} 未找到。")


col1, col2 = st.columns(2)

with col1:
    st.subheader("二维变换")
    st.write("用 2x2 矩阵、特征向量和 SVD 观察平面中的线性动作。")
    if st.button("进入二维变换", key="h2_2d"):
        st.switch_page("pages/1_2D_Transform.py")
    autoplay_gif_panel(
        VIDEOS_DIR / "SVD2x2Demo.gif",
        "##### 二维 SVD 演示",
        height=400,
    )

with col2:
    st.subheader("三维变换")
    st.write("观察 3x3 矩阵、特征向量与三维 SVD 路径。")
    if st.button("进入三维变换", key="h2_3d"):
        st.switch_page("pages/2_3D_Transform.py")
    autoplay_gif_panel(
        VIDEOS_DIR / "SVD3x3Demo.gif",
        "##### 三维 SVD 演示",
        height=400,
    )

st.markdown("---")

col3, col4 = st.columns(2)

with col3:
    st.subheader("三维到二维投影 (R3 -> R2)")
    st.write("借助 SVD，把三维点云投影到二维平面。")
    if st.button("进入三维到二维投影", key="h2_2x3"):
        st.switch_page("pages/3_2x3_Projection.py")
    autoplay_gif_panel(
        VIDEOS_DIR / "SVD2x3ProjectionDemo.gif",
        "##### 2x3 SVD 演示",
        height=400,
    )

with col4:
    st.subheader("二维到三维提升 (R2 -> R3)")
    st.write("借助 SVD，把二维点云提升到三维空间。")
    if st.button("进入二维到三维提升", key="h2_3x2"):
        st.switch_page("pages/4_3x2_Lifting.py")
    autoplay_gif_panel(
        VIDEOS_DIR / "SVD3x2LiftingDemo.gif",
        "##### 3x2 SVD 演示",
        height=400,
    )

st.markdown("---")

col5, col6 = st.columns(2)

with col5:
    st.subheader("PCA 演示")
    st.write("把 PCA 看成旋转与投影的组合。")
    if st.button("进入 PCA 演示", key="h2_pca"):
        st.switch_page("pages/5_PCA_Demo.py")
    autoplay_gif_panel(
        VIDEOS_DIR / "PCACartoon2D.gif",
        "##### PCA 演示",
        height=400,
    )

with col6:
    st.subheader("SVD 图像压缩")
    st.write("保留前 k 个奇异值，观察图像重建效果。")
    if st.button("进入 SVD 图像压缩", key="h2_svd_img"):
        st.switch_page("pages/6_SVDImgCompression.py")
    autoplay_gif_panel(
        VIDEOS_DIR / "SVD_Img.gif",
        "##### SVD 图像重建",
        height=400,
    )

st.markdown("---")

col7, col8 = st.columns(2)

with col7:
    st.subheader("PCA 图像压缩")
    st.write("观察均值图像、主成分与重建误差。")
    if st.button("进入 PCA 图像压缩", key="h2_pca_img"):
        st.switch_page("pages/7_PCAImgCompression.py")
    autoplay_gif_panel(
        VIDEOS_DIR / "PCAFacesDemo.gif",
        "##### PCA 人脸重建",
        height=400,
    )

with col8:
    st.subheader("最小二乘")
    st.write("用平面、交线、残差和最优点理解过定方程组。")
    if st.button("进入最小二乘", key="h2_lse"):
        st.switch_page("pages/8_LSE.py")
    autoplay_gif_panel(
        VIDEOS_DIR / "LSE3D_Demo.gif",
        "##### 最小二乘三维演示",
        height=400,
    )

st.markdown("---")
st.subheader("访问者地图")
st.write("查看访问者大致来自哪些地区。")

visitor_map_html = """
<div style="display:flex; justify-content:center; align-items:center; margin: 10px 0 20px 0;">
    <iframe
        src="https://s05.flagcounter.com/map/uX01/size_s/txt_000000/border_CCCCCC/pageviews_1/viewers_0/flags_0/"
        width="500"
        height="300"
        style="border:none;"
        scrolling="no">
    </iframe>
</div>
<p style="text-align:center; color:gray; font-size:13px;">
    访客数量与位置为近似统计结果。
</p>
"""
components.html(visitor_map_html, height=400)
