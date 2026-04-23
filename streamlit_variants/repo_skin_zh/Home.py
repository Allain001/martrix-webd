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

BASE_DIR = Path(__file__).parent
VIDEOS_DIR = BASE_DIR / "videos"


@st.cache_data(show_spinner=False)
def load_gif_b64(path_str: str, file_mtime: float) -> str:
    with open(path_str, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def autoplay_gif_panel(gif_path: Path, height: int = 330) -> None:
    if gif_path.exists():
        b64 = load_gif_b64(str(gif_path), gif_path.stat().st_mtime)
        gif_html = (
            f'<img src="data:image/gif;base64,{b64}" '
            'style="width: 100%; max-width: 560px; height: auto; margin: 0 auto; display: block; border-radius: 14px;" />'
        )
        components.html(gif_html, height=height)
    else:
        st.info(f"{gif_path.as_posix()} 未找到。")


st.markdown(
    """
    <style>
      .mv-hero {
        padding: 18px 8px 8px 8px;
      }
      .mv-kicker {
        color: #2563eb;
        font-size: 15px;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
      }
      .mv-title {
        font-size: 52px;
        font-weight: 800;
        line-height: 1.08;
        color: #111827;
        margin-top: 10px;
      }
      .mv-subtitle {
        font-size: 19px;
        line-height: 1.9;
        color: #4b5563;
        max-width: 860px;
        margin-top: 18px;
      }
      .mv-note {
        margin-top: 18px;
        padding: 14px 18px;
        border-radius: 16px;
        background: linear-gradient(135deg, #eff6ff 0%, #f8fafc 100%);
        border: 1px solid #dbeafe;
        color: #334155;
        font-size: 16px;
        line-height: 1.8;
      }
      .mv-band {
        margin-top: 28px;
        padding: 18px 22px;
        border-radius: 18px;
        background: #0f172a;
        color: #e2e8f0;
      }
      .mv-band-title {
        font-size: 15px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #67e8f9;
        font-weight: 700;
      }
      .mv-band-body {
        margin-top: 10px;
        font-size: 18px;
        line-height: 1.75;
      }
      .mv-section-title {
        font-size: 30px;
        font-weight: 800;
        color: #0f172a;
        margin-top: 20px;
      }
      .mv-section-body {
        font-size: 17px;
        color: #475569;
        line-height: 1.8;
        margin-top: 8px;
        max-width: 900px;
      }
      .mv-card {
        border: 1px solid #e5e7eb;
        border-radius: 20px;
        padding: 20px 22px;
        background: #ffffff;
        min-height: 188px;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.05);
      }
      .mv-card-tag {
        color: #7c3aed;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
      }
      .mv-card-title {
        font-size: 25px;
        font-weight: 800;
        color: #111827;
        margin-top: 10px;
      }
      .mv-card-text {
        font-size: 16px;
        line-height: 1.8;
        color: #475569;
        margin-top: 12px;
      }
      .mv-route {
        border-left: 4px solid #2563eb;
        padding: 10px 0 10px 18px;
        margin-top: 6px;
      }
      .mv-route strong {
        color: #0f172a;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="mv-hero">
      <div class="mv-kicker">martrixvis</div>
      <div class="mv-title">把线性代数讲成<br/>能看见的图形过程</div>
      <div class="mv-subtitle">
        用图形、动画和参数交互，把二维变换、三维空间、投影、PCA、SVD 与最小二乘这些概念放到同一套学习路径里。
      </div>
      <div class="mv-note">
        答辩推荐路线：<strong>首页 → 二维变换 → 最小二乘</strong>。如果评委继续追问，再补充投影、PCA 或图像压缩。
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

hero_left, hero_right = st.columns([7, 5], gap="large")

with hero_left:
    st.markdown(
        """
        <div class="mv-band">
          <div class="mv-band-title">推荐演示路线</div>
          <div class="mv-band-body">
            先用二维变换讲清矩阵怎样改变图形，再用最小二乘说明什么叫最佳近似。
            这一前一后，最能体现网站的教学价值。
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="mv-section-title">首页只保留三条主线</div>
        <div class="mv-section-body">
          先看几何变化，再看数据结构，最后看模型近似。答辩时不需要把每个模块都点一遍，重点讲清一条完整路径即可。
        </div>
        """,
        unsafe_allow_html=True,
    )

with hero_right:
    autoplay_gif_panel(VIDEOS_DIR / "SVD2x2Demo.gif")

st.markdown("---")

st.markdown(
    """
    <div class="mv-section-title">矩阵怎样动起来</div>
    <div class="mv-section-body">
      从平面到空间，从投影到提升，把矩阵作用直接变成图形变化。
    </div>
    """,
    unsafe_allow_html=True,
)

geo_a, geo_b, geo_c = st.columns(3, gap="large")

with geo_a:
    st.markdown(
        """
        <div class="mv-card">
          <div class="mv-card-tag">Geometry</div>
          <div class="mv-card-title">二维变换</div>
          <div class="mv-card-text">
            用单位正方形、网格和基向量解释 2x2 矩阵的旋转、拉伸与剪切。
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("进入二维变换", key="home_2d"):
        st.switch_page("pages/1_2D_Transform.py")

with geo_b:
    st.markdown(
        """
        <div class="mv-card">
          <div class="mv-card-tag">Space</div>
          <div class="mv-card-title">三维变换</div>
          <div class="mv-card-text">
            用立方体、点云和特征向量观察 3x3 矩阵如何改变空间结构。
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("进入三维变换", key="home_3d"):
        st.switch_page("pages/2_3D_Transform.py")

with geo_c:
    st.markdown(
        """
        <div class="mv-card">
          <div class="mv-card-tag">Projection</div>
          <div class="mv-card-title">投影与提升</div>
          <div class="mv-card-text">
            用降维和升维两页，解释维度变化时信息如何保留、映射与重组。
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    proj_btn, lift_btn = st.columns(2)
    with proj_btn:
        if st.button("投影", key="home_projection"):
            st.switch_page("pages/3_2x3_Projection.py")
    with lift_btn:
        if st.button("提升", key="home_lifting"):
            st.switch_page("pages/4_3x2_Lifting.py")

st.markdown("---")

left_data, right_data = st.columns([6, 6], gap="large")

with left_data:
    st.markdown(
        """
        <div class="mv-section-title">数据里的主方向</div>
        <div class="mv-section-body">
          PCA 和 SVD 图像压缩一起回答一个问题：怎样用更少的信息保留主要结构。
        </div>
        <div class="mv-route">
          <strong>PCA：</strong> 看主轴、投影和重建。<br/>
          <strong>SVD 图像压缩：</strong> 看保留秩与重建质量的关系。
        </div>
        """,
        unsafe_allow_html=True,
    )
    pca_btn, svd_btn = st.columns(2)
    with pca_btn:
        if st.button("进入 PCA 演示", key="home_pca"):
            st.switch_page("pages/5_PCA_Demo.py")
    with svd_btn:
        if st.button("进入 SVD 图像压缩", key="home_svd_img"):
            st.switch_page("pages/6_SVDImgCompression.py")

with right_data:
    autoplay_gif_panel(VIDEOS_DIR / "PCACartoon2D.gif")

st.markdown("---")

left_fit, right_fit = st.columns([6, 6], gap="large")

with left_fit:
    autoplay_gif_panel(VIDEOS_DIR / "LSE3D_Demo.gif")

with right_fit:
    st.markdown(
        """
        <div class="mv-section-title">什么叫最佳近似</div>
        <div class="mv-section-body">
          最小二乘模块把平面、残差和最优点放进同一视图里，帮助理解“没有精确解时怎样找最好解”。
        </div>
        <div class="mv-route">
          <strong>答辩建议：</strong> 如果只展示两个模块，就展示二维变换和最小二乘。
        </div>
        """,
        unsafe_allow_html=True,
    )
    fit_btn, extra_btn = st.columns(2)
    with fit_btn:
        if st.button("进入最小二乘", key="home_lse"):
            st.switch_page("pages/8_LSE.py")
    with extra_btn:
        if st.button("进入 PCA 图像压缩", key="home_pca_img"):
            st.switch_page("pages/7_PCAImgCompression.py")
