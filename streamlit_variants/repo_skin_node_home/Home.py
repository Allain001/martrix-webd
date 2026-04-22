# -*- coding: utf-8 -*-

import base64
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="martrixvis", layout="wide")

BASE_DIR = Path(__file__).parent
VIDEOS_DIR = BASE_DIR / "videos"


@st.cache_data(show_spinner=False)
def load_gif_b64(path_str: str, file_mtime: float) -> str:
    with open(path_str, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def autoplay_gif_panel(gif_path: Path, height: int = 340) -> None:
    if gif_path.exists():
        b64 = load_gif_b64(str(gif_path), gif_path.stat().st_mtime)
        gif_html = (
            f'<img src="data:image/gif;base64,{b64}" '
            'style="width:100%;max-width:560px;height:auto;display:block;margin:0 auto;border-radius:18px;" />'
        )
        components.html(gif_html, height=height)
    else:
        st.info(f"{gif_path.as_posix()} 未找到。")


st.markdown(
    """
    <style>
      .mv-shell {
        padding-top: 8px;
      }
      .mv-hero {
        position: relative;
        overflow: hidden;
        border-radius: 34px;
        padding: 42px 40px;
        background:
          radial-gradient(circle at top right, rgba(124, 58, 237, 0.22), transparent 28%),
          radial-gradient(circle at left center, rgba(0, 212, 255, 0.18), transparent 30%),
          linear-gradient(135deg, #07111f 0%, #101a31 55%, #0f172a 100%);
        color: #f8fafc;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 34px 100px rgba(2, 6, 23, 0.32);
      }
      .mv-kicker {
        display: inline-flex;
        padding: 8px 14px;
        border-radius: 999px;
        background: rgba(0, 212, 255, 0.10);
        border: 1px solid rgba(0, 212, 255, 0.24);
        color: #7dd3fc;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.22em;
        font-weight: 700;
      }
      .mv-title {
        font-size: 58px;
        font-weight: 800;
        line-height: 1.04;
        margin-top: 18px;
        letter-spacing: -0.03em;
      }
      .mv-subtitle {
        margin-top: 12px;
        font-size: 24px;
        color: #c4b5fd;
        font-weight: 600;
      }
      .mv-body {
        margin-top: 18px;
        max-width: 760px;
        font-size: 17px;
        line-height: 1.9;
        color: rgba(248, 250, 252, 0.84);
      }
      .mv-note {
        margin-top: 22px;
        padding: 16px 18px;
        border-radius: 18px;
        background: rgba(15, 23, 42, 0.42);
        border: 1px solid rgba(148, 163, 184, 0.18);
        color: #e2e8f0;
        font-size: 15px;
        line-height: 1.85;
      }
      .mv-stat-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 18px;
        margin-top: 28px;
      }
      .mv-stat {
        border-radius: 20px;
        padding: 18px;
        background: rgba(2, 6, 23, 0.28);
        border: 1px solid rgba(255,255,255,0.08);
      }
      .mv-stat-label {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.18em;
        color: #94a3b8;
      }
      .mv-stat-value {
        font-size: 28px;
        font-weight: 800;
        color: #fafafa;
        margin-top: 10px;
      }
      .mv-section-title {
        margin-top: 24px;
        font-size: 31px;
        font-weight: 800;
        color: #0f172a;
      }
      .mv-section-body {
        margin-top: 8px;
        font-size: 17px;
        line-height: 1.85;
        color: #475569;
        max-width: 920px;
      }
      .mv-card {
        border-radius: 24px;
        padding: 22px;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        box-shadow: 0 12px 34px rgba(15, 23, 42, 0.06);
        min-height: 198px;
      }
      .mv-card-tag {
        color: #00a8d8;
        font-size: 12px;
        letter-spacing: 0.22em;
        text-transform: uppercase;
        font-weight: 700;
      }
      .mv-card-title {
        margin-top: 10px;
        font-size: 24px;
        font-weight: 800;
        color: #111827;
      }
      .mv-card-text {
        margin-top: 12px;
        font-size: 16px;
        line-height: 1.8;
        color: #475569;
      }
      .mv-route {
        border-radius: 24px;
        padding: 22px;
        background: linear-gradient(135deg, #f8fbff 0%, #eef6ff 100%);
        border: 1px solid #dbeafe;
      }
      .mv-route-step {
        margin-top: 12px;
        border-radius: 18px;
        padding: 16px 18px;
        background: rgba(255,255,255,0.82);
        border: 1px solid #dbeafe;
      }
      .mv-route-kicker {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.24em;
        font-weight: 700;
        color: #0ea5e9;
      }
      .mv-route-title {
        margin-top: 8px;
        font-size: 24px;
        font-weight: 800;
        color: #0f172a;
      }
      .mv-route-text {
        margin-top: 8px;
        font-size: 16px;
        line-height: 1.8;
        color: #475569;
      }
      .mv-mini {
        border-radius: 22px;
        padding: 20px;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        box-shadow: 0 10px 28px rgba(15, 23, 42, 0.05);
        min-height: 170px;
      }
      .mv-mini-title {
        margin-top: 10px;
        font-size: 22px;
        font-weight: 800;
        color: #111827;
      }
      .mv-mini-text {
        margin-top: 10px;
        font-size: 15px;
        line-height: 1.8;
        color: #475569;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="mv-shell">', unsafe_allow_html=True)

st.markdown(
    """
    <div class="mv-hero">
      <div class="mv-kicker">martrixvis</div>
      <div class="mv-title">看见线性代数</div>
      <div class="mv-subtitle">从公式到直觉</div>
      <div class="mv-body">
        用图形、动画和参数交互，把二维变换、三维空间、投影、PCA、SVD 与最小二乘这些概念放到同一套可浏览、可操作、可讲解的网站里。
      </div>
      <div class="mv-note">
        推荐答辩路线：<strong>首页 → 二维变换 → 最小二乘</strong>。如果评委继续追问，再补充投影、PCA 或图像压缩模块。
      </div>
      <div class="mv-stat-grid">
        <div class="mv-stat">
          <div class="mv-stat-label">Core Modules</div>
          <div class="mv-stat-value">8</div>
        </div>
        <div class="mv-stat">
          <div class="mv-stat-label">Main Demo</div>
          <div class="mv-stat-value">2D Transform</div>
        </div>
        <div class="mv-stat">
          <div class="mv-stat-label">Best Follow-up</div>
          <div class="mv-stat-value">Least Squares</div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

hero_left, hero_right = st.columns([7, 5], gap="large")

with hero_left:
    st.markdown(
        """
        <div class="mv-section-title">模块导航</div>
        <div class="mv-section-body">
          首页只负责统一入口，真正的解释放到模块页完成。你可以直接从这里进入矩阵变换、数据分析和模型近似三条主线。
        </div>
        """,
        unsafe_allow_html=True,
    )

with hero_right:
    autoplay_gif_panel(VIDEOS_DIR / "SVD2x2Demo.gif")

st.markdown("---")

card_a, card_b, card_c = st.columns(3, gap="large")

with card_a:
    st.markdown(
        """
        <div class="mv-card">
          <div class="mv-card-tag">Geometry</div>
          <div class="mv-card-title">二维与三维变换</div>
          <div class="mv-card-text">
            从单位正方形到空间立方体，直接看矩阵如何改变几何对象。
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        if st.button("二维变换", key="node_home_2d"):
            st.switch_page("pages/1_2D_Transform.py")
    with c2:
        if st.button("三维变换", key="node_home_3d"):
            st.switch_page("pages/2_3D_Transform.py")

with card_b:
    st.markdown(
        """
        <div class="mv-card">
          <div class="mv-card-tag">Projection</div>
          <div class="mv-card-title">投影与提升</div>
          <div class="mv-card-text">
            从三维到二维，再从二维回到三维，理解维度变化时信息如何映射。
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c3, c4 = st.columns(2)
    with c3:
        if st.button("三维投影", key="node_home_projection"):
            st.switch_page("pages/3_2x3_Projection.py")
    with c4:
        if st.button("二维提升", key="node_home_lifting"):
            st.switch_page("pages/4_3x2_Lifting.py")

with card_c:
    st.markdown(
        """
        <div class="mv-card">
          <div class="mv-card-tag">Approximation</div>
          <div class="mv-card-title">最优近似与重建</div>
          <div class="mv-card-text">
            用 PCA、SVD 和最小二乘，把主方向、低秩重建和最佳拟合讲清楚。
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c5, c6 = st.columns(2)
    with c5:
        if st.button("PCA 演示", key="node_home_pca"):
            st.switch_page("pages/5_PCA_Demo.py")
    with c6:
        if st.button("最小二乘", key="node_home_lse"):
            st.switch_page("pages/8_LSE.py")
    c7, c8 = st.columns(2)
    with c7:
        if st.button("SVD 压缩", key="node_home_svd"):
            st.switch_page("pages/6_SVDImgCompression.py")
    with c8:
        if st.button("PCA 压缩", key="node_home_pca_img"):
            st.switch_page("pages/7_PCAImgCompression.py")

st.markdown("---")

route_left, route_right = st.columns([6, 6], gap="large")

with route_left:
    st.markdown(
        """
        <div class="mv-route">
          <div class="mv-section-title" style="margin-top:0;">推荐演示路线</div>
          <div class="mv-section-body" style="margin-top:6px;">
            用最少的页面，把网站价值讲清楚。
          </div>
          <div class="mv-route-step">
            <div class="mv-route-kicker">Step 01</div>
            <div class="mv-route-title">先看首页</div>
            <div class="mv-route-text">
              说明这是一套完整可运行的线性代数网站，而不是单个演示脚本。
            </div>
          </div>
          <div class="mv-route-step">
            <div class="mv-route-kicker">Step 02</div>
            <div class="mv-route-title">重点讲二维变换</div>
            <div class="mv-route-text">
              通过单位正方形、网格和基向量，把矩阵从符号变成动作。
            </div>
          </div>
          <div class="mv-route-step">
            <div class="mv-route-kicker">Step 03</div>
            <div class="mv-route-title">补最小二乘或投影</div>
            <div class="mv-route-text">
              用残差、最佳近似或降维映射说明网站不仅能展示，也能解释数学意义。
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with route_right:
    autoplay_gif_panel(VIDEOS_DIR / "LSE3D_Demo.gif")

st.markdown("---")

mini_a, mini_b, mini_c = st.columns(3, gap="large")

with mini_a:
    st.markdown(
        """
        <div class="mv-mini">
          <div class="mv-card-tag">Visual</div>
          <div class="mv-mini-title">几何直观</div>
          <div class="mv-mini-text">
            把旋转、拉伸、投影和拟合从公式里抽出来，直接做成图形变化。
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with mini_b:
    st.markdown(
        """
        <div class="mv-mini">
          <div class="mv-card-tag">Formula</div>
          <div class="mv-mini-title">公式共显</div>
          <div class="mv-mini-text">
            数学公式、参数控制和结果解释出现在同一页面里，方便边讲边看。
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with mini_c:
    st.markdown(
        """
        <div class="mv-mini">
          <div class="mv-card-tag">Teaching</div>
          <div class="mv-mini-title">教学导向</div>
          <div class="mv-mini-text">
            网站不是替用户算题，而是帮助用户建立对线性代数概念的图形理解。
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("</div>", unsafe_allow_html=True)
