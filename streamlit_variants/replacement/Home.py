from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components


BASE_DIR = Path(__file__).parent
VIDEOS_DIR = BASE_DIR / "videos"


def autoplay_gif_panel(gif_path: Path, title: str, height: int = 360) -> None:
    if not gif_path.exists():
        st.info(f"{gif_path.name} not found.")
        return
    data = base64.b64encode(gif_path.read_bytes()).decode("utf-8")
    html = (
        f'<img src="data:image/gif;base64,{data}" '
        'style="width:100%;max-width:520px;height:auto;display:block;margin:0 auto;border-radius:18px;" />'
    )
    st.markdown(title)
    components.html(html, height=height)


st.set_page_config(page_title="MatrixVis Studio", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1280px;}
    .hero {
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 28px;
        padding: 30px 34px;
        background:
            radial-gradient(circle at top right, rgba(88,160,255,0.16), transparent 28%),
            radial-gradient(circle at top left, rgba(242,176,97,0.14), transparent 24%),
            linear-gradient(135deg, rgba(18,22,31,0.96), rgba(10,12,18,0.94));
        box-shadow: 0 22px 72px rgba(0,0,0,0.3);
        margin-bottom: 1rem;
    }
    .hero-kicker {
        color: #d7a05f;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        font-weight: 800;
        font-size: 0.82rem;
    }
    .hero-title {
        color: #f5efe5;
        font-size: 3rem;
        font-weight: 900;
        line-height: 1.04;
        margin: 0.45rem 0 0.8rem 0;
    }
    .hero-copy {
        color: #bdb5a8;
        line-height: 1.8;
        max-width: 900px;
        font-size: 1rem;
    }
    .module-card {
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 24px;
        padding: 22px;
        background: rgba(17,21,30,0.88);
        min-height: 100%;
    }
    .module-card h3 {
        color: #f5efe5;
        margin-bottom: 0.55rem;
    }
    .module-card p {
        color: #b8b1a5;
        line-height: 1.75;
    }
    </style>
    <div class="hero">
        <div class="hero-kicker">MatrixVis Studio</div>
        <div class="hero-title">仓库替换版 MatrixVis</div>
        <div class="hero-copy">
            这一版保留了参考仓库那种“多模块首页”的组织方式，但每个模块都换成了 MatrixVis 自己的内容。
            你现在看到的是一个更像课程实验室的网站入口，而不是精简产品页。
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("### MatrixVis 模块总览")

row1_col1, row1_col2 = st.columns(2)
with row1_col1:
    st.markdown(
        """
        <div class="module-card">
            <h3>二维矩阵实验室</h3>
            <p>展示 2×2 矩阵如何改变点云、单位边界和特征方向，是最适合开场讲解的实验模块。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("进入二维矩阵实验室", key="go_matrix_lab", use_container_width=True):
        st.switch_page("pages/1_Matrix_Lab.py")
    autoplay_gif_panel(VIDEOS_DIR / "SVD2x2Demo.gif", "#### 预览动画")

with row1_col2:
    st.markdown(
        """
        <div class="module-card">
            <h3>案例走廊</h3>
            <p>把剪切、旋转、镜像、解方程这些场景变成可切换的讲解案例，适合答辩时快速切场。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("进入案例走廊", key="go_cases", use_container_width=True):
        st.switch_page("pages/2_Case_Gallery.py")
    autoplay_gif_panel(VIDEOS_DIR / "SVD2x3ProjectionDemo.gif", "#### 场景预览")

st.markdown("---")

row2_col1, row2_col2 = st.columns(2)
with row2_col1:
    st.markdown(
        """
        <div class="module-card">
            <h3>几何讲解板</h3>
            <p>把 det(A)、面积缩放、方向翻转、特征方向这些概念集中放进一个讲解面板里。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("进入几何讲解板", key="go_geometry_story", use_container_width=True):
        st.switch_page("pages/3_Geometry_Story.py")
    autoplay_gif_panel(VIDEOS_DIR / "SVD3x2LiftingDemo.gif", "#### 几何预览")

with row2_col2:
    st.markdown(
        """
        <div class="module-card">
            <h3>Ax = b 工作站</h3>
            <p>用 3D 平面系统讲清楚过定系统、最小二乘解和残差，是把图形直觉带回方程求解的模块。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("进入 Ax = b 工作站", key="go_solver_studio", use_container_width=True):
        st.switch_page("pages/4_Ax_b_Studio.py")
    autoplay_gif_panel(VIDEOS_DIR / "LSE3D_Demo.gif", "#### 求解预览")

st.markdown("---")
st.markdown(
    """
    #### 建议使用顺序
    1. 先从二维矩阵实验室建立图形直觉。  
    2. 然后切到案例走廊，快速展示你们设计好的教学场景。  
    3. 再用几何讲解板把 determinant、eigen、rank 讲成故事。  
    4. 最后用 Ax = b 工作站把几何直觉连接回方程求解。  
    """
)
