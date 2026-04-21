from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components


BASE_DIR = Path(__file__).parent
VIDEOS_DIR = BASE_DIR / "videos"


st.set_page_config(page_title="martrixvis lite", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.2rem; padding-bottom: 2.2rem; max-width: 1240px;}
    .hero {
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 28px;
        padding: 30px 34px;
        background:
            radial-gradient(circle at top right, rgba(91,167,255,0.16), transparent 28%),
            radial-gradient(circle at top left, rgba(240,171,84,0.14), transparent 26%),
            linear-gradient(135deg, rgba(18,22,31,0.96), rgba(10,13,19,0.94));
        box-shadow: 0 22px 70px rgba(0,0,0,0.32);
        margin-bottom: 1rem;
    }
    .kicker {
        color: #d4a165;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        font-weight: 800;
        font-size: 0.82rem;
    }
    .title {
        color: #f5efe4;
        font-size: 3rem;
        font-weight: 900;
        margin: 0.55rem 0 0.75rem 0;
        line-height: 1.05;
    }
    .body {
        color: #beb6a9;
        line-height: 1.8;
        font-size: 1rem;
        max-width: 860px;
    }
    .section-card {
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 24px;
        padding: 22px;
        background: rgba(16,20,28,0.88);
        min-height: 100%;
    }
    .section-card h3 {
        color: #f5efe4;
        margin-bottom: 0.5rem;
    }
    .section-card p {
        color: #b7b0a4;
        line-height: 1.75;
    }
    </style>
    <div class="hero">
        <div class="kicker">martrixvis lite</div>
        <div class="title">矩阵可视化实验室</div>
        <div class="body">
            这是一版真正建立在可交互数学仓库内容上的 MatrixVis 网站入口。
            我们保留了“模块化实验室”的组织方式，但把首页叙事、教学目标和展示重点改成了更适合你们项目的中文版本。
            现在首页只留下两个最核心的模块：二维矩阵变换实验室，以及 Ax = b / 最小二乘工作站。
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_gif_b64(path_str: str, file_mtime: float) -> str:
    with open(path_str, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")


def autoplay_gif_panel(gif_path: Path, title: str, height: int = 360) -> None:
    if not gif_path.exists():
        st.info(f"{gif_path.as_posix()} 未找到。")
        return

    st.markdown(title)
    b64 = load_gif_b64(str(gif_path), gif_path.stat().st_mtime)
    gif_html = (
        f'<img src="data:image/gif;base64,{b64}" '
        f'style="width: 100%; max-width: 540px; height: auto; margin: 0 auto; display: block; border-radius: 18px;" />'
    )
    components.html(gif_html, height=height)


st.markdown("### 选择实验模块")

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        <div class="section-card">
            <h3>二维矩阵变换实验室</h3>
            <p>
                适合展示 2×2 矩阵如何改变点云、单位方格和方向结构。
                你可以用它讲旋转、缩放、镜像、特征方向，以及矩阵分解的动态过程。
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("进入二维矩阵实验室", key="go_lab", use_container_width=True):
        st.switch_page("pages/1_Matrix_Lab.py")
    autoplay_gif_panel(VIDEOS_DIR / "SVD2x2Demo.gif", "#### 预览动画")

with col2:
    st.markdown(
        """
        <div class="section-card">
            <h3>Ax = b / 最小二乘工作站</h3>
            <p>
                适合展示线性方程组、过定系统和最小二乘解的空间意义。
                页面会同时给出几何图像、最优点和误差解释，适合做讲解与答辩演示。
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("进入 Ax = b 工作站", key="go_solver", use_container_width=True):
        st.switch_page("pages/2_Ax_b_Studio.py")
    autoplay_gif_panel(VIDEOS_DIR / "LSE3D_Demo.gif", "#### 预览动画")

st.markdown("---")
st.markdown(
    """
    #### 使用建议

    1. 先从“二维矩阵变换实验室”开始，让观众建立图形直觉。  
    2. 再切到 “Ax = b / 最小二乘工作站”，把几何直觉连回方程求解。  
    3. 如果要现场展示，建议提前打开一次页面，确保在线服务已唤醒。  
    """
)
