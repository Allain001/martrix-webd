from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st


def hex_to_rgb(hex_str: str) -> tuple[int, int, int]:
    hex_str = hex_str.lstrip("#")
    return tuple(int(hex_str[i : i + 2], 16) for i in (0, 2, 4))


def blend_colors(hex1: str, hex2: str) -> str:
    rgb1 = hex_to_rgb(hex1)
    rgb2 = hex_to_rgb(hex2)
    avg_rgb = tuple(int((c1 + c2) / 2) for c1, c2 in zip(rgb1, rgb2))
    return f"rgb({avg_rgb[0]}, {avg_rgb[1]}, {avg_rgb[2]})"


def solve_lse_least_squares(planes: list[tuple[float, float, float, float]]):
    a_matrix = []
    b_vector = []
    normalized_normals = []

    for a_val, b_val, c_val, d_val in planes:
        normal = np.array([a_val, b_val, c_val], dtype=float)
        norm_val = np.linalg.norm(normal)
        if norm_val < 1e-10:
            continue
        a_matrix.append(normal / norm_val)
        b_vector.append(-d_val / norm_val)
        normalized_normals.append(normal / norm_val)

    if not a_matrix:
        return None, None, None, None

    a_matrix = np.array(a_matrix)
    b_vector = np.array(b_vector)
    x_ls, _, _, _ = np.linalg.lstsq(a_matrix, b_vector, rcond=None)
    dists = [abs(np.dot(a_matrix[i], x_ls) - b_vector[i]) for i in range(len(a_matrix))]
    rms_error = np.sqrt(np.mean(np.square(dists)))
    return x_ls, dists, rms_error, normalized_normals


def make_plane_mesh(a_val: float, b_val: float, c_val: float, d_val: float, extent: int = 5, n: int = 40):
    axis = np.linspace(-extent, extent, n)
    coeffs = np.array([abs(a_val), abs(b_val), abs(c_val)])
    main_axis = int(np.argmax(coeffs))
    if main_axis == 2:
        x_grid, y_grid = np.meshgrid(axis, axis)
        z_grid = -(a_val * x_grid + b_val * y_grid + d_val) / c_val
        return x_grid, y_grid, z_grid
    if main_axis == 1:
        x_grid, z_grid = np.meshgrid(axis, axis)
        y_grid = -(a_val * x_grid + c_val * z_grid + d_val) / b_val
        return x_grid, y_grid, z_grid
    y_grid, z_grid = np.meshgrid(axis, axis)
    x_grid = -(b_val * y_grid + c_val * z_grid + d_val) / a_val
    return x_grid, y_grid, z_grid


def get_intersection_line(p1, p2, extent: int):
    n1, n2 = np.array(p1[:3]), np.array(p2[:3])
    direction = np.cross(n1, n2)
    if np.linalg.norm(direction) < 1e-10:
        return None
    system = np.vstack([n1, n2])
    rhs = np.array([-p1[3], -p2[3]])
    point, _, _, _ = np.linalg.lstsq(system, rhs, rcond=None)
    scale = np.array([-extent * 2, extent * 2])
    return point + scale[:, np.newaxis] * (direction / np.linalg.norm(direction))


st.set_page_config(page_title="MatrixVis | Ax = b 工作站", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
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
        <h1>Ax = b / 最小二乘工作站</h1>
        <p>
            这个模块保留了原仓库里最有展示力的 3D 线性系统可视化方式，
            但把讲解口径改成了更适合 MatrixVis 的“方程组 + 几何解释”版本。
            你可以用它说明过定系统为什么没有精确交点，以及最小二乘解为什么是一个“误差最小”的空间点。
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header("1. 定义平面方程")
num_planes = st.sidebar.slider("方程数量", 2, 8, 4)

planes = []
defaults = [
    (1.0, 1.0, 1.0, -3.0),
    (1.0, -1.0, 0.0, 0.0),
    (0.0, 1.0, -1.0, 1.0),
    (1.0, 0.0, -2.0, 2.0),
]

columns = st.sidebar.columns(4)
for index, label in enumerate(["A", "B", "C", "D"]):
    columns[index].markdown(f"**{label}**")

for i in range(num_planes):
    row = st.sidebar.columns(4)
    default_values = defaults[i] if i < len(defaults) else (1.0, 0.0, 0.0, 1.0)
    values = [
        row[j].number_input(
            f"{label}{i}",
            value=default_values[j],
            key=f"{label}{i}",
            label_visibility="collapsed",
        )
        for j, label in enumerate("abcd")
    ]
    planes.append(tuple(values))

st.sidebar.markdown("---")
show_intersections = st.sidebar.toggle("显示平面交线", value=True)
show_projections = st.sidebar.toggle("显示残差线段", value=True)
plot_extent = st.sidebar.slider("图形范围", 1, 15, 5)

x_ls, dists, rms_error, normals = solve_lse_least_squares(planes)

figure = go.Figure()
plane_colors = ["#E56B6F", "#6FCF97", "#5B9DFF", "#F2B36D", "#C088F9", "#4FD1C5", "#F2994A", "#8D99AE"]

for i, plane in enumerate(planes):
    try:
        x_grid, y_grid, z_grid = make_plane_mesh(*plane, extent=plot_extent)
        figure.add_trace(
            go.Surface(
                x=x_grid,
                y=y_grid,
                z=z_grid,
                colorscale=[[0, plane_colors[i % len(plane_colors)]], [1, plane_colors[i % len(plane_colors)]]],
                opacity=0.28,
                showscale=False,
                name=f"平面 {i + 1}",
            )
        )
    except Exception:
        continue

if show_intersections:
    for i in range(len(planes)):
        for j in range(i + 1, len(planes)):
            line_points = get_intersection_line(planes[i], planes[j], plot_extent)
            if line_points is None:
                continue
            line_color = blend_colors(plane_colors[i % len(plane_colors)], plane_colors[j % len(plane_colors)])
            figure.add_trace(
                go.Scatter3d(
                    x=line_points[:, 0],
                    y=line_points[:, 1],
                    z=line_points[:, 2],
                    mode="lines",
                    line=dict(color=line_color, width=6),
                    name=f"交线 {i + 1}-{j + 1}",
                )
            )

if x_ls is not None:
    if show_projections:
        for plane, normal, color in zip(planes, normals, plane_colors):
            distance = np.dot(normal, x_ls) + (plane[3] / np.linalg.norm(plane[:3]))
            point_on_plane = x_ls - distance * normal
            figure.add_trace(
                go.Scatter3d(
                    x=[x_ls[0], point_on_plane[0]],
                    y=[x_ls[1], point_on_plane[1]],
                    z=[x_ls[2], point_on_plane[2]],
                    mode="lines",
                    line=dict(color=color, width=4),
                    showlegend=False,
                )
            )

    figure.add_trace(
        go.Scatter3d(
            x=[x_ls[0]],
            y=[x_ls[1]],
            z=[x_ls[2]],
            mode="markers+text",
            marker=dict(size=10, color="black", symbol="diamond", line=dict(width=2, color="white")),
            text=["LS 解"],
            textposition="top center",
            name="LS 解",
        )
    )

figure.update_layout(
    scene=dict(
        xaxis=dict(range=[-plot_extent, plot_extent]),
        yaxis=dict(range=[-plot_extent, plot_extent]),
        zaxis=dict(range=[-plot_extent, plot_extent]),
        aspectmode="cube",
    ),
    height=760,
    margin=dict(l=0, r=0, b=0, t=0),
)

st.plotly_chart(figure, use_container_width=True)

st.markdown("---")
left, right = st.columns([1, 2])

with left:
    st.subheader("数值结果")
    if x_ls is not None:
        st.markdown("**最优点**")
        st.latex(rf"x={x_ls[0]:.4f}, \quad y={x_ls[1]:.4f}, \quad z={x_ls[2]:.4f}")
        st.metric("RMS 总误差", f"{rms_error:.5f}")
        with st.expander("残差详情"):
            for i, dist in enumerate(dists):
                st.write(f"平面 {i + 1}: {dist:.5f}")
    else:
        st.warning("当前输入无法形成有效的最小二乘系统。")

with right:
    st.subheader("讲解面板")
    st.markdown(
        """
        把多个平面方程写成矩阵形式以后，我们得到：

        $$
        A x = y
        $$

        当方程数量多于未知数时，所有平面往往不会在同一点精确相交。
        这时我们寻找的不是“完全满足全部方程”的解，而是让误差平方和最小的最小二乘解。
        """
    )
    st.latex(
        r"""
        A = \begin{bmatrix}
        A_1 & B_1 & C_1 \\
        \vdots & \vdots & \vdots \\
        A_m & B_m & C_m
        \end{bmatrix},
        \quad
        y = \begin{bmatrix}
        -D_1 \\
        \vdots \\
        -D_m
        \end{bmatrix}
        """
    )
    st.markdown("正规方程写成：")
    st.latex(r"A^T A x = A^T y \quad \Rightarrow \quad x_{LS} = (A^T A)^{-1} A^T y")
    st.info("图中的短线表示 LS 点到各平面的垂直距离，也就是残差。")
