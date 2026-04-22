import numpy as np
import plotly.graph_objects as go
import streamlit as st


def hex_to_rgb(hex_str):
    hex_str = hex_str.lstrip("#")
    return tuple(int(hex_str[i:i + 2], 16) for i in (0, 2, 4))


def blend_colors(hex1, hex2):
    rgb1 = hex_to_rgb(hex1)
    rgb2 = hex_to_rgb(hex2)
    avg_rgb = tuple(int((c1 + c2) / 2) for c1, c2 in zip(rgb1, rgb2))
    return f"rgb({avg_rgb[0]}, {avg_rgb[1]}, {avg_rgb[2]})"


def solve_lse_least_squares(planes):
    a_matrix, b_vector, normalized_normals = [], [], []
    for (a_val, b_val, c_val, d_val) in planes:
        normal = np.array([a_val, b_val, c_val], dtype=float)
        norm_val = np.linalg.norm(normal)
        if norm_val < 1e-10:
            continue
        a_matrix.append(normal / norm_val)
        b_vector.append(-d_val / norm_val)
        normalized_normals.append(normal / norm_val)

    if not a_matrix:
        return None, None, None, None

    a_matrix, b_vector = np.array(a_matrix), np.array(b_vector)
    x_ls, _, _, _ = np.linalg.lstsq(a_matrix, b_vector, rcond=None)
    dists = [abs(np.dot(a_matrix[i], x_ls) - b_vector[i]) for i in range(len(a_matrix))]
    rms_error = np.sqrt(np.mean(np.square(dists)))
    return x_ls, dists, rms_error, normalized_normals


def make_plane_mesh(a_val, b_val, c_val, d_val, extent=5, n=40):
    lin = np.linspace(-extent, extent, n)
    coeffs = np.array([abs(a_val), abs(b_val), abs(c_val)])
    k = np.argmax(coeffs)
    if k == 2:
        x_grid, y_grid = np.meshgrid(lin, lin)
        z_grid = -(a_val * x_grid + b_val * y_grid + d_val) / c_val
        return x_grid, y_grid, z_grid
    if k == 1:
        x_grid, z_grid = np.meshgrid(lin, lin)
        y_grid = -(a_val * x_grid + c_val * z_grid + d_val) / b_val
        return x_grid, y_grid, z_grid
    y_grid, z_grid = np.meshgrid(lin, lin)
    x_grid = -(b_val * y_grid + c_val * z_grid + d_val) / a_val
    return x_grid, y_grid, z_grid


def get_intersection_line(p1, p2, extent):
    n1, n2 = np.array(p1[:3]), np.array(p2[:3])
    direction = np.cross(n1, n2)
    if np.linalg.norm(direction) < 1e-10:
        return None
    a_sys = np.vstack([n1, n2])
    b_sys = np.array([-p1[3], -p2[3]])
    p_line, _, _, _ = np.linalg.lstsq(a_sys, b_sys, rcond=None)
    t = np.array([-extent * 2, extent * 2])
    return p_line + t[:, np.newaxis] * (direction / np.linalg.norm(direction))


st.set_page_config(page_title="martrixvis | 最小二乘", layout="wide")
st.title("三维线性系统与最小二乘")

st.sidebar.header("1. 定义平面")
num_planes = st.sidebar.slider("方程数量", 2, 8, 4)

planes = []
defaults = [(1.0, 1.0, 1.0, -3.0), (1.0, -1.0, 0.0, 0.0), (0.0, 1.0, -1.0, 1.0), (1.0, 0.0, -2.0, 2.0)]

cols = st.sidebar.columns(4)
for idx, lbl in enumerate(["A", "B", "C", "D"]):
    cols[idx].markdown(f"**{lbl}**")

for i in range(num_planes):
    row_cols = st.sidebar.columns(4)
    dv = defaults[i] if i < len(defaults) else (1.0, 0.0, 0.0, 1.0)
    vals = [
        row_cols[j].number_input(f"{label}{i}", value=dv[j], key=f"{label}{i}", label_visibility="collapsed")
        for j, label in enumerate("abcd")
    ]
    planes.append(tuple(vals))

st.sidebar.markdown("---")
show_intersections = st.sidebar.toggle("显示平面交线", value=True)
show_projections = st.sidebar.toggle("显示残差", value=True)
plot_extent = st.sidebar.slider("绘图范围", 1, 15, 5)

x_ls, dists, rms_error, normals = solve_lse_least_squares(planes)

fig = go.Figure()
plane_colors = ["#E74C3C", "#2ECC71", "#3498DB", "#F39C12", "#9B59B6", "#1ABC9C", "#D35400", "#2C3E50"]

for i, plane in enumerate(planes):
    try:
        x_grid, y_grid, z_grid = make_plane_mesh(*plane, extent=plot_extent)
        fig.add_trace(
            go.Surface(
                x=x_grid,
                y=y_grid,
                z=z_grid,
                colorscale=[[0, plane_colors[i % 8]], [1, plane_colors[i % 8]]],
                opacity=0.3,
                showscale=False,
                name=f"平面 {i + 1}",
            )
        )
    except Exception:
        continue

if show_intersections:
    for i in range(len(planes)):
        for j in range(i + 1, len(planes)):
            pts = get_intersection_line(planes[i], planes[j], plot_extent)
            if pts is not None:
                line_color = blend_colors(plane_colors[i % 8], plane_colors[j % 8])
                fig.add_trace(
                    go.Scatter3d(
                        x=pts[:, 0],
                        y=pts[:, 1],
                        z=pts[:, 2],
                        mode="lines",
                        line=dict(color=line_color, width=6),
                        name=f"交线 {i + 1}&{j + 1}",
                    )
                )

if x_ls is not None:
    if show_projections:
        for i, (plane, normal) in enumerate(zip(planes, normals)):
            d_val = np.dot(normal, x_ls) + (plane[3] / np.linalg.norm(plane[:3]))
            p_on_plane = x_ls - d_val * normal
            fig.add_trace(
                go.Scatter3d(
                    x=[x_ls[0], p_on_plane[0]],
                    y=[x_ls[1], p_on_plane[1]],
                    z=[x_ls[2], p_on_plane[2]],
                    mode="lines",
                    line=dict(color=plane_colors[i % 8], width=4),
                    showlegend=False,
                )
            )

    fig.add_trace(
        go.Scatter3d(
            x=[x_ls[0]],
            y=[x_ls[1]],
            z=[x_ls[2]],
            mode="markers+text",
            marker=dict(size=10, color="black", symbol="diamond", line=dict(width=2, color="white")),
            text=["最小二乘解"],
            textposition="top center",
            name="最小二乘解",
        )
    )

fig.update_layout(
    scene=dict(
        xaxis=dict(range=[-plot_extent, plot_extent]),
        yaxis=dict(range=[-plot_extent, plot_extent]),
        zaxis=dict(range=[-plot_extent, plot_extent]),
        aspectmode="cube",
    ),
    height=750,
    margin=dict(l=0, r=0, b=0, t=0),
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown(
    """
    <style>
    .section-header {
        font-size: 36px !important;
        font-weight: bold;
    }
    .big-font {
        font-size: 24px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

res_col1, res_col2 = st.columns([1, 2])

with res_col1:
    st.markdown('<p class="section-header">数值结果</p>', unsafe_allow_html=True)
    if x_ls is not None:
        st.markdown('<p class="big-font">最优点 x_LS：</p>', unsafe_allow_html=True)
        st.latex(rf"x={x_ls[0]:.4f}, \quad y={x_ls[1]:.4f}, \quad z={x_ls[2]:.4f}")
        st.metric("总 RMS 误差", f"{rms_error:.5f}")
        with st.expander("残差详情"):
            for i, d_val in enumerate(dists):
                st.markdown(f'<p class="big-font">平面 {i + 1}: {d_val:.5f}</p>', unsafe_allow_html=True)

with res_col2:
    st.markdown('<p class="section-header">矩阵解释</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="big-font">当方程数量多于未知数时，所有平面通常不会交于同一点。'
        '最小二乘的目标，就是找到一个点，让它到所有平面的距离平方和尽量小。</p>',
        unsafe_allow_html=True,
    )
    st.latex(r"""
    A = \begin{bmatrix}
    A_1 & B_1 & C_1 \\
    \vdots & \vdots & \vdots \\
    A_m & B_m & C_m
    \end{bmatrix}, \quad
    \mathbf{y} = \begin{bmatrix} -D_1 \\ \vdots \\ -D_m \end{bmatrix}
    """)
    st.markdown('<p class="big-font">对应的正规方程为：</p>', unsafe_allow_html=True)
    st.latex(r"A^T A \mathbf{x} = A^T \mathbf{y} \quad \implies \quad \mathbf{x}_{LS} = (A^T A)^{-1} A^T \mathbf{y}")
    st.info("图中的短线就是残差，表示最小二乘解到各个平面的垂直距离。")
