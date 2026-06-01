"""
=====================================================================
app.py — Streamlit UI
2D Plane Strain FEM for Pavement Analysis
เรียกใช้ fem_engine.py เป็น backend

โดย: อาจารย์อิทธิพล มีผล
KMUTNB - Pavement Engineering
=====================================================================
"""

import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.patches as mpatches

from fem_engine import (
    generate_mesh,
    assemble_global_K,
    apply_load,
    apply_boundary_conditions,
    solve_fem,
    compute_stress,
)

# =====================================================================
# Page Config
# =====================================================================

st.set_page_config(
    page_title="Pavement FEM Analysis",
    page_icon="🛣️",
    layout="wide",
)

# =====================================================================
# Custom CSS
# =====================================================================

st.markdown("""
<style>
    /* Header */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border-left: 5px solid #e94560;
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 1.8rem;
        margin: 0;
        font-weight: 700;
        letter-spacing: 1px;
    }
    .main-header p {
        color: #a8b2c1;
        margin: 0.3rem 0 0 0;
        font-size: 0.9rem;
    }

    /* Result metric cards */
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        border-left: 4px solid #0f3460;
        margin-bottom: 0.5rem;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #666;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1a1a2e;
    }
    .metric-unit {
        font-size: 0.85rem;
        color: #888;
    }

    /* Layer card */
    .layer-info {
        background: #f0f4ff;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        margin-bottom: 0.4rem;
        font-size: 0.85rem;
    }

    /* Warning / info */
    .info-box {
        background: #e8f4f8;
        border-left: 4px solid #17a2b8;
        padding: 0.8rem 1rem;
        border-radius: 6px;
        font-size: 0.85rem;
        color: #1a1a2e;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #1a1a2e;
    }
    section[data-testid="stSidebar"] * {
        color: #d0d8e4 !important;
    }
    section[data-testid="stSidebar"] .stSlider label {
        color: #a8b2c1 !important;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# Header
# =====================================================================

st.markdown("""
<div class="main-header">
    <h1>🛣️ Pavement FEM Analysis</h1>
    <p>2D Plane Strain Finite Element Analysis — KMUTNB Civil Engineering</p>
</div>
""", unsafe_allow_html=True)

# =====================================================================
# Sidebar — Input Parameters
# =====================================================================

with st.sidebar:
    st.markdown("## ⚙️ Parameters")

    st.markdown("### 🚛 Loading")
    pressure   = st.slider("Tire Pressure (MPa)", 0.3, 1.2, 0.70, 0.05,
                            help="ความดันล้อ ~ 0.55-0.75 MPa สำหรับรถบรรทุก")
    load_width = st.slider("Contact Width (m)",   0.10, 0.60, 0.35, 0.05,
                            help="ความกว้างพื้นที่สัมผัสล้อกับผิวถนน")

    st.markdown("### 📐 Mesh")
    nx     = st.slider("Elements (horizontal)", 6, 30, 12, 2,
                        help="จำนวน element แนวนอน เพิ่ม = ละเอียดขึ้น ช้าลง")
    width  = st.slider("Model Width (m)",       1.0, 5.0, 2.0, 0.5,
                        help="ความกว้างของโมเดล FEM")

    st.markdown("### 🗺️ Layer Properties")
    st.markdown("**Layer 1 — Asphalt Concrete**")
    E1  = st.slider("E₁ (MPa)", 500,  8000, 3000, 100)
    nu1 = st.slider("ν₁",       0.20, 0.45, 0.35, 0.01)
    h1  = st.slider("h₁ (m)",   0.05, 0.30, 0.15, 0.01)
    ny1 = st.slider("ny₁ (elements)", 2, 6, 3, 1)

    st.markdown("**Layer 2 — Base Course**")
    E2  = st.slider("E₂ (MPa)", 100,  1000, 300, 50)
    nu2 = st.slider("ν₂",       0.20, 0.45, 0.30, 0.01)
    h2  = st.slider("h₂ (m)",   0.10, 0.40, 0.20, 0.01)
    ny2 = st.slider("ny₂ (elements)", 2, 6, 4, 1)

    st.markdown("**Layer 3 — Subbase**")
    E3  = st.slider("E₃ (MPa)", 50,   500,  150, 25)
    nu3 = st.slider("ν₃",       0.20, 0.45, 0.30, 0.01)
    h3  = st.slider("h₃ (m)",   0.10, 0.50, 0.30, 0.05)
    ny3 = st.slider("ny₃ (elements)", 2, 8, 5, 1)

    st.markdown("**Layer 4 — Subgrade**")
    E4  = st.slider("E₄ (MPa)", 10,   200,  50,  10)
    nu4 = st.slider("ν₄",       0.25, 0.48, 0.40, 0.01)
    h4  = st.slider("h₄ (m)",   0.30, 1.00, 0.50, 0.05)
    ny4 = st.slider("ny₄ (elements)", 3, 10, 6, 1)

    st.markdown("### 🎨 Visualization")
    plot_type = st.selectbox("Result to Display", [
        "Von Mises Stress (MPa)",
        "Vertical Stress σ_y (MPa)",
        "Horizontal Stress σ_x (MPa)",
        "Shear Stress τ_xy (MPa)",
        "Settlement (mm)",
        "All Results (6 plots)",
    ])
    cmap_choice = st.selectbox("Color Map", [
        "jet", "rainbow", "RdYlBu_r", "plasma", "inferno", "RdBu_r", "PuOr"
    ])
    deformed    = st.checkbox("Show Deformed Shape", value=True)
    disp_scale  = st.slider("Deformation Scale", 100, 2000, 500, 100,
                             help="ขยาย displacement เพื่อให้เห็นชัด")

    run_btn = st.button("▶ RUN ANALYSIS", type="primary", use_container_width=True)

# =====================================================================
# Build Layer List
# =====================================================================

def build_layers(E1, nu1, h1, ny1, E2, nu2, h2, ny2,
                 E3, nu3, h3, ny3, E4, nu4, h4, ny4):
    return [
        {"name": "Asphalt Concrete", "E": E1, "nu": nu1,
         "thickness": h1, "ny": ny1, "color": "#2c2c2c"},
        {"name": "Base Course",      "E": E2, "nu": nu2,
         "thickness": h2, "ny": ny2, "color": "#8B7355"},
        {"name": "Subbase Course",   "E": E3, "nu": nu3,
         "thickness": h3, "ny": ny3, "color": "#C4A882"},
        {"name": "Subgrade",         "E": E4, "nu": nu4,
         "thickness": h4, "ny": ny4, "color": "#D4C5A9"},
    ]

# =====================================================================
# Plot Functions (inline ไม่ต้องใช้ไฟล์)
# =====================================================================

def plot_mesh_st(nodes, elements, mat_ids, layers, ax):
    layer_colors = [l["color"] for l in layers]
    for e_idx, elem in enumerate(elements):
        color = layer_colors[mat_ids[e_idx]]
        xe = nodes[elem[[0,1,2,3,0]], 0]
        ye = nodes[elem[[0,1,2,3,0]], 1]
        ax.fill(xe, ye, color=color, alpha=0.8, linewidth=0)
        ax.plot(xe, ye, 'k-', linewidth=0.3, alpha=0.4)

    patches = [mpatches.Patch(color=l["color"],
               label=f"{l['name']} E={l['E']} MPa")
               for l in layers]
    ax.legend(handles=patches, loc='lower right', fontsize=7)
    ax.set_xlabel("x (m)", fontsize=9)
    ax.set_ylabel("y (m)", fontsize=9)
    ax.set_title("FEM Mesh", fontsize=10, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)


def elem_to_node_avg(nodes, elements, values):
    """Average element values ไปยัง nodes"""
    n_nodes   = nodes.shape[0]
    node_vals = np.zeros(n_nodes)
    node_cnt  = np.zeros(n_nodes)
    for e_idx, elem in enumerate(elements):
        for nid in elem:
            node_vals[nid] += values[e_idx]
            node_cnt[nid]  += 1
    mask = node_cnt > 0
    node_vals[mask] /= node_cnt[mask]
    return node_vals


def plot_contour_st(nodes, elements, values, title, cmap, ax,
                    deformed=False, u=None, scale=500):
    node_vals = elem_to_node_avg(nodes, elements, values)

    if deformed and u is not None:
        xp = nodes[:, 0] + u[0::2] * scale
        yp = nodes[:, 1] + u[1::2] * scale
    else:
        xp = nodes[:, 0]
        yp = nodes[:, 1]

    triangles = []
    for elem in elements:
        n1, n2, n3, n4 = elem
        triangles.append([n1, n2, n3])
        triangles.append([n1, n3, n4])
    triangles = np.array(triangles)

    triang = mtri.Triangulation(xp, yp, triangles)
    tcf = ax.tricontourf(triang, node_vals, levels=50, cmap=cmap)
    plt.colorbar(tcf, ax=ax, shrink=0.85, pad=0.02)
    ax.triplot(triang, 'k-', linewidth=0.15, alpha=0.25)

    suffix = " (Deformed)" if (deformed and u is not None) else ""
    ax.set_title(f"{title}{suffix}", fontsize=10, fontweight='bold')
    ax.set_xlabel("x (m)", fontsize=9)
    ax.set_ylabel("y (m)", fontsize=9)
    ax.set_aspect('equal')


def make_single_fig(nodes, elements, mat_ids, layers,
                    u, stress, vm, plot_type, cmap, deformed, scale):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('#f5f7fa')

    # Left: Mesh
    plot_mesh_st(nodes, elements, mat_ids, layers, axes[0])

    # Right: Result contour
    if plot_type == "Von Mises Stress (MPa)":
        vals  = vm
        title = "Von Mises Stress (MPa)"
    elif plot_type == "Vertical Stress σ_y (MPa)":
        vals  = stress[:, 1]
        title = "σ_y Vertical Stress (MPa)"
    elif plot_type == "Horizontal Stress σ_x (MPa)":
        vals  = stress[:, 0]
        title = "σ_x Horizontal Stress (MPa)"
    elif plot_type == "Shear Stress τ_xy (MPa)":
        vals  = stress[:, 2]
        title = "τ_xy Shear Stress (MPa)"
    else:  # Settlement
        vals  = np.array([
            np.mean([u[2*nid+1] for nid in elem]) * 1000
            for elem in elements
        ])
        title = "Settlement (mm)"

    plot_contour_st(nodes, elements, vals, title, cmap,
                    axes[1], deformed, u, scale)

    plt.tight_layout()
    return fig


def make_all_fig(nodes, elements, mat_ids, layers,
                 u, stress, vm, cmap, deformed, scale):
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.patch.set_facecolor('#f5f7fa')
    fig.suptitle("2D Plane Strain FEM — Pavement Analysis",
                 fontsize=13, fontweight='bold', y=1.01)

    plot_mesh_st(nodes, elements, mat_ids, layers, axes[0, 0])

    v_elem = np.array([np.mean([u[2*nid+1] for nid in elem]) * 1000
                       for elem in elements])

    plot_contour_st(nodes, elements, v_elem,
                    "Settlement (mm)", "RdYlBu_r",
                    axes[0, 1], deformed, u, scale)

    plot_contour_st(nodes, elements, vm,
                    "Von Mises Stress (MPa)", cmap,
                    axes[0, 2], deformed, u, scale)

    plot_contour_st(nodes, elements, stress[:, 1],
                    "σ_y Vertical Stress (MPa)", "RdBu_r",
                    axes[1, 0])

    plot_contour_st(nodes, elements, stress[:, 0],
                    "σ_x Horizontal Stress (MPa)", "RdBu_r",
                    axes[1, 1])

    plot_contour_st(nodes, elements, stress[:, 2],
                    "τ_xy Shear Stress (MPa)", "PuOr",
                    axes[1, 2])

    plt.tight_layout()
    return fig

# =====================================================================
# Main Layout
# =====================================================================

# Info box ก่อนรัน
if "results" not in st.session_state:
    st.markdown("""
    <div class="info-box">
    📌 <b>วิธีใช้:</b> ปรับค่า Parameter ใน Sidebar ด้านซ้าย แล้วกด <b>▶ RUN ANALYSIS</b>
    <br>โปรแกรมจะ Solve FEM และแสดง Stress/Displacement Contour ทันที
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📋 Layer System (Default)")
    cols = st.columns(4)
    defaults = [
        ("Asphalt Concrete", 3000, 0.35, 0.15, "#2c2c2c"),
        ("Base Course",       300, 0.30, 0.20, "#8B7355"),
        ("Subbase Course",    150, 0.30, 0.30, "#C4A882"),
        ("Subgrade",           50, 0.40, 0.50, "#D4C5A9"),
    ]
    for col, (name, E, nu, h, color) in zip(cols, defaults):
        with col:
            st.markdown(f"""
            <div class="layer-info" style="border-left: 4px solid {color};">
            <b>{name}</b><br>
            E = {E} MPa<br>
            ν = {nu}<br>
            h = {h} m
            </div>
            """, unsafe_allow_html=True)

# =====================================================================
# Run Analysis
# =====================================================================

if run_btn:
    layers = build_layers(E1, nu1, h1, ny1, E2, nu2, h2, ny2,
                          E3, nu3, h3, ny3, E4, nu4, h4, ny4)

    with st.spinner("⚙️ Running FEM Analysis..."):
        try:
            nodes, elements, mat_ids, y_levels = generate_mesh(layers, width, nx)
            K, B_mats, D_mats = assemble_global_K(nodes, elements, mat_ids, layers)
            n_dof = 2 * len(nodes)
            F = apply_load(nodes, n_dof, width, load_width, pressure)
            K_mod, F_mod, _ = apply_boundary_conditions(K, F, nodes, n_dof)
            u = solve_fem(K_mod, F_mod)
            stress, strain, vm = compute_stress(u, elements, B_mats, D_mats)

            # บันทึก session state
            st.session_state["results"] = {
                "nodes": nodes, "elements": elements,
                "mat_ids": mat_ids, "layers": layers,
                "y_levels": y_levels, "u": u,
                "stress": stress, "strain": strain, "vm": vm,
            }
            st.success("✅ Analysis Complete!")

        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.stop()

# =====================================================================
# Display Results
# =====================================================================

if "results" in st.session_state:
    res = st.session_state["results"]
    nodes    = res["nodes"]
    elements = res["elements"]
    mat_ids  = res["mat_ids"]
    layers   = res["layers"]
    u        = res["u"]
    stress   = res["stress"]
    vm       = res["vm"]

    # --- Metric Cards ---
    st.markdown("### 📊 Key Results")
    c1, c2, c3, c4, c5 = st.columns(5)

    max_settlement = abs(np.min(u[1::2])) * 1000
    max_sigma_y    = abs(np.min(stress[:, 1]))
    max_sigma_x    = np.max(np.abs(stress[:, 0]))
    max_tau        = np.max(np.abs(stress[:, 2]))
    max_vm         = np.max(vm)

    with c1:
        st.metric("Settlement", f"{max_settlement:.3f} mm",
                  delta=None, help="การทรุดตัวสูงสุดที่ผิวถนน")
    with c2:
        st.metric("σ_y max", f"{max_sigma_y:.4f} MPa",
                  help="Vertical compressive stress สูงสุด")
    with c3:
        st.metric("σ_x max", f"{max_sigma_x:.4f} MPa",
                  help="Horizontal stress สูงสุด")
    with c4:
        st.metric("τ_xy max", f"{max_tau:.4f} MPa",
                  help="Shear stress สูงสุด")
    with c5:
        st.metric("Von Mises", f"{max_vm:.4f} MPa",
                  help="Von Mises stress สูงสุด")

    st.markdown("---")

    # --- Plot ---
    st.markdown("### 🎨 Visualization")

    if plot_type == "All Results (6 plots)":
        fig = make_all_fig(nodes, elements, mat_ids, layers,
                           u, stress, vm, cmap_choice, deformed, disp_scale)
    else:
        fig = make_single_fig(nodes, elements, mat_ids, layers,
                              u, stress, vm, plot_type,
                              cmap_choice, deformed, disp_scale)

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # --- Mesh Info ---
    with st.expander("📐 Mesh Information"):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.write(f"**Nodes:** {len(nodes)}")
            st.write(f"**Elements:** {len(elements)}")
        with col_b:
            st.write(f"**DOF:** {2 * len(nodes)}")
            st.write(f"**nx:** {nx}")
        with col_c:
            total_h = sum(l["thickness"] for l in layers)
            st.write(f"**Total depth:** {total_h} m")
            st.write(f"**Width:** {width} m")

    # --- Layer Summary Table ---
    with st.expander("📋 Layer Properties"):
        import pandas as pd
        df = pd.DataFrame([{
            "Layer": l["name"],
            "E (MPa)": l["E"],
            "ν": l["nu"],
            "h (m)": l["thickness"],
            "ny": l["ny"],
        } for l in layers])
        st.dataframe(df, use_container_width=True, hide_index=True)
