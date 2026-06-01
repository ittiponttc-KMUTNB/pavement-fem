"""
=====================================================================
FEM Engine - Phase 1
2D Plane Strain Finite Element Analysis
สำหรับ Pavement Layered System
เขียนแบบ Scratch เพื่อการสอน

โดย: อาจารย์อิทธิพล มีผล
KMUTNB - Pavement Engineering
=====================================================================

โครงสร้างโปรแกรม:
1. กำหนด Material และ Layer
2. สร้าง Mesh (Q4 Element)
3. คำนวณ Element Stiffness Matrix [k]
4. Assemble Global Stiffness Matrix [K]
5. Apply Boundary Conditions
6. Solve {u} = [K]^-1 {f}
7. คำนวณ Strain และ Stress
8. แสดงผล (text + matplotlib)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.patches as mpatches

# =====================================================================
# SECTION 1: กำหนด Material Properties และ Layer
# =====================================================================

def define_layers():
    """
    กำหนด Layer สำหรับ Pavement System
    
    Returns:
        layers: list of dict แต่ละ layer มี
                - name     : ชื่อ layer
                - E        : Young's Modulus (MPa)
                - nu       : Poisson's ratio
                - thickness: ความหนา (m)
                - ny       : จำนวน element ในแนวตั้ง
                - color    : สีสำหรับ plot
    """
    layers = [
        {
            "name"      : "Asphalt Concrete",
            "E"         : 3000.0,   # MPa
            "nu"        : 0.35,
            "thickness" : 0.15,     # m (15 cm)
            "ny"        : 3,
            "color"     : "#2c2c2c"
        },
        {
            "name"      : "Base Course",
            "E"         : 300.0,    # MPa
            "nu"        : 0.30,
            "thickness" : 0.20,     # m (20 cm)
            "ny"        : 4,
            "color"     : "#8B7355"
        },
        {
            "name"      : "Subbase Course",
            "E"         : 150.0,    # MPa
            "nu"        : 0.30,
            "thickness" : 0.30,     # m (30 cm)
            "ny"        : 5,
            "color"     : "#C4A882"
        },
        {
            "name"      : "Subgrade",
            "E"         : 50.0,     # MPa
            "nu"        : 0.40,
            "thickness" : 0.50,     # m (50 cm)
            "ny"        : 6,
            "color"     : "#D4C5A9"
        },
    ]
    return layers


# =====================================================================
# SECTION 2: สร้าง Mesh (Q4 Quadrilateral Element)
# =====================================================================

def generate_mesh(layers, width=2.0, nx=10):
    """
    สร้าง Mesh แบบ Structured Q4 สำหรับ Layered System

    Q4 Element มี 4 Node เรียงแบบนี้:
    
        4 ------- 3
        |         |
        |    e    |
        |         |
        1 ------- 2
    
    Node numbering: bottom-left → bottom-right → top-right → top-left

    Parameters:
        layers : list of layer dict
        width  : ความกว้างของโมเดล (m)
        nx     : จำนวน element ในแนวนอน

    Returns:
        nodes    : array (n_nodes, 2) = [x, y] coordinates
        elements : array (n_elems, 4) = node indices (0-based)
        mat_ids  : array (n_elems,)   = layer index ของแต่ละ element
        layer_y  : list of y-coordinate ขอบล่างแต่ละ layer (จากบนลงล่าง)
    """

    # --- สร้าง x-coordinates (แนวนอน) ---
    x_coords = np.linspace(0.0, width, nx + 1)

    # --- สร้าง y-coordinates (แนวตั้ง จากบนลงล่าง) ---
    # y=0 คือผิวถนน, y<0 คือลงดิน
    y_levels = [0.0]
    for layer in layers:
        y_levels.append(y_levels[-1] - layer["thickness"])

    # สร้าง y-coordinates ทั้งหมด
    y_coords_list = []
    for i, layer in enumerate(layers):
        ny = layer["ny"]
        y_top = y_levels[i]
        y_bot = y_levels[i + 1]
        y_layer = np.linspace(y_top, y_bot, ny + 1)
        if i == 0:
            y_coords_list.extend(y_layer)
        else:
            # ไม่เอา y ซ้ำที่ interface
            y_coords_list.extend(y_layer[1:])

    y_coords = np.array(y_coords_list)
    n_y = len(y_coords)
    n_x = len(x_coords)

    # --- สร้าง Node Array ---
    # Node index = row * n_x + col
    # row 0 = y_coords[0] = ผิวถนน (บนสุด)
    nodes = np.zeros((n_y * n_x, 2))
    for row in range(n_y):
        for col in range(n_x):
            nid = row * n_x + col
            nodes[nid, 0] = x_coords[col]
            nodes[nid, 1] = y_coords[row]

    # --- สร้าง Element Array ---
    elements = []
    mat_ids  = []

    # หา layer index ของแต่ละ row ของ element
    # row ของ element 0 = ระหว่าง y_coords[0] และ y_coords[1]
    row_layer = []
    y_ptr = 0
    for i, layer in enumerate(layers):
        for j in range(layer["ny"]):
            row_layer.append(i)
            y_ptr += 1

    elem_row = 0
    for row in range(n_y - 1):
        for col in range(n_x - 1):
            # Q4 node order: bottom-left, bottom-right, top-right, top-left
            # (ใน array: row ล่าง = row+1 เพราะ y ลดลง)
            n1 = (row + 1) * n_x + col       # bottom-left
            n2 = (row + 1) * n_x + col + 1   # bottom-right
            n3 = row * n_x + col + 1          # top-right
            n4 = row * n_x + col              # top-left
            elements.append([n1, n2, n3, n4])
            mat_ids.append(row_layer[row])
        elem_row += 1

    elements = np.array(elements, dtype=int)
    mat_ids  = np.array(mat_ids, dtype=int)

    return nodes, elements, mat_ids, y_levels


# =====================================================================
# SECTION 3: Element Stiffness Matrix [k_e]
# =====================================================================

def get_D_matrix(E, nu):
    """
    สร้าง Constitutive Matrix [D] สำหรับ Plane Strain
    
    Plane Strain: ε_z = 0 (เหมาะกับ pavement ที่ยาวมาก)
    
    {σ_x}   [D11 D12  0 ] {ε_x}
    {σ_y} = [D12 D11  0 ] {ε_y}
    {τ_xy}  [ 0   0  D33] {γ_xy}
    
    Parameters:
        E  : Young's Modulus (MPa)
        nu : Poisson's ratio
    
    Returns:
        D : (3x3) numpy array
    """
    factor = E / ((1 + nu) * (1 - 2 * nu))
    D = factor * np.array([
        [1 - nu,    nu,          0.0         ],
        [nu,        1 - nu,      0.0         ],
        [0.0,       0.0,        (1 - 2*nu)/2 ]
    ])
    return D


def shape_functions(xi, eta):
    """
    Shape Functions ของ Q4 Element ใน Natural Coordinates (xi, eta)
    
    xi, eta ∈ [-1, 1]
    
    N1 = 1/4 (1-xi)(1-eta)  → node 1 (bottom-left)
    N2 = 1/4 (1+xi)(1-eta)  → node 2 (bottom-right)
    N3 = 1/4 (1+xi)(1+eta)  → node 3 (top-right)
    N4 = 1/4 (1-xi)(1+eta)  → node 4 (top-left)
    
    Returns:
        N     : (4,) shape function values
        dN_dxi: (2,4) derivatives [dN/dxi; dN/deta]
    """
    N = 0.25 * np.array([
        (1 - xi) * (1 - eta),
        (1 + xi) * (1 - eta),
        (1 + xi) * (1 + eta),
        (1 - xi) * (1 + eta)
    ])

    dN_dxi = 0.25 * np.array([
        [-(1 - eta),  (1 - eta), (1 + eta), -(1 + eta)],  # dN/dxi
        [-(1 - xi), -(1 + xi),  (1 + xi),   (1 - xi) ]   # dN/deta
    ])

    return N, dN_dxi


def element_stiffness(node_coords, D, thickness=1.0):
    """
    คำนวณ Element Stiffness Matrix [k_e] ขนาด (8x8)
    สำหรับ Q4 Plane Strain Element
    
    ใช้ Gauss Quadrature 2x2 point (เพียงพอสำหรับ Q4)
    
    [k_e] = ∫∫ [B]^T [D] [B] t dΩ
          ≈ Σ Σ [B(xi_i,eta_j)]^T [D] [B(xi_i,eta_j)] |J| wi wj t
    
    Parameters:
        node_coords : (4,2) coordinates ของ 4 node [x,y]
        D           : (3,3) constitutive matrix
        thickness   : ความหนาออกจอ plane (default=1.0 m สำหรับ plane strain)
    
    Returns:
        ke : (8,8) element stiffness matrix
        B_center : B matrix ที่จุดกึ่งกลาง (สำหรับ stress calculation)
    """
    # Gauss Points และ Weights สำหรับ 2x2 integration
    gp = np.array([-1.0/np.sqrt(3), 1.0/np.sqrt(3)])
    gw = np.array([1.0, 1.0])

    ke = np.zeros((8, 8))
    B_center = None

    for i, xi in enumerate(gp):
        for j, eta in enumerate(gp):
            N, dN_dxi = shape_functions(xi, eta)

            # Jacobian Matrix [J] = dN/dxi * x_node
            # J = [dx/dxi  dy/dxi ]
            #     [dx/deta dy/deta]
            J = dN_dxi @ node_coords  # (2,4) @ (4,2) = (2,2)
            detJ = np.linalg.det(J)

            if detJ <= 0:
                print(f"⚠️  Warning: Negative Jacobian detJ={detJ:.4f}")

            # dN/dx, dN/dy ใน Physical coordinates
            J_inv = np.linalg.inv(J)
            dN_dx = J_inv @ dN_dxi  # (2,4)

            # Strain-Displacement Matrix [B] ขนาด (3,8)
            # {ε} = [B] {u_e}
            # {ε} = {ε_x, ε_y, γ_xy}
            # {u_e} = {u1, v1, u2, v2, u3, v3, u4, v4}
            B = np.zeros((3, 8))
            for k in range(4):
                B[0, 2*k]     = dN_dx[0, k]   # dN/dx
                B[1, 2*k + 1] = dN_dx[1, k]   # dN/dy
                B[2, 2*k]     = dN_dx[1, k]   # dN/dy
                B[2, 2*k + 1] = dN_dx[0, k]   # dN/dx

            # ke += B^T D B |J| w_i w_j t
            ke += (B.T @ D @ B) * detJ * gw[i] * gw[j] * thickness

            # เก็บ B ที่จุดกลาง (xi=0, eta=0) สำหรับ stress
            if i == 0 and j == 0:
                N0, dN0 = shape_functions(0.0, 0.0)
                J0 = dN0 @ node_coords
                Ji0 = np.linalg.inv(J0)
                dN0_dx = Ji0 @ dN0
                B_center = np.zeros((3, 8))
                for k in range(4):
                    B_center[0, 2*k]     = dN0_dx[0, k]
                    B_center[1, 2*k + 1] = dN0_dx[1, k]
                    B_center[2, 2*k]     = dN0_dx[1, k]
                    B_center[2, 2*k + 1] = dN0_dx[0, k]

    return ke, B_center


# =====================================================================
# SECTION 4: Assemble Global Stiffness Matrix [K]
# =====================================================================

def assemble_global_K(nodes, elements, mat_ids, layers):
    """
    Assemble Global Stiffness Matrix [K] จาก Element Stiffness [k_e]
    
    DOF = 2 ต่อ node (u=แนวนอน, v=แนวตั้ง)
    Total DOF = 2 * n_nodes
    
    Global DOF ของ node i:
        u_i → DOF index 2*i
        v_i → DOF index 2*i + 1
    
    Returns:
        K       : (n_dof, n_dof) global stiffness matrix
        B_mats  : list of B matrix ของแต่ละ element (สำหรับ stress)
        D_mats  : list of D matrix ของแต่ละ element
    """
    n_nodes = nodes.shape[0]
    n_dof   = 2 * n_nodes
    K       = np.zeros((n_dof, n_dof))
    B_mats  = []
    D_mats  = []

    # สร้าง D matrix ของแต่ละ layer ก่อน
    D_layers = []
    for layer in layers:
        D = get_D_matrix(layer["E"], layer["nu"])
        D_layers.append(D)

    print(f"📐 Assembling Global K ({n_dof} x {n_dof})...")
    print(f"   จำนวน elements : {len(elements)}")
    print(f"   จำนวน nodes    : {n_nodes}")

    for e_idx, elem in enumerate(elements):
        # Node coordinates ของ element นี้ (4 nodes, 2 coords)
        node_coords = nodes[elem, :]   # (4,2)

        # D matrix ของ layer ที่ element นี้อยู่
        layer_idx = mat_ids[e_idx]
        D = D_layers[layer_idx]

        # Element Stiffness Matrix
        ke, B_center = element_stiffness(node_coords, D)
        B_mats.append(B_center)
        D_mats.append(D)

        # Assembly: วาง ke เข้า K ตาม DOF mapping
        # DOF ของ element = [2*n1, 2*n1+1, 2*n2, 2*n2+1, ...]
        dof_map = []
        for nid in elem:
            dof_map.append(2 * nid)
            dof_map.append(2 * nid + 1)
        dof_map = np.array(dof_map)  # (8,)

        # วาง ke เข้า K
        for i in range(8):
            for j in range(8):
                K[dof_map[i], dof_map[j]] += ke[i, j]

    print(f"   ✅ Assembly เสร็จแล้ว")
    return K, B_mats, D_mats


# =====================================================================
# SECTION 5: Load Vector และ Boundary Conditions
# =====================================================================

def apply_load(nodes, n_dof, width, load_width=0.30, pressure=0.7):
    """
    Apply Distributed Load บนผิวถนน (y = 0)
    
    จำลองล้อรถ: Uniform pressure บนพื้นที่สัมผัส
    
    Load อยู่กึ่งกลาง: x = [width/2 - load_width/2, width/2 + load_width/2]
    
    Parameters:
        nodes      : node array
        n_dof      : จำนวน DOF ทั้งหมด
        width      : ความกว้างโมเดล (m)
        load_width : ความกว้างพื้นที่รับโหลด (m) ~ contact width ของล้อ
        pressure   : ความดัน (MPa) ~ tire pressure
    
    Returns:
        F : (n_dof,) force vector
    """
    F = np.zeros(n_dof)

    x_center = width / 2.0
    y_tol    = 1e-8

    # หา surface nodes ทั้งหมดและ node spacing
    all_surf = sorted([i for i, nd in enumerate(nodes) if abs(nd[1]) < y_tol],
                      key=lambda i: nodes[i, 0])
    dx = (nodes[all_surf[1], 0] - nodes[all_surf[0], 0]) if len(all_surf) >= 2 else 0.0

    # ขยาย load zone ให้ครอบคลุม node spacing เพื่อให้ได้ >= 2 nodes เสมอ
    actual_load_width = max(load_width, 2.0 * dx + 1e-10)
    x_left  = x_center - actual_load_width / 2.0
    x_right = x_center + actual_load_width / 2.0

    # หา surface nodes ที่อยู่ใต้โหลด
    surface_nodes = sorted([i for i in all_surf
                            if (x_left - 1e-10) <= nodes[i, 0] <= (x_right + 1e-10)],
                           key=lambda i: nodes[i, 0])

    # กระจาย load ไปยัง surface nodes (tributary length)
    n_surf = len(surface_nodes)
    if n_surf < 2:
        print("⚠️  ไม่พบ surface node ใต้โหลด")
        return F

    # คำนวณ nodal force จาก distributed load
    # F_node = pressure * tributary_length * thickness (1.0 m)
    for idx, nid in enumerate(surface_nodes):
        x = nodes[nid, 0]
        # หา tributary length
        if idx == 0:
            x_next = nodes[surface_nodes[1], 0]
            trib = (x_next - x) / 2.0
        elif idx == n_surf - 1:
            x_prev = nodes[surface_nodes[-2], 0]
            trib = (x - x_prev) / 2.0
        else:
            x_prev = nodes[surface_nodes[idx - 1], 0]
            x_next = nodes[surface_nodes[idx + 1], 0]
            trib = (x_next - x_prev) / 2.0

        # Force ในแนวตั้ง (v direction = DOF 2*nid+1) เป็นลบ (กดลง)
        F[2 * nid + 1] = -pressure * trib * 1.0   # MPa * m * m = MN

    total_load = abs(sum(F))
    print(f"🚛 Applied Load:")
    print(f"   Pressure    : {pressure} MPa")
    print(f"   Load width  : {load_width} m")
    print(f"   Total force : {total_load:.4f} MN/m")

    return F


def apply_boundary_conditions(K, F, nodes, n_dof):
    """
    Apply Boundary Conditions (Penalty Method)
    
    Boundary Conditions สำหรับ Pavement Model:
    - ด้านล่าง (y = y_min) : u=0, v=0 (fixed)
    - ด้านซ้าย (x = 0)    : u=0       (roller, symmetry)
    - ด้านขวา (x = x_max) : u=0       (roller)
    - ด้านบน (y = 0)      : free      (รับโหลด)
    
    Penalty Method: K[dof,dof] += P, F[dof] += P * prescribed_value
    P = 1e15 * max(K)  (ค่าใหญ่มากแทน fixed)
    
    Returns:
        K_mod : K หลัง apply BC
        F_mod : F หลัง apply BC
        fixed_dofs : list of DOF ที่ถูก fix
    """
    K_mod = K.copy()
    F_mod = F.copy()

    x_min = nodes[:, 0].min()
    x_max = nodes[:, 0].max()
    y_min = nodes[:, 1].min()

    penalty = 1e15 * np.max(np.abs(K))
    fixed_dofs = []

    for i, node in enumerate(nodes):
        x, y = node

        # ด้านล่าง: fix ทั้ง u และ v
        if abs(y - y_min) < 1e-10:
            K_mod[2*i,   2*i]   += penalty  # u = 0
            K_mod[2*i+1, 2*i+1] += penalty  # v = 0
            fixed_dofs.extend([2*i, 2*i+1])

        # ด้านซ้าย: fix u เท่านั้น (roller)
        if abs(x - x_min) < 1e-10:
            K_mod[2*i, 2*i] += penalty       # u = 0
            if 2*i not in fixed_dofs:
                fixed_dofs.append(2*i)

        # ด้านขวา: fix u เท่านั้น (roller)
        if abs(x - x_max) < 1e-10:
            K_mod[2*i, 2*i] += penalty       # u = 0
            if 2*i not in fixed_dofs:
                fixed_dofs.append(2*i)

    print(f"🔒 Boundary Conditions Applied")
    print(f"   Fixed DOFs: {len(set(fixed_dofs))}")

    return K_mod, F_mod, list(set(fixed_dofs))


# =====================================================================
# SECTION 6: Solve System {u} = [K]^-1 {F}
# =====================================================================

def solve_fem(K_mod, F_mod):
    """
    แก้ระบบสมการ Linear: [K]{u} = {F}
    
    ใช้ np.linalg.solve (Gaussian Elimination)
    สำหรับปัญหาขนาดใหญ่ควรใช้ sparse solver
    
    Returns:
        u : (n_dof,) displacement vector
    """
    print(f"🔢 Solving system [{K_mod.shape[0]}x{K_mod.shape[1]}]...")
    u = np.linalg.solve(K_mod, F_mod)
    print(f"   ✅ Solved!")
    print(f"   Max |u| : {np.max(np.abs(u)):.6f} m")
    return u


# =====================================================================
# SECTION 7: คำนวณ Stress และ Strain
# =====================================================================

def compute_stress(u, elements, B_mats, D_mats):
    """
    คำนวณ Stress และ Strain ของแต่ละ element
    ที่จุดกึ่งกลาง (centroid)
    
    {σ} = [D][B]{u_e}
    {ε} = [B]{u_e}
    
    Returns:
        stress : (n_elem, 3) = [σ_x, σ_y, τ_xy] (MPa)
        strain : (n_elem, 3) = [ε_x, ε_y, γ_xy]
        vm     : (n_elem,)   = Von Mises Stress (MPa)
    """
    n_elem = len(elements)
    stress = np.zeros((n_elem, 3))
    strain = np.zeros((n_elem, 3))

    for e_idx, elem in enumerate(elements):
        # DOF ของ element นี้
        dof_map = []
        for nid in elem:
            dof_map.append(2 * nid)
            dof_map.append(2 * nid + 1)

        # Displacement ของ element
        u_e = u[dof_map]  # (8,)

        # Strain และ Stress
        B = B_mats[e_idx]   # (3,8)
        D = D_mats[e_idx]   # (3,3)

        eps = B @ u_e        # {ε} = [B]{u_e}
        sig = D @ eps        # {σ} = [D]{ε}

        strain[e_idx] = eps
        stress[e_idx] = sig

    # Von Mises Stress (plane strain)
    # σ_vm = sqrt(σx² - σxσy + σy² + 3τ²)
    sx  = stress[:, 0]
    sy  = stress[:, 1]
    txy = stress[:, 2]
    vm  = np.sqrt(sx**2 - sx*sy + sy**2 + 3*txy**2)

    print(f"\n📊 Stress Results:")
    print(f"   Max σ_y (vertical)  : {np.min(stress[:,1]):.4f} MPa (กด)")
    print(f"   Max σ_x (horizontal): {np.max(np.abs(stress[:,0])):.4f} MPa")
    print(f"   Max τ_xy (shear)    : {np.max(np.abs(stress[:,2])):.4f} MPa")
    print(f"   Max Von Mises       : {np.max(vm):.4f} MPa")
    print(f"\n📏 Displacement Results:")
    print(f"   Max settlement (v)  : {np.min(u[1::2])*1000:.4f} mm")
    print(f"   Max lateral (u)     : {np.max(np.abs(u[0::2]))*1000:.4f} mm")

    return stress, strain, vm


# =====================================================================
# SECTION 8: Visualization
# =====================================================================

def plot_mesh(nodes, elements, mat_ids, layers, y_levels, ax=None):
    """
    Plot Mesh แสดง Layer สี
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    layer_colors = [layer["color"] for layer in layers]

    for e_idx, elem in enumerate(elements):
        layer_idx = mat_ids[e_idx]
        color = layer_colors[layer_idx]
        x_e = nodes[elem, 0]
        y_e = nodes[elem, 1]
        # วน polygon
        x_poly = np.append(x_e[[0,1,2,3]], x_e[0])
        y_poly = np.append(y_e[[0,1,2,3]], y_e[0])
        ax.fill(x_poly, y_poly, color=color, alpha=0.7, linewidth=0)
        ax.plot(x_poly, y_poly, 'k-', linewidth=0.3, alpha=0.5)

    # Legend
    patches = []
    for i, layer in enumerate(layers):
        patch = mpatches.Patch(color=layer["color"], 
                               label=f"{layer['name']} (E={layer['E']} MPa)")
        patches.append(patch)
    ax.legend(handles=patches, loc='lower right', fontsize=8)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("FEM Mesh - Pavement Layered System")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    return ax


def plot_contour(nodes, elements, values, title, cmap='jet',
                 deformed=False, u=None, scale=500, ax=None):
    """
    Plot Contour ด้วย Triangulation (แบ่ง Q4 → 2 Triangles)
    แสดงค่า stress หรือ displacement เป็นเฉดสี
    
    Parameters:
        values  : ค่าที่ต้องการ plot (element-centered หรือ node-centered)
        scale   : Scale factor สำหรับ deformed shape
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    # ถ้าค่าเป็น element-centered → interpolate ไป node
    # วิธีง่าย: ใช้ค่าเฉลี่ยของ elements รอบๆ node
    n_nodes = nodes.shape[0]
    node_vals  = np.zeros(n_nodes)
    node_count = np.zeros(n_nodes)

    for e_idx, elem in enumerate(elements):
        for nid in elem:
            node_vals[nid]  += values[e_idx]
            node_count[nid] += 1

    # หาร average
    mask = node_count > 0
    node_vals[mask] /= node_count[mask]

    # Coordinates (deformed หรือ original)
    if deformed and u is not None:
        x_plot = nodes[:, 0] + u[0::2] * scale
        y_plot = nodes[:, 1] + u[1::2] * scale
    else:
        x_plot = nodes[:, 0]
        y_plot = nodes[:, 1]

    # สร้าง triangulation จาก Q4 elements
    triangles = []
    for elem in elements:
        n1, n2, n3, n4 = elem
        triangles.append([n1, n2, n3])  # triangle 1
        triangles.append([n1, n3, n4])  # triangle 2

    triangles = np.array(triangles)
    triang = mtri.Triangulation(x_plot, y_plot, triangles)

    # Plot filled contour
    levels = 50
    tcf = ax.tricontourf(triang, node_vals, levels=levels, cmap=cmap)
    plt.colorbar(tcf, ax=ax, label=title, shrink=0.8)

    # Plot mesh boundary (เบาๆ)
    ax.triplot(triang, 'k-', linewidth=0.2, alpha=0.3)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    title_full = f"{title}" + (" (Deformed)" if deformed else "")
    ax.set_title(title_full)
    ax.set_aspect('equal')

    return ax


def plot_all_results(nodes, elements, mat_ids, layers, y_levels, 
                     u, stress, strain, vm):
    """
    Plot ผลลัพธ์ทั้งหมดใน Figure เดียว (2x3 subplots)
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("2D Plane Strain FEM - Pavement Analysis Results", 
                 fontsize=14, fontweight='bold')

    # (0,0) Mesh
    plot_mesh(nodes, elements, mat_ids, layers, y_levels, ax=axes[0, 0])
    axes[0, 0].set_title("FEM Mesh")

    # (0,1) Vertical Displacement (Settlement)
    v_nodes = np.zeros(len(nodes))
    plot_contour(nodes, elements, stress[:, 1] * 0,  # dummy
                 "Vertical Displacement (m)", ax=axes[0, 1])
    # แก้ใหม่: ใช้ node displacement โดยตรง
    axes[0, 1].cla()
    v_elem = np.zeros(len(elements))
    for e_idx, elem in enumerate(elements):
        v_e = [u[2*nid+1] for nid in elem]
        v_elem[e_idx] = np.mean(v_e)
    plot_contour(nodes, elements, v_elem * 1000,
                 "Settlement (mm)", cmap='RdYlBu_r', ax=axes[0, 1])

    # (0,2) Deformed Shape + Von Mises
    plot_contour(nodes, elements, vm,
                 "Von Mises Stress (MPa)", cmap='jet',
                 deformed=True, u=u, scale=300, ax=axes[0, 2])

    # (1,0) Vertical Stress σ_y
    plot_contour(nodes, elements, stress[:, 1],
                 "σ_y Vertical Stress (MPa)", cmap='RdBu_r', ax=axes[1, 0])

    # (1,1) Horizontal Stress σ_x
    plot_contour(nodes, elements, stress[:, 0],
                 "σ_x Horizontal Stress (MPa)", cmap='RdBu_r', ax=axes[1, 1])

    # (1,2) Shear Stress τ_xy
    plot_contour(nodes, elements, stress[:, 2],
                 "τ_xy Shear Stress (MPa)", cmap='PuOr', ax=axes[1, 2])

    plt.tight_layout()
    return fig


# =====================================================================
# SECTION 9: Main Program
# =====================================================================

def run_fem_analysis(nx=10, width=2.0, load_width=0.15, pressure=0.7,
                     layers=None, verbose=True):
    """
    รัน FEM Analysis ทั้งหมด
    
    Parameters:
        nx         : จำนวน element แนวนอน
        width      : ความกว้างโมเดล (m)
        load_width : ความกว้างโหลด (m)
        pressure   : ความดันโหลด (MPa)
        layers     : list of layer dict (None = ใช้ default)
    
    Returns:
        dict ของผลลัพธ์ทั้งหมด
    """
    print("=" * 60)
    print("  2D Plane Strain FEM - Pavement Analysis")
    print("=" * 60)

    # Step 1: Define layers
    if layers is None:
        layers = define_layers()
    print(f"\n📋 Layer System:")
    for i, layer in enumerate(layers):
        print(f"   Layer {i+1}: {layer['name']}")
        print(f"            E={layer['E']} MPa, ν={layer['nu']}, "
              f"h={layer['thickness']} m, ny={layer['ny']}")

    # Step 2: Generate mesh
    print(f"\n🔧 Generating Mesh (nx={nx})...")
    nodes, elements, mat_ids, y_levels = generate_mesh(layers, width, nx)
    print(f"   Nodes    : {len(nodes)}")
    print(f"   Elements : {len(elements)}")

    # Step 3: Assemble K
    print(f"\n⚙️  Assembling Stiffness Matrix...")
    K, B_mats, D_mats = assemble_global_K(nodes, elements, mat_ids, layers)

    # Step 4: Apply Load
    print(f"\n📦 Applying Load...")
    n_dof = 2 * len(nodes)
    F = apply_load(nodes, n_dof, width, load_width, pressure)

    # Step 5: Boundary Conditions
    print(f"\n🔒 Applying Boundary Conditions...")
    K_mod, F_mod, fixed_dofs = apply_boundary_conditions(K, F, nodes, n_dof)

    # Step 6: Solve
    print(f"\n🔢 Solving...")
    u = solve_fem(K_mod, F_mod)

    # Step 7: Compute Stress
    print(f"\n📊 Computing Stress...")
    stress, strain, vm = compute_stress(u, elements, B_mats, D_mats)

    print("\n" + "=" * 60)
    print("  ✅ Analysis Complete!")
    print("=" * 60)

    return {
        "nodes"    : nodes,
        "elements" : elements,
        "mat_ids"  : mat_ids,
        "layers"   : layers,
        "y_levels" : y_levels,
        "u"        : u,
        "stress"   : stress,
        "strain"   : strain,
        "vm"       : vm,
        "width"    : width,
        "nx"       : nx,
    }


# =====================================================================
# SECTION 10: Run
# =====================================================================

if __name__ == "__main__":
    # รัน analysis
    results = run_fem_analysis(
        nx         = 12,       # elements แนวนอน (เพิ่มเพื่อความละเอียด)
        width      = 2.0,      # ความกว้างโมเดล (m)
        load_width = 0.35,     # ความกว้างโหลด (m) ~ contact width
        pressure   = 0.70,     # ความดัน (MPa) ~ tire pressure
    )

    # Plot ผลลัพธ์
    fig = plot_all_results(
        results["nodes"],
        results["elements"],
        results["mat_ids"],
        results["layers"],
        results["y_levels"],
        results["u"],
        results["stress"],
        results["strain"],
        results["vm"],
    )

    plt.savefig("fem_results.png", dpi=150, bbox_inches='tight')
    print("\n💾 บันทึกภาพ: fem_results.png")
    plt.show()
