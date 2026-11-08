"""
trajectory_visualization.py
===========================
2D Area of Convergence and 3D Spatial Trajectory Reconstruction Visualization Engine
for the Forensic Laboratory Suite (Phase 4.2).

Scientific & Architectural Principles:
--------------------------------------
1. Reuses trajectory.py exclusively for all mathematical, trigonometric, coordinate-transform,
   least-squares intersection, and uncertainty propagation calculations.
2. Strictly distinguishes:
   - Measured stain positions (X_s, Y_s)
   - Calculated impact angles (alpha = arcsin(W / L))
   - Supplied direction-of-travel angles (gamma in [0, 360) deg)
   - Back-projected trajectory rays (beta = (gamma + 180) mod 360 deg)
   - Estimated 2D Area of Convergence (planar region with RMS residual & uncertainty radius)
   - Estimated 3D Area of Origin (Tangent Method height range & 3D Skew-Ray CPA volume)
3. Supports multiple receiving surface orientations:
   - Horizontal Floor (Z_lab = 0, normal +Z_lab)
   - Vertical North Wall (Y_lab = wall_offset, normal -Y_lab)
   - Vertical West Wall (X_lab = 0, normal +X_lab)
   - Inclined Surface (custom pitch angle in degrees)
4. Generates both high-contrast dark forensic Matplotlib figures (PNG bytes / Base64 / file export)
   and structured 3D scene geometry JSON for interactive WebGL/Canvas inspection.
"""

import base64
import io
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from trajectory import (
    FORENSIC_TRAJECTORY_DISCLAIMER,
    StainTrajectoryInput,
    TrajectoryError,
    estimate_area_of_convergence_2d,
    estimate_area_of_origin_3d,
    reconstruct_single_stain_trajectory,
    surface_to_lab_coordinates,
)

# Preset receiving surface orientations in Global 3D Laboratory Coordinates (X_L, Y_L, Z_L in mm)
SURFACE_PRESETS: Dict[str, Dict[str, Any]] = {
    "Horizontal Floor": {
        "origin_lab_mm": (0.0, 0.0, 0.0),
        "axis_x_lab": (1.0, 0.0, 0.0),
        "axis_y_lab": (0.0, 1.0, 0.0),
        "description": "Horizontal receiving surface at Z_lab = 0 mm with upward normal [0, 0, +1].",
    },
    "Vertical Wall": {
        "origin_lab_mm": (0.0, 0.0, 0.0),
        "axis_x_lab": (1.0, 0.0, 0.0),
        "axis_y_lab": (0.0, 0.0, 1.0),
        "description": "Vertical wall in the (X_lab, Z_lab) plane at Y_lab = 0 mm with outward normal [0, -1, 0] or [0, +1, 0].",
    },
    "Vertical Wall (Y=0, +Y Normal)": {
        "origin_lab_mm": (0.0, 0.0, 0.0),
        "axis_x_lab": (0.0, 0.0, 1.0),
        "axis_y_lab": (1.0, 0.0, 0.0),
        "description": "Vertical wall at Y_lab = 0 mm with room-facing normal [0, +1, 0].",
    },
    "Inclined Surface (45°)": {
        "origin_lab_mm": (0.0, 0.0, 0.0),
        "axis_x_lab": (1.0, 0.0, 0.0),
        "axis_y_lab": (0.0, math.cos(math.radians(45.0)), math.sin(math.radians(45.0))),
        "description": "Inclined planar surface pitched at 45° relative to the horizontal floor.",
    },
}

# Forensic Dark Navy Theme Palette matching static/css/style.css
THEME_BG_MAIN = "#0b0f19"
THEME_BG_PANEL = "#131b2e"
THEME_GRID = "#24324f"
THEME_TEXT = "#f8fafc"
THEME_MUTED = "#94a3b8"
THEME_CRIMSON = "#e11d48"
THEME_CYAN = "#38bdf8"
THEME_EMERALD = "#10b981"
THEME_AMBER = "#f59e0b"
THEME_PURPLE = "#c084fc"

STAIN_PALETTE = [
    "#38bdf8",  # Sky Cyan
    "#fb7185",  # Rose
    "#34d399",  # Emerald
    "#fbbf24",  # Amber
    "#a78bfa",  # Violet
    "#f472b6",  # Pink
    "#2dd4bf",  # Teal
    "#f97316",  # Orange
]


def get_surface_basis(
    surface_type: str = "Horizontal Floor",
    incline_deg: Optional[float] = None,
    surface_origin_lab_mm: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """
    Resolve the 3D laboratory coordinate basis (origin, axis_x, axis_y) for a named
    receiving surface orientation or custom incline angle.
    """
    clean_type = (surface_type or "Horizontal Floor").strip()

    if incline_deg is not None:
        inc = float(incline_deg)
        if not math.isfinite(inc) or inc < -90.0 or inc > 90.0:
            raise TrajectoryError(f"Surface incline angle must be in [-90, 90] degrees; got {incline_deg}.")
        rad = math.radians(inc)
        origin = tuple(surface_origin_lab_mm) if surface_origin_lab_mm is not None else (0.0, 0.0, 0.0)
        return {
            "surface_type": f"Inclined Surface ({inc:.1f}°)",
            "origin_lab_mm": origin,
            "axis_x_lab": (1.0, 0.0, 0.0),
            "axis_y_lab": (0.0, math.cos(rad), math.sin(rad)),
            "description": f"Planar receiving surface inclined at {inc:.1f}° to horizontal.",
        }

    if clean_type == "Vertical Wall":
        # Choose right-handed axes in (X_lab, Z_lab) so e_x = [1,0,0], e_y = [0,0,1],
        # or flip so normal points into +Y_lab room space: e_x = [1,0,0], e_y = [0,0,-1] -> normal [0,1,0]
        origin = tuple(surface_origin_lab_mm) if surface_origin_lab_mm is not None else (0.0, 0.0, 0.0)
        return {
            "surface_type": "Vertical Wall",
            "origin_lab_mm": origin,
            "axis_x_lab": (1.0, 0.0, 0.0),
            "axis_y_lab": (0.0, 0.0, -1.0),
            "description": "Vertical wall at Y_lab = 0 mm with room-facing normal [0, +1, 0].",
        }

    if clean_type in SURFACE_PRESETS:
        preset = dict(SURFACE_PRESETS[clean_type])
        preset["surface_type"] = clean_type
        if surface_origin_lab_mm is not None:
            preset["origin_lab_mm"] = tuple(surface_origin_lab_mm)
        return preset

    raise TrajectoryError(
        f"Unsupported surface_type '{surface_type}'. Supported types: "
        f"{', '.join(SURFACE_PRESETS.keys())}."
    )


# ===========================================================================
# 1. Complete 2D & 3D Scene Data Builder
# ===========================================================================


def build_trajectory_scene_data(
    stains: Sequence[StainTrajectoryInput],
    surface_type: str = "Horizontal Floor",
    incline_deg: Optional[float] = None,
    surface_origin_lab_mm: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """
    Compute all 2D convergence and 3D origin estimates via trajectory.py and transform
    every stain, ray, convergence point, and 3D origin estimate into both:
      - Local Receiving Surface Coordinates (X_s, Y_s, Z_s) in mm
      - Global 3D Laboratory Coordinates (X_L, Y_L, Z_L) in mm

    Returns a serializable dictionary suitable for both Matplotlib rendering and
    interactive frontend 3D WebGL/Canvas inspection.
    """
    if not stains or len(stains) < 2:
        raise TrajectoryError("At least 2 directional bloodstains are required to reconstruct a trajectory scene.")

    basis = get_surface_basis(
        surface_type=surface_type,
        incline_deg=incline_deg,
        surface_origin_lab_mm=surface_origin_lab_mm,
    )

    # Delegate all scientific math to trajectory.py
    origin_3d_res = estimate_area_of_origin_3d(stains, require_calibrated=True, require_forward_rays=True)
    conv_2d = origin_3d_res["convergence_2d"]
    tan_orig = origin_3d_res["tangent_method_origin"]
    cpa_orig = origin_3d_res["skew_ray_cpa_origin"]

    # Transform 2D convergence point (X_c, Y_c, 0) into 3D Lab frame
    conv_lab = surface_to_lab_coordinates(
        point_surface_mm=[conv_2d["convergence_x_mm"], conv_2d["convergence_y_mm"], 0.0],
        surface_origin_lab_mm=basis["origin_lab_mm"],
        surface_axis_x_lab=basis["axis_x_lab"],
        surface_axis_y_lab=basis["axis_y_lab"],
    )["point_lab_mm"]

    # Transform Tangent-Method 3D origin (X_c, Y_c, Z_mean) into 3D Lab frame
    tan_lab = surface_to_lab_coordinates(
        point_surface_mm=[tan_orig["x_mm"], tan_orig["y_mm"], tan_orig["z_mean_mm"]],
        surface_origin_lab_mm=basis["origin_lab_mm"],
        surface_axis_x_lab=basis["axis_x_lab"],
        surface_axis_y_lab=basis["axis_y_lab"],
    )["point_lab_mm"]

    # Transform Skew-Ray CPA 3D origin (X*, Y*, Z*) into 3D Lab frame
    cpa_lab = surface_to_lab_coordinates(
        point_surface_mm=[cpa_orig["x_mm"], cpa_orig["y_mm"], cpa_orig["z_mm"]],
        surface_origin_lab_mm=basis["origin_lab_mm"],
        surface_axis_x_lab=basis["axis_x_lab"],
        surface_axis_y_lab=basis["axis_y_lab"],
    )["point_lab_mm"]

    c_2d_vec = np.array([conv_2d["convergence_x_mm"], conv_2d["convergence_y_mm"]], dtype=np.float64)
    cpa_surf_vec = np.array([cpa_orig["x_mm"], cpa_orig["y_mm"], cpa_orig["z_mm"]], dtype=np.float64)

    stain_records: List[Dict[str, Any]] = []
    for idx, st in enumerate(stains):
        rec = reconstruct_single_stain_trajectory(st)
        per_stain = origin_3d_res["per_stain_results"][idx]
        color = STAIN_PALETTE[idx % len(STAIN_PALETTE)]

        # Surface coordinates
        p_surf_2d = np.array([rec["surface_x_mm"], rec["surface_y_mm"]], dtype=np.float64)
        p_surf_3d = rec["origin_surface_mm"]
        u_surf_2d = rec["direction"]["planar_unit_vector"]
        d_surf_3d = rec["direction"]["vector_3d"]

        # Closest point on 2D back-projected ray to 2D convergence point (for miss-distance visualization)
        diff_2d = c_2d_vec - p_surf_2d
        t_2d = max(float(np.dot(diff_2d, u_surf_2d)), 0.0)
        closest_2d_on_ray = p_surf_2d + t_2d * u_surf_2d

        # 3D Ray endpoint extending slightly past the CPA/Tangent origin for clear visualization
        diff_3d = cpa_surf_vec - p_surf_3d
        t_3d_cpa = max(float(np.dot(diff_3d, d_surf_3d)), per_stain["planar_distance_mm"])
        ray_length_3d = t_3d_cpa * 1.12
        ray_end_surf_3d = p_surf_3d + ray_length_3d * d_surf_3d

        # Transform stain start & 3D ray endpoint into Global 3D Lab frame
        lab_start_info = surface_to_lab_coordinates(
            point_surface_mm=p_surf_3d,
            vector_surface=d_surf_3d,
            surface_origin_lab_mm=basis["origin_lab_mm"],
            surface_axis_x_lab=basis["axis_x_lab"],
            surface_axis_y_lab=basis["axis_y_lab"],
        )
        lab_end_info = surface_to_lab_coordinates(
            point_surface_mm=ray_end_surf_3d,
            surface_origin_lab_mm=basis["origin_lab_mm"],
            surface_axis_x_lab=basis["axis_x_lab"],
            surface_axis_y_lab=basis["axis_y_lab"],
        )

        stain_records.append({
            "stain_id": st.stain_id,
            "color": color,
            "width_mm": st.width,
            "length_mm": st.length,
            "ratio": rec["components"]["ratio"],
            "angle_degrees": rec["components"]["angle_degrees"],
            "sigma_angle_degrees": rec["uncertainty"]["sigma_angle_degrees"],
            "gamma_deg": rec["direction"]["gamma_deg"],
            "reverse_azimuth_deg": rec["direction"]["reverse_azimuth_deg"],
            "surface_x_mm": round(float(p_surf_2d[0]), 3),
            "surface_y_mm": round(float(p_surf_2d[1]), 3),
            "planar_unit_vector": [round(float(u_surf_2d[0]), 6), round(float(u_surf_2d[1]), 6)],
            "closest_2d_on_ray_mm": [round(float(closest_2d_on_ray[0]), 3), round(float(closest_2d_on_ray[1]), 3)],
            "perp_residual_2d_mm": conv_2d["per_stain_residuals_mm"][idx],
            "planar_distance_mm": per_stain["planar_distance_mm"],
            "tangent_height_z_mm": per_stain["height_z_mm"],
            "sigma_z_mm": per_stain["sigma_z_mm"],
            "ray_start_surface_mm": [round(float(x), 3) for x in p_surf_3d],
            "ray_end_surface_mm": [round(float(x), 3) for x in ray_end_surf_3d],
            "ray_start_lab_mm": [round(float(x), 3) for x in lab_start_info["point_lab_mm"]],
            "ray_end_lab_mm": [round(float(x), 3) for x in lab_end_info["point_lab_mm"]],
            "vector_lab": [round(float(x), 6) for x in lab_start_info["vector_lab"]],
        })

    return {
        "surface_basis": {
            "surface_type": basis["surface_type"],
            "description": basis["description"],
            "origin_lab_mm": list(basis["origin_lab_mm"]),
            "axis_x_lab": list(basis["axis_x_lab"]),
            "axis_y_lab": list(basis["axis_y_lab"]),
        },
        "stains": stain_records,
        "convergence_2d": {
            "x_surface_mm": conv_2d["convergence_x_mm"],
            "y_surface_mm": conv_2d["convergence_y_mm"],
            "point_lab_mm": [round(float(x), 3) for x in conv_lab],
            "rms_residual_mm": conv_2d["rms_residual_mm"],
            "max_residual_mm": conv_2d["max_residual_mm"],
            "convergence_radius_mm": conv_2d["convergence_radius_mm"],
            "condition_number": conv_2d["condition_number"],
        },
        "tangent_origin_3d": {
            "x_surface_mm": tan_orig["x_mm"],
            "y_surface_mm": tan_orig["y_mm"],
            "z_mean_surface_mm": tan_orig["z_mean_mm"],
            "z_median_surface_mm": tan_orig["z_median_mm"],
            "z_std_surface_mm": tan_orig["z_std_mm"],
            "z_combined_uncertainty_mm": tan_orig["z_combined_uncertainty_mm"],
            "z_range_surface_mm": list(tan_orig["z_range_mm"]),
            "point_lab_mm": [round(float(x), 3) for x in tan_lab],
        },
        "cpa_origin_3d": {
            "x_surface_mm": cpa_orig["x_mm"],
            "y_surface_mm": cpa_orig["y_mm"],
            "z_surface_mm": cpa_orig["z_mm"],
            "rms_3d_miss_mm": cpa_orig["rms_3d_miss_mm"],
            "point_lab_mm": [round(float(x), 3) for x in cpa_lab],
        },
        "stain_count": len(stain_records),
        "disclaimer": FORENSIC_TRAJECTORY_DISCLAIMER,
    }


# ===========================================================================
# 2. 2D Area of Convergence Matplotlib Visualization
# ===========================================================================


def render_2d_convergence_plot(
    stains: Sequence[StainTrajectoryInput],
    surface_type: str = "Horizontal Floor",
    output_path: Optional[str] = None,
    dpi: int = 140,
) -> Dict[str, Any]:
    """
    Generate a high-contrast forensic 2D Area of Convergence figure displaying:
    - Measured bloodstain positions (X_s, Y_s) in millimeters
    - Forward travel direction arrows (gamma)
    - Back-projected trajectory rays (beta = gamma + 180°)
    - Individual perpendicular miss-distance segments to the convergence centroid
    - Estimated 2D Convergence Centroid (X_c, Y_c)
    - Convergence Uncertainty Radius circle & RMS residual annotation
    """
    scene = build_trajectory_scene_data(stains=stains, surface_type=surface_type)
    conv = scene["convergence_2d"]
    cx, cy = conv["x_surface_mm"], conv["y_surface_mm"]
    radius = conv["convergence_radius_mm"]

    fig, ax = plt.subplots(figsize=(8.2, 6.2), dpi=dpi)
    fig.patch.set_facecolor(THEME_BG_MAIN)
    ax.set_facecolor(THEME_BG_PANEL)

    # Coordinate bounds calculation
    all_x = [cx - radius * 1.5, cx + radius * 1.5]
    all_y = [cy - radius * 1.5, cy + radius * 1.5]
    for st in scene["stains"]:
        all_x.append(st["surface_x_mm"])
        all_y.append(st["surface_y_mm"])

    span = max(max(all_x) - min(all_x), max(all_y) - min(all_y), 100.0)
    arrow_len = span * 0.08

    # 1. Draw each stain, its travel direction arrow, back-projected ray, and miss-distance vector
    for idx, st in enumerate(scene["stains"]):
        sx, sy = st["surface_x_mm"], st["surface_y_mm"]
        rx, ry = st["closest_2d_on_ray_mm"]
        ux, uy = st["planar_unit_vector"]
        color = st["color"]

        # Back-projected trajectory ray (extended slightly past closest approach)
        ext_x = rx + ux * (span * 0.12)
        ext_y = ry + uy * (span * 0.12)
        ax.plot(
            [sx, ext_x],
            [sy, ext_y],
            color=color,
            linestyle="--",
            linewidth=1.6,
            alpha=0.85,
            label=f"{st['stain_id']} Back-Ray (α={st['angle_degrees']:.1f}°, β={st['reverse_azimuth_deg']:.1f}°)",
        )

        # Forward direction-of-travel arrow (gamma = opposite of back-projection unit vector)
        fwd_dx = -ux * arrow_len
        fwd_dy = -uy * arrow_len
        ax.annotate(
            "",
            xy=(sx + fwd_dx, sy + fwd_dy),
            xytext=(sx, sy),
            arrowprops=dict(arrowstyle="->", color=color, lw=2.0),
        )

        # Perpendicular miss-distance segment from convergence estimate to ray
        if st["perp_residual_2d_mm"] > 1e-3:
            ax.plot(
                [cx, rx],
                [cy, ry],
                color=THEME_AMBER,
                linestyle=":",
                linewidth=1.3,
                alpha=0.9,
                label="Perpendicular Miss Distance" if idx == 0 else None,
            )

        # Measured stain marker
        ax.scatter(
            [sx],
            [sy],
            color=color,
            edgecolors="#ffffff",
            s=75,
            zorder=5,
        )
        ax.text(
            sx + span * 0.018,
            sy + span * 0.018,
            f"{st['stain_id']}\n({sx:.0f}, {sy:.0f}) mm",
            color=THEME_TEXT,
            fontsize=8,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=THEME_BG_MAIN, edgecolor=color, alpha=0.8),
        )

    # 2. Draw Convergence Uncertainty Region Circle
    unc_circle = Circle(
        (cx, cy),
        radius=max(radius, span * 0.02),
        facecolor=(225 / 255, 29 / 255, 72 / 255, 0.18),
        edgecolor=THEME_CRIMSON,
        linestyle="-",
        linewidth=1.8,
        label=f"Area of Convergence (R = ±{radius:.1f} mm)",
        zorder=4,
    )
    ax.add_patch(unc_circle)

    # 3. Draw Estimated Convergence Centroid
    ax.scatter(
        [cx],
        [cy],
        color=THEME_CRIMSON,
        edgecolors="#ffffff",
        marker="X",
        s=130,
        zorder=6,
        label=f"Convergence Estimate ({cx:.1f}, {cy:.1f}) mm",
    )

    # Styling & Grid
    pad = span * 0.22
    ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
    ax.set_ylim(min(all_y) - pad, max(all_y) + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color=THEME_GRID, linestyle="--", linewidth=0.7, alpha=0.7)

    ax.set_title(
        f"2D Area of Convergence Reconstruction — {surface_type}\n"
        f"Centroid: ({cx:.1f}, {cy:.1f}) mm | RMS Residual: {conv['rms_residual_mm']:.2f} mm | Uncertainty Radius: ±{radius:.1f} mm",
        color=THEME_TEXT,
        fontsize=10.5,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Receiving Surface X_s Coordinate (mm)", color=THEME_TEXT, fontsize=9.5)
    ax.set_ylabel("Receiving Surface Y_s Coordinate (mm)", color=THEME_TEXT, fontsize=9.5)
    ax.tick_params(colors=THEME_MUTED, labelsize=8.5)
    for spine in ax.spines.values():
        spine.set_color(THEME_GRID)

    legend = ax.legend(loc="upper right", fontsize=7.5, framealpha=0.88, facecolor=THEME_BG_MAIN, edgecolor=THEME_GRID)
    for text in legend.get_texts():
        text.set_color(THEME_TEXT)

    # Forensic limitation footer
    fig.text(
        0.5,
        0.015,
        "Forensic Note: Planar back-projection assumes linear droplet headings. "
        "Shaded zone represents a 2D Area of Convergence on the target plane, NOT a 3D Area of Origin.",
        ha="center",
        color=THEME_MUTED,
        fontsize=7.5,
        style="italic",
    )

    plt.tight_layout(rect=[0.0, 0.04, 1.0, 1.0])

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=fig.get_facecolor())
    if output_path:
        fig.savefig(output_path, format="png", dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)

    png_bytes = buf.getvalue()
    b64_str = base64.b64encode(png_bytes).decode("ascii")

    return {
        "png_bytes": png_bytes,
        "base64_png": f"data:image/png;base64,{b64_str}",
        "scene": scene,
    }


# ===========================================================================
# 3. 3D Trajectory Reconstruction Scene Visualization
# ===========================================================================


def render_3d_trajectory_plot(
    stains: Sequence[StainTrajectoryInput],
    surface_type: str = "Horizontal Floor",
    incline_deg: Optional[float] = None,
    output_path: Optional[str] = None,
    elev: float = 26.0,
    azim: float = -52.0,
    dpi: int = 140,
) -> Dict[str, Any]:
    """
    Generate a 3D Forensic Trajectory Reconstruction figure containing:
    - Global 3D Laboratory X, Y, Z axes in millimeters
    - Receiving surface plane mesh (Horizontal Floor, Vertical Wall, or Inclined Surface)
    - Measured 3D stain positions on the receiving surface
    - 3D back-projected trajectory rays
    - Tangent-Method Area of Origin estimate + vertical uncertainty range
    - 3D Skew-Ray Closest Point of Approach (CPA) estimate + uncertainty ellipsoid
    """
    scene = build_trajectory_scene_data(
        stains=stains,
        surface_type=surface_type,
        incline_deg=incline_deg,
    )
    tan_orig = scene["tangent_origin_3d"]
    cpa_orig = scene["cpa_origin_3d"]
    conv_2d = scene["convergence_2d"]

    fig = plt.figure(figsize=(9.0, 6.8), dpi=dpi)
    fig.patch.set_facecolor(THEME_BG_MAIN)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(THEME_BG_PANEL)

    # Collect all 3D laboratory points for bounding box
    pts_lab = [
        np.asarray(tan_orig["point_lab_mm"], dtype=np.float64),
        np.asarray(cpa_orig["point_lab_mm"], dtype=np.float64),
        np.asarray(conv_2d["point_lab_mm"], dtype=np.float64),
    ]
    for st in scene["stains"]:
        pts_lab.append(np.asarray(st["ray_start_lab_mm"], dtype=np.float64))
        pts_lab.append(np.asarray(st["ray_end_lab_mm"], dtype=np.float64))

    pts_arr = np.vstack(pts_lab)
    min_xyz = np.min(pts_arr, axis=0)
    max_xyz = np.max(pts_arr, axis=0)
    span = float(max(np.max(max_xyz - min_xyz), 150.0))

    # 1. Draw Receiving Surface Plane Polygon in 3D Lab Coordinates
    surf_xs = [st["surface_x_mm"] for st in scene["stains"]] + [conv_2d["x_surface_mm"]]
    surf_ys = [st["surface_y_mm"] for st in scene["stains"]] + [conv_2d["y_surface_mm"]]
    pad_s = max(span * 0.15, 40.0)
    s_xmin, s_xmax = min(surf_xs) - pad_s, max(surf_xs) + pad_s
    s_ymin, s_ymax = min(surf_ys) - pad_s, max(surf_ys) + pad_s

    basis = scene["surface_basis"]
    corners_surf = [
        [s_xmin, s_ymin, 0.0],
        [s_xmax, s_ymin, 0.0],
        [s_xmax, s_ymax, 0.0],
        [s_xmin, s_ymax, 0.0],
    ]
    corners_lab = [
        surface_to_lab_coordinates(
            c,
            surface_origin_lab_mm=basis["origin_lab_mm"],
            surface_axis_x_lab=basis["axis_x_lab"],
            surface_axis_y_lab=basis["axis_y_lab"],
        )["point_lab_mm"]
        for c in corners_surf
    ]
    surface_poly = Poly3DCollection(
        [corners_lab],
        alpha=0.22,
        facecolor="#1e293b",
        edgecolor=THEME_CYAN,
        linewidths=1.2,
    )
    ax.add_collection3d(surface_poly)

    # 2. Plot Individual Stains and 3D Back-Projected Trajectory Rays
    for st in scene["stains"]:
        p_start = st["ray_start_lab_mm"]
        p_end = st["ray_end_lab_mm"]
        color = st["color"]

        ax.plot(
            [p_start[0], p_end[0]],
            [p_start[1], p_end[1]],
            [p_start[2], p_end[2]],
            color=color,
            linewidth=2.0,
            alpha=0.9,
            label=f"{st['stain_id']} (α={st['angle_degrees']:.1f}°, Z_i={st['tangent_height_z_mm']:.0f} mm)",
        )
        ax.scatter(
            [p_start[0]],
            [p_start[1]],
            [p_start[2]],
            color=color,
            edgecolors="#ffffff",
            s=65,
            depthshade=False,
        )
        ax.text(
            p_start[0],
            p_start[1],
            p_start[2] + span * 0.02,
            f" {st['stain_id']}",
            color=THEME_TEXT,
            fontsize=8,
            fontweight="bold",
        )

    # 3. Plot 2D Convergence Projection on the Receiving Surface + Normal Line to Origin
    c_lab = conv_2d["point_lab_mm"]
    t_lab = tan_orig["point_lab_mm"]
    cpa_pt = cpa_orig["point_lab_mm"]

    ax.plot(
        [c_lab[0], t_lab[0]],
        [c_lab[1], t_lab[1]],
        [c_lab[2], t_lab[2]],
        color=THEME_MUTED,
        linestyle=":",
        linewidth=1.5,
        label="Surface Normal Projection",
    )
    ax.scatter(
        [c_lab[0]],
        [c_lab[1]],
        [c_lab[2]],
        color=THEME_AMBER,
        marker="o",
        s=70,
        label=f"2D Surface Convergence ({conv_2d['x_surface_mm']:.0f}, {conv_2d['y_surface_mm']:.0f}) mm",
    )

    # 4. Plot Tangent-Method 3D Area of Origin & Vertical Uncertainty Bar
    z_unc = max(tan_orig["z_combined_uncertainty_mm"], 5.0)
    low_surf = [tan_orig["x_surface_mm"], tan_orig["y_surface_mm"], max(0.0, tan_orig["z_mean_surface_mm"] - z_unc)]
    high_surf = [tan_orig["x_surface_mm"], tan_orig["y_surface_mm"], tan_orig["z_mean_surface_mm"] + z_unc]
    low_lab = surface_to_lab_coordinates(
        low_surf,
        surface_origin_lab_mm=basis["origin_lab_mm"],
        surface_axis_x_lab=basis["axis_x_lab"],
        surface_axis_y_lab=basis["axis_y_lab"],
    )["point_lab_mm"]
    high_lab = surface_to_lab_coordinates(
        high_surf,
        surface_origin_lab_mm=basis["origin_lab_mm"],
        surface_axis_x_lab=basis["axis_x_lab"],
        surface_axis_y_lab=basis["axis_y_lab"],
    )["point_lab_mm"]

    ax.plot(
        [low_lab[0], high_lab[0]],
        [low_lab[1], high_lab[1]],
        [low_lab[2], high_lab[2]],
        color=THEME_CRIMSON,
        linewidth=3.2,
        alpha=0.85,
        label=f"Origin Z Uncertainty (±{z_unc:.1f} mm)",
    )
    ax.scatter(
        [t_lab[0]],
        [t_lab[1]],
        [t_lab[2]],
        color=THEME_CRIMSON,
        edgecolors="#ffffff",
        marker="^",
        s=130,
        depthshade=False,
        label=f"Tangent Area of Origin (Z={tan_orig['z_mean_surface_mm']:.1f} mm)",
    )

    # 5. Plot 3D Skew-Ray CPA Origin Estimate + Wireframe Uncertainty Sphere
    ax.scatter(
        [cpa_pt[0]],
        [cpa_pt[1]],
        [cpa_pt[2]],
        color=THEME_EMERALD,
        edgecolors="#ffffff",
        marker="D",
        s=115,
        depthshade=False,
        label=f"3D Skew-Ray CPA ({cpa_pt[0]:.0f}, {cpa_pt[1]:.0f}, {cpa_pt[2]:.0f}) mm",
    )

    sphere_r = max(conv_2d["convergence_radius_mm"], cpa_orig["rms_3d_miss_mm"], span * 0.035)
    u_sph = np.linspace(0, 2 * np.pi, 14)
    v_sph = np.linspace(0, np.pi, 10)
    xs_sph = cpa_pt[0] + sphere_r * np.outer(np.cos(u_sph), np.sin(v_sph))
    ys_sph = cpa_pt[1] + sphere_r * np.outer(np.sin(u_sph), np.sin(v_sph))
    zs_sph = cpa_pt[2] + sphere_r * np.outer(np.ones_like(u_sph), np.cos(v_sph))
    ax.plot_wireframe(xs_sph, ys_sph, zs_sph, color=THEME_CRIMSON, alpha=0.28, linewidth=0.7)

    # 3D Axes & Pane Formatting
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("Lab X (mm)", color=THEME_TEXT, fontsize=9, labelpad=6)
    ax.set_ylabel("Lab Y (mm)", color=THEME_TEXT, fontsize=9, labelpad=6)
    ax.set_zlabel("Lab Z Height (mm)", color=THEME_TEXT, fontsize=9, labelpad=6)
    ax.tick_params(colors=THEME_MUTED, labelsize=8)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((19 / 255, 27 / 255, 46 / 255, 0.65))
        axis.pane.set_edgecolor(THEME_GRID)

    ax.set_title(
        f"3D Spatial Trajectory Reconstruction — {basis['surface_type']}\n"
        f"Tangent Origin Z: {tan_orig['z_mean_surface_mm']:.1f} ± {z_unc:.1f} mm | "
        f"3D CPA: ({cpa_pt[0]:.1f}, {cpa_pt[1]:.1f}, {cpa_pt[2]:.1f}) mm (RMS Miss: {cpa_orig['rms_3d_miss_mm']:.2f} mm)",
        color=THEME_TEXT,
        fontsize=10,
        fontweight="bold",
        pad=10,
    )

    legend = ax.legend(loc="upper left", fontsize=7.2, framealpha=0.88, facecolor=THEME_BG_MAIN, edgecolor=THEME_GRID)
    for text in legend.get_texts():
        text.set_color(THEME_TEXT)

    fig.text(
        0.5,
        0.015,
        "Scientific Limitation: Straight-line trigonometric rays neglect gravity and aerodynamic drag "
        "(systematically overestimating Z). Represents an approximate 3D Area of Origin.",
        ha="center",
        color=THEME_MUTED,
        fontsize=7.5,
        style="italic",
    )

    plt.tight_layout(rect=[0.0, 0.04, 1.0, 1.0])

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=fig.get_facecolor())
    if output_path:
        fig.savefig(output_path, format="png", dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)

    png_bytes = buf.getvalue()
    b64_str = base64.b64encode(png_bytes).decode("ascii")

    return {
        "png_bytes": png_bytes,
        "base64_png": f"data:image/png;base64,{b64_str}",
        "scene": scene,
    }

