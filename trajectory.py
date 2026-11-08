"""
trajectory.py
=============
3D Trajectory Reconstruction, Spatial Coordinate Transformation, and Multi-Stain
Convergence Engine for Bloodstain Pattern Analysis (Phase 4.1).

Scientific & Geometric Scope:
-----------------------------
1. Impact Angle (alpha):
   Delegated strictly to calculator.calculate_impact_angle(width, length):
       alpha = arcsin(W / L),  0 deg < alpha <= 90 deg

2. Distinct Directional Quantities:
   - Angle of Impact (alpha): Acute angle between the droplet path and the target surface plane.
   - Direction of Travel / Surface Azimuth (gamma in [0, 360) deg): Directed 2D heading of the
     droplet across the receiving surface, established from physical tail/spine morphology.
     *CRITICAL*: An undirected ellipse alone has a 180-degree directional ambiguity. This module
     never fabricates directional sense when gamma is omitted.
   - Reverse Source Bearing (beta = (gamma + 180) mod 360 deg): Back-projected planar direction
     pointing from the bloodstain back toward the area of convergence.
   - Trajectory Inclination Components:
     Horizontal (in-plane) component = cos(alpha)
     Vertical (surface-normal) component = sin(alpha)
     Tangent height relation: Z = D * tan(alpha)

3. Area of Convergence (2D) & Area of Origin (3D):
   - Computes least-squares intersections of multiple back-projected rays using NumPy and SciPy.
   - Reports spatial estimates with RMS miss distances, standard deviations, and bounding radii.
   - Never claims a mathematical intersection is a definitive point of origin; straight-line
     trigonometric stringing neglects gravity and aerodynamic drag, systematically overestimating Z.
"""

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import linalg as sp_linalg
from scipy import optimize as sp_optimize

from calculator import (
    InvalidMeasurementError,
    calculate_impact_angle,
    validate_measurements,
)

FORENSIC_TRAJECTORY_DISCLAIMER = (
    "Forensic & Scientific Limitation: Straight-line trigonometric trajectory projection "
    "('stringing') assumes linear flight and neglects gravity and aerodynamic drag, which "
    "systematically overestimates source height (Z). Calculated intersections represent "
    "approximate 2D Areas of Convergence and 3D Areas of Origin with uncertainty bounds, "
    "NEVER a definitive point of origin."
)

PARALLEL_DETERMINANT_TOL = 1e-4
PERPENDICULAR_ANGLE_TOL_DEG = 89.9


class TrajectoryError(ValueError):
    """Exception raised when trajectory inputs or geometric configurations are invalid."""
    pass


@dataclass(frozen=True)
class StainTrajectoryInput:
    """
    Validated input representation of a single bloodstain for trajectory reconstruction.

    Attributes:
        stain_id: Identifier label for the stain (e.g., 'Stain-01').
        width: Minor axis (W) of the ellipse (> 0, W <= L).
        length: Major axis (L) of the ellipse (> 0).
        surface_x_mm: Calibrated X coordinate of the stain center on the target surface (mm).
        surface_y_mm: Calibrated Y coordinate of the stain center on the target surface (mm).
        gamma_deg: Resolved 2D direction of droplet travel on the surface in degrees [0, 360),
                   measured counter-clockwise from +X_surface. Must be explicitly provided for
                   directional ray projection.
        sigma_width: Measurement uncertainty (1 standard deviation) in width (same units as width).
        sigma_length: Measurement uncertainty (1 standard deviation) in length (same units as length).
        sigma_gamma_deg: Directional azimuth uncertainty in degrees.
        is_calibrated: Whether spatial coordinates are calibrated in physical millimeters.
    """
    stain_id: str
    width: float
    length: float
    surface_x_mm: float
    surface_y_mm: float
    gamma_deg: Optional[float] = None
    sigma_width: float = 0.1
    sigma_length: float = 0.1
    sigma_gamma_deg: float = 2.0
    is_calibrated: bool = True


# ===========================================================================
# 1. Input Validation Helpers
# ===========================================================================


def _validate_finite_float(value: Any, name: str) -> float:
    """Validate that a parameter is a finite real number and return it as float."""
    if value is None or isinstance(value, bool):
        raise TrajectoryError(f"Parameter '{name}' cannot be None or boolean.")
    if isinstance(value, str) and not value.strip():
        raise TrajectoryError(f"Parameter '{name}' cannot be blank.")
    try:
        val = float(value)
    except (ValueError, TypeError):
        raise TrajectoryError(f"Parameter '{name}' must be numeric; received '{value}'.")
    if math.isnan(val) or math.isinf(val):
        raise TrajectoryError(f"Parameter '{name}' must be finite (cannot be NaN or infinite).")
    return val


def normalize_angle_360(angle_deg: float) -> float:
    """Normalize an angle in degrees into the canonical interval [0.0, 360.0)."""
    val = _validate_finite_float(angle_deg, "angle_deg")
    res = val % 360.0
    if math.isclose(res, 360.0, abs_tol=1e-12):
        res = 0.0
    return float(res)


# ===========================================================================
# 2. Impact Angle & Trajectory Component Breakdown
# ===========================================================================


def compute_trajectory_components(
    width: Any,
    length: Any,
    planar_distance_mm: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Convert stain width (W) and length (L) into impact angle (alpha) and decompose
    into horizontal (in-plane) and vertical (surface-normal) trajectory components.

    Delegates impact angle calculation directly to calculator.calculate_impact_angle().

    Relationships:
        alpha = arcsin(W / L)
        horizontal_component = cos(alpha) = sqrt(1 - (W/L)^2)
        vertical_component   = sin(alpha) = W / L
        slope_tan            = tan(alpha) = sin(alpha) / cos(alpha)
        height_at_distance   = planar_distance_mm * tan(alpha)  (when alpha < 90 deg)
    """
    try:
        base_calc = calculate_impact_angle(width, length)
    except InvalidMeasurementError as exc:
        raise TrajectoryError(str(exc)) from exc

    w, l = base_calc["width"], base_calc["length"]
    ratio = min(w / l, 1.0)
    alpha_rad = math.asin(ratio)
    alpha_deg = math.degrees(alpha_rad)

    horiz_comp = math.cos(alpha_rad)
    vert_comp = math.sin(alpha_rad)
    is_perp = base_calc["is_perpendicular"] or (alpha_deg >= PERPENDICULAR_ANGLE_TOL_DEG)

    if is_perp or math.isclose(horiz_comp, 0.0, abs_tol=1e-9):
        slope_tan = float("inf")
    else:
        slope_tan = math.tan(alpha_rad)

    estimated_height_mm: Optional[float] = None
    if planar_distance_mm is not None:
        dist = _validate_finite_float(planar_distance_mm, "planar_distance_mm")
        if dist < 0.0:
            raise TrajectoryError(f"planar_distance_mm must be non-negative; received {dist}.")
        if is_perp:
            raise TrajectoryError(
                "Cannot compute finite tangent height Z = D * tan(alpha) for a perpendicular (90°) impact."
            )
        estimated_height_mm = round(float(dist * slope_tan), 4)

    return {
        "width": w,
        "length": l,
        "ratio": base_calc["ratio"],
        "angle_degrees": round(alpha_deg, 4),
        "angle_radians": round(alpha_rad, 6),
        "horizontal_component": round(horiz_comp, 6),
        "vertical_component": round(vert_comp, 6),
        "tangent_slope": round(slope_tan, 6) if math.isfinite(slope_tan) else None,
        "is_perpendicular": bool(is_perp),
        "planar_distance_mm": planar_distance_mm,
        "estimated_height_mm": estimated_height_mm,
        "interpretation": base_calc["interpretation"],
        "disclaimer": FORENSIC_TRAJECTORY_DISCLAIMER,
    }


# ===========================================================================
# 3. 3D Direction Vector Generation & Normalization
# ===========================================================================


def normalize_vector(vec: Union[Sequence[float], np.ndarray]) -> np.ndarray:
    """
    Normalize a 2D or 3D vector to unit Euclidean length (||v||_2 = 1).

    Raises:
        TrajectoryError: If vector contains non-finite values or has zero norm.
    """
    arr = np.asarray(vec, dtype=np.float64)
    if arr.ndim != 1 or arr.size not in (2, 3):
        raise TrajectoryError(f"Vector must be 1D of length 2 or 3; got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise TrajectoryError("Vector components must be finite real numbers.")
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        raise TrajectoryError("Cannot normalize a zero-length vector.")
    return arr / norm


def compute_direction_vector_3d(
    angle_degrees: Any,
    gamma_deg: Any,
    back_project: bool = True,
) -> Dict[str, Any]:
    """
    Compute the 3D unit trajectory vector in the local Right-Handed Surface Coordinate Frame.

    Definitions:
    - Target surface lies in the (X_s, Y_s) plane at Z_s = 0, with surface normal +Z_s.
    - angle_degrees (alpha in (0, 90]): Impact angle between path and surface plane.
    - gamma_deg (gamma in [0, 360)): Forward direction of droplet motion projected onto the
      surface plane (measured counter-clockwise from +X_s).
      *NOTE*: gamma_deg is REQUIRED unless alpha == 90 deg. An ellipse without resolved tail/spine
      directionality cannot establish gamma_deg.
    - back_project:
      * If True (default): Returns the unit vector pointing FROM the stain BACK TOWARD the
        origin/source (azimuth beta = (gamma + 180) mod 360, elevation +alpha above surface).
        u_back = [cos(alpha)*cos(beta), cos(alpha)*sin(beta), +sin(alpha)]
      * If False: Returns the forward impact velocity unit vector of the droplet striking
        the surface (azimuth gamma, downward normal component -sin(alpha)).
        u_fwd  = [cos(alpha)*cos(gamma), cos(alpha)*sin(gamma), -sin(alpha)]
    """
    alpha_deg = _validate_finite_float(angle_degrees, "angle_degrees")
    if alpha_deg <= 0.0 or alpha_deg > 90.0:
        raise TrajectoryError(f"Impact angle must satisfy 0 < alpha <= 90 degrees; received {alpha_deg}°.")

    is_perp = math.isclose(alpha_deg, 90.0, abs_tol=1e-5)

    if gamma_deg is None:
        if not is_perp:
            raise TrajectoryError(
                "Direction of travel (gamma_deg) is required for non-perpendicular stains. "
                "An elliptical stain alone has a 180° directional ambiguity and cannot establish 3D direction without tail/spine orientation."
            )
        gamma_norm = 0.0
    else:
        gamma_norm = normalize_angle_360(gamma_deg)

    beta_norm = normalize_angle_360(gamma_norm + 180.0)
    alpha_rad = math.radians(alpha_deg)

    if is_perp:
        vx, vy = 0.0, 0.0
        vz = 1.0 if back_project else -1.0
        planar_unit = np.array([0.0, 0.0], dtype=np.float64)
    else:
        active_az_deg = beta_norm if back_project else gamma_norm
        az_rad = math.radians(active_az_deg)
        horiz = math.cos(alpha_rad)
        vx = horiz * math.cos(az_rad)
        vy = horiz * math.sin(az_rad)
        vz = math.sin(alpha_rad) if back_project else -math.sin(alpha_rad)
        planar_unit = np.array([math.cos(az_rad), math.sin(az_rad)], dtype=np.float64)

    vec_3d = normalize_vector([vx, vy, vz])

    return {
        "vector_3d": vec_3d,
        "vx": round(float(vec_3d[0]), 6),
        "vy": round(float(vec_3d[1]), 6),
        "vz": round(float(vec_3d[2]), 6),
        "norm": round(float(np.linalg.norm(vec_3d)), 6),
        "planar_unit_vector": planar_unit,
        "angle_degrees": round(alpha_deg, 4),
        "gamma_deg": round(gamma_norm, 4) if not is_perp else None,
        "reverse_azimuth_deg": round(beta_norm, 4) if not is_perp else None,
        "back_projected": bool(back_project),
        "is_perpendicular": is_perp,
    }


# ===========================================================================
# 4. Coordinate System Transformations (Image <-> Surface <-> 3D Lab)
# ===========================================================================


def resolve_opencv_ellipse_azimuth(
    opencv_angle_deg: float,
    invert_y_axis: bool = True,
    flip_sense_180: bool = False,
) -> float:
    """
    Convert an OpenCV fitEllipse orientation angle into a right-handed surface azimuth
    ONLY when the examiner explicitly specifies the directional sense (flip_sense_180).

    OpenCV Convention:
    - Image coordinates have +u right, +v down.
    - OpenCV's box angle is measured clockwise from vertical/horizontal depending on axis ordering.
    - Because an ellipse has 180-degree rotational symmetry, the examiner MUST confirm whether
      the droplet traveled along the base orientation or +180 degrees opposite (flip_sense_180).
    """
    ang = _validate_finite_float(opencv_angle_deg, "opencv_angle_deg")
    # Convert clockwise image angle (+v down) to standard counter-clockwise Cartesian angle (+Y up)
    cartesian_deg = -ang if invert_y_axis else ang
    if flip_sense_180:
        cartesian_deg += 180.0
    return normalize_angle_360(cartesian_deg)


def image_to_surface_coordinates(
    u_px: Any,
    v_px: Any,
    mm_per_pixel: Any,
    origin_u_px: float = 0.0,
    origin_v_px: float = 0.0,
    invert_y: bool = True,
) -> Dict[str, float]:
    """
    Transform 2D image pixel coordinates (u_px, v_px) into physical 2D target surface
    coordinates (X_s, Y_s) in millimeters.

    Parameters:
        u_px: Horizontal pixel column coordinate (+right).
        v_px: Vertical pixel row coordinate (+down in standard image space).
        mm_per_pixel: Positive spatial calibration factor (mm / pixel).
        origin_u_px: Reference origin X in image pixels.
        origin_v_px: Reference origin Y in image pixels (e.g., image height for bottom-left origin).
        invert_y: If True (default), flips the Y-axis so +Y_s points upward in right-handed space:
                  X_s = (u_px - origin_u_px) * mm_per_pixel
                  Y_s = (origin_v_px - v_px) * mm_per_pixel
    """
    u = _validate_finite_float(u_px, "u_px")
    v = _validate_finite_float(v_px, "v_px")
    scale = _validate_finite_float(mm_per_pixel, "mm_per_pixel")
    u0 = _validate_finite_float(origin_u_px, "origin_u_px")
    v0 = _validate_finite_float(origin_v_px, "origin_v_px")

    if scale <= 0.0:
        raise TrajectoryError(f"Scale factor (mm_per_pixel) must be strictly positive; received {scale}.")

    x_mm = (u - u0) * scale
    y_mm = ((v0 - v) * scale) if invert_y else ((v - v0) * scale)

    return {
        "surface_x_mm": round(float(x_mm), 4),
        "surface_y_mm": round(float(y_mm), 4),
        "u_px": u,
        "v_px": v,
        "mm_per_pixel": scale,
    }


def surface_to_image_coordinates(
    surface_x_mm: Any,
    surface_y_mm: Any,
    mm_per_pixel: Any,
    origin_u_px: float = 0.0,
    origin_v_px: float = 0.0,
    invert_y: bool = True,
) -> Dict[str, float]:
    """Inverse transformation from 2D surface coordinates (mm) back to image pixels (u_px, v_px)."""
    x_mm = _validate_finite_float(surface_x_mm, "surface_x_mm")
    y_mm = _validate_finite_float(surface_y_mm, "surface_y_mm")
    scale = _validate_finite_float(mm_per_pixel, "mm_per_pixel")
    u0 = _validate_finite_float(origin_u_px, "origin_u_px")
    v0 = _validate_finite_float(origin_v_px, "origin_v_px")

    if scale <= 0.0:
        raise TrajectoryError(f"Scale factor (mm_per_pixel) must be strictly positive; received {scale}.")

    u_px = u0 + (x_mm / scale)
    v_px = (v0 - (y_mm / scale)) if invert_y else (v0 + (y_mm / scale))

    return {
        "u_px": round(float(u_px), 4),
        "v_px": round(float(v_px), 4),
        "surface_x_mm": x_mm,
        "surface_y_mm": y_mm,
        "mm_per_pixel": scale,
    }


def surface_to_lab_coordinates(
    point_surface_mm: Sequence[float],
    vector_surface: Optional[Sequence[float]] = None,
    surface_origin_lab_mm: Sequence[float] = (0.0, 0.0, 0.0),
    surface_axis_x_lab: Sequence[float] = (1.0, 0.0, 0.0),
    surface_axis_y_lab: Sequence[float] = (0.0, 1.0, 0.0),
) -> Dict[str, Any]:
    """
    Transform a 3D point [X_s, Y_s, Z_s] and optional 3D direction vector from the local
    Target Surface Coordinate System into the Global 3D Laboratory Coordinate System.

    The local surface basis (e_x, e_y, e_z) is constructed from orthonormalized
    surface_axis_x_lab and surface_axis_y_lab, with surface normal e_z = e_x x e_y.

    Examples of target surfaces:
    - Horizontal Floor (default): e_x = [1,0,0], e_y = [0,1,0] -> normal e_z = [0,0,1] (upward)
    - Vertical North Wall:        e_x = [1,0,0], e_y = [0,0,1] -> normal e_z = [0,-1,0] (into room)
    """
    pt_s = np.asarray(point_surface_mm, dtype=np.float64)
    if pt_s.shape != (3,) or not np.all(np.isfinite(pt_s)):
        raise TrajectoryError("point_surface_mm must be a finite 3-element sequence [X_s, Y_s, Z_s].")

    p0 = np.asarray(surface_origin_lab_mm, dtype=np.float64)
    if p0.shape != (3,) or not np.all(np.isfinite(p0)):
        raise TrajectoryError("surface_origin_lab_mm must be a finite 3-element sequence [X_0, Y_0, Z_0].")

    ex = normalize_vector(surface_axis_x_lab)
    ey_raw = normalize_vector(surface_axis_y_lab)
    if ex.size != 3 or ey_raw.size != 3:
        raise TrajectoryError("Surface basis axes must be 3D vectors.")

    # Gram-Schmidt orthonormalization to ensure a valid right-handed rotation matrix R in SO(3)
    ez_cross = np.cross(ex, ey_raw)
    if np.linalg.norm(ez_cross) <= 1e-8:
        raise TrajectoryError("surface_axis_x_lab and surface_axis_y_lab cannot be collinear.")
    ez = ez_cross / np.linalg.norm(ez_cross)
    ey = np.cross(ez, ex)

    # Columns of R are [ex, ey, ez]
    rot_matrix = np.column_stack((ex, ey, ez))
    pt_lab = p0 + rot_matrix @ pt_s

    vec_lab = None
    if vector_surface is not None:
        v_s = normalize_vector(vector_surface)
        if v_s.size != 3:
            raise TrajectoryError("vector_surface must be a 3D vector.")
        vec_lab = normalize_vector(rot_matrix @ v_s)

    return {
        "point_lab_mm": pt_lab,
        "vector_lab": vec_lab,
        "rotation_matrix": rot_matrix,
        "surface_normal_lab": ez,
    }


# ===========================================================================
# 5. Uncertainty Propagation
# ===========================================================================


def propagate_uncertainty(
    width: Any,
    length: Any,
    sigma_width: float = 0.1,
    sigma_length: float = 0.1,
    planar_distance_mm: Optional[float] = None,
    sigma_distance_mm: float = 1.0,
) -> Dict[str, Any]:
    """
    Perform first-order Taylor series uncertainty propagation for impact angle
    alpha = arcsin(W / L) and optional origin height Z = D * tan(alpha).

    Partial Derivatives:
        d(alpha)/dW =  1 / sqrt(L^2 - W^2)
        d(alpha)/dL = -W / (L * sqrt(L^2 - W^2))
        sigma_alpha_rad = sqrt((d(alpha)/dW * sigma_W)^2 + (d(alpha)/dL * sigma_L)^2)

    Near W -> L (alpha -> 90 deg), the analytical derivative of arcsin has a square-root
    singularity; in that regime, a bounded finite-difference perturbation fallback is used
    and a high-sensitivity warning is emitted.
    """
    try:
        w, l = validate_measurements(width, length)
    except InvalidMeasurementError as exc:
        raise TrajectoryError(str(exc)) from exc

    sw = _validate_finite_float(sigma_width, "sigma_width")
    sl = _validate_finite_float(sigma_length, "sigma_length")
    if sw < 0.0 or sl < 0.0:
        raise TrajectoryError("Measurement uncertainties (sigma_width, sigma_length) must be non-negative.")

    ratio = min(w / l, 1.0)
    alpha_rad = math.asin(ratio)
    alpha_deg = math.degrees(alpha_rad)

    denom_sq = max(l * l - w * w, 0.0)
    near_singular = denom_sq <= (1e-4 * l * l)

    if not near_singular:
        denom = math.sqrt(denom_sq)
        dalpha_dw = 1.0 / denom
        dalpha_dl = -w / (l * denom)
        sigma_alpha_rad = math.sqrt((dalpha_dw * sw) ** 2 + (dalpha_dl * sl) ** 2)
    else:
        # Bounded one-sided perturbation near W == L
        w_low = max(w - sw, 1e-6)
        l_high = l + sl
        r_perturbed = min(w_low / l_high, 1.0)
        sigma_alpha_rad = abs(alpha_rad - math.asin(r_perturbed))

    sigma_alpha_deg = math.degrees(sigma_alpha_rad)

    height_mm = None
    sigma_height_mm = None
    if planar_distance_mm is not None:
        dist = _validate_finite_float(planar_distance_mm, "planar_distance_mm")
        s_dist = _validate_finite_float(sigma_distance_mm, "sigma_distance_mm")
        if dist < 0.0 or s_dist < 0.0:
            raise TrajectoryError("planar_distance_mm and sigma_distance_mm must be non-negative.")
        if alpha_deg < PERPENDICULAR_ANGLE_TOL_DEG:
            tan_a = math.tan(alpha_rad)
            sec2_a = 1.0 / (math.cos(alpha_rad) ** 2)
            height_mm = dist * tan_a
            sigma_height_mm = math.sqrt((tan_a * s_dist) ** 2 + (dist * sec2_a * sigma_alpha_rad) ** 2)

    return {
        "angle_degrees": round(alpha_deg, 4),
        "sigma_angle_degrees": round(sigma_alpha_deg, 4),
        "sigma_angle_radians": round(sigma_alpha_rad, 6),
        "angle_ci95_degrees": (
            round(max(0.0, alpha_deg - 1.96 * sigma_alpha_deg), 2),
            round(min(90.0, alpha_deg + 1.96 * sigma_alpha_deg), 2),
        ),
        "near_perpendicular_singularity": near_singular,
        "estimated_height_mm": round(height_mm, 4) if height_mm is not None else None,
        "sigma_height_mm": round(sigma_height_mm, 4) if sigma_height_mm is not None else None,
    }


# ===========================================================================
# 6. Single-Stain Trajectory Reconstruction
# ===========================================================================


def reconstruct_single_stain_trajectory(stain: StainTrajectoryInput) -> Dict[str, Any]:
    """
    Reconstruct the complete 2D and 3D back-projected trajectory geometry and uncertainty
    profile for an individual bloodstain on a planar receiving surface.
    """
    if not isinstance(stain, StainTrajectoryInput):
        raise TrajectoryError("Input must be a StainTrajectoryInput instance.")

    sx = _validate_finite_float(stain.surface_x_mm, "surface_x_mm")
    sy = _validate_finite_float(stain.surface_y_mm, "surface_y_mm")

    comp = compute_trajectory_components(stain.width, stain.length)
    vec_info = compute_direction_vector_3d(
        angle_degrees=comp["angle_degrees"],
        gamma_deg=stain.gamma_deg,
        back_project=True,
    )
    uncert = propagate_uncertainty(
        width=stain.width,
        length=stain.length,
        sigma_width=stain.sigma_width,
        sigma_length=stain.sigma_length,
    )

    origin_3d = np.array([sx, sy, 0.0], dtype=np.float64)

    return {
        "stain_id": stain.stain_id,
        "origin_surface_mm": origin_3d,
        "surface_x_mm": sx,
        "surface_y_mm": sy,
        "is_calibrated": bool(stain.is_calibrated),
        "components": comp,
        "direction": vec_info,
        "uncertainty": uncert,
        "disclaimer": FORENSIC_TRAJECTORY_DISCLAIMER,
    }


# ===========================================================================
# 7. 2D Area of Convergence (Pairwise & Multi-Ray Least Squares)
# ===========================================================================


def intersect_two_2d_rays(
    p1: Sequence[float],
    u1: Sequence[float],
    p2: Sequence[float],
    u2: Sequence[float],
    require_forward_intersection: bool = True,
) -> Dict[str, Any]:
    """
    Compute the exact 2D intersection of two planar rays:
        r1(t1) = p1 + t1 * u1
        r2(t2) = p2 + t2 * u2

    Raises:
        TrajectoryError: If rays are parallel/collinear or diverge (t1 < 0 or t2 < 0
                         when require_forward_intersection is True).
    """
    pt1 = np.asarray(p1, dtype=np.float64)
    pt2 = np.asarray(p2, dtype=np.float64)
    dir1 = normalize_vector(u1)
    dir2 = normalize_vector(u2)

    if pt1.shape != (2,) or pt2.shape != (2,) or dir1.size != 2 or dir2.size != 2:
        raise TrajectoryError("2D ray intersection requires 2D points and 2D direction vectors.")

    # Solve [dir1, -dir2] * [t1, t2]^T = pt2 - pt1
    mat = np.column_stack((dir1, -dir2))
    det = float(np.linalg.det(mat))

    if abs(det) < PARALLEL_DETERMINANT_TOL:
        raise TrajectoryError(
            "Trajectories are parallel or nearly collinear (|sin(delta_theta)| < 1e-4); "
            "a unique 2D intersection cannot be established."
        )

    rhs = pt2 - pt1
    params = sp_linalg.solve(mat, rhs)
    t1, t2 = float(params[0]), float(params[1])

    if require_forward_intersection and (t1 < -1e-6 or t2 < -1e-6):
        raise TrajectoryError(
            f"Trajectories diverge in the forward back-projection sense (ray parameters t1={t1:.2f}, t2={t2:.2f}). "
            "Check directional sense (gamma_deg) of the stains."
        )

    intersection = pt1 + t1 * dir1
    return {
        "convergence_x_mm": round(float(intersection[0]), 4),
        "convergence_y_mm": round(float(intersection[1]), 4),
        "point_2d": intersection,
        "t1_mm": round(t1, 4),
        "t2_mm": round(t2, 4),
        "intersection_angle_deg": round(math.degrees(math.acos(np.clip(float(np.dot(dir1, dir2)), -1.0, 1.0))), 2),
    }


def estimate_area_of_convergence_2d(
    stains: Sequence[StainTrajectoryInput],
    require_calibrated: bool = True,
    require_forward_rays: bool = True,
) -> Dict[str, Any]:
    """
    Estimate the 2D Area of Convergence on the receiving surface from N >= 2 directional
    bloodstains using Orthogonal Distance Least-Squares minimization:

        min_{c in R^2} sum_{i=1}^N || (I_2 - u_i u_i^T) (c - p_i) ||_2^2

    Scientific Safeguards:
    - Requires at least 2 non-perpendicular stains with explicitly supplied gamma_deg.
    - Rejects parallel/collinear ray bundles via singular value condition check.
    - Verifies that the estimated convergence point lies in the back-projected half-space
      of the stains when require_forward_rays=True.
    - Returns RMS residual miss distance, per-stain perpendicular residuals, and a bounding
      convergence radius defining an Area of Convergence.
    """
    if not stains or len(stains) < 2:
        raise TrajectoryError("At least 2 directional bloodstains are required to estimate an Area of Convergence.")

    reconstructed: List[Dict[str, Any]] = []
    for st in stains:
        if require_calibrated and not st.is_calibrated:
            raise TrajectoryError(
                f"Stain '{st.stain_id}' is uncalibrated. Multi-stain spatial convergence requires calibrated millimeter coordinates."
            )
        rec = reconstruct_single_stain_trajectory(st)
        if rec["components"]["is_perpendicular"]:
            raise TrajectoryError(
                f"Stain '{st.stain_id}' has a 90° perpendicular impact angle and lacks a defined 2D surface azimuth for planar convergence."
            )
        reconstructed.append(rec)

    # Assemble linear least-squares normal equations: A * c = b
    a_mat = np.zeros((2, 2), dtype=np.float64)
    b_vec = np.zeros(2, dtype=np.float64)
    identity_2 = np.eye(2, dtype=np.float64)

    points_2d: List[np.ndarray] = []
    dirs_2d: List[np.ndarray] = []

    for rec in reconstructed:
        p_i = np.array([rec["surface_x_mm"], rec["surface_y_mm"]], dtype=np.float64)
        u_i = rec["direction"]["planar_unit_vector"]
        proj_perp = identity_2 - np.outer(u_i, u_i)
        a_mat += proj_perp
        b_vec += proj_perp @ p_i
        points_2d.append(p_i)
        dirs_2d.append(u_i)

    # Check conditioning via eigenvalues / singular values of A
    singular_vals = sp_linalg.svdvals(a_mat)
    if singular_vals[-1] < PARALLEL_DETERMINANT_TOL:
        raise TrajectoryError(
            "Stain trajectories are parallel or nearly collinear; normal matrix is rank-deficient "
            f"(smallest singular value = {singular_vals[-1]:.2e})."
        )

    c_est = sp_linalg.solve(a_mat, b_vec, assume_a="sym")

    # Compute per-stain ray parameters t_i and perpendicular miss distances
    ray_distances: List[float] = []
    perp_residuals: List[float] = []
    backward_stains: List[str] = []

    for idx, rec in enumerate(reconstructed):
        p_i = points_2d[idx]
        u_i = dirs_2d[idx]
        diff = c_est - p_i
        t_i = float(np.dot(diff, u_i))
        if t_i < -1e-4:
            backward_stains.append(rec["stain_id"])
        perp_vec = diff - t_i * u_i
        miss_dist = float(np.linalg.norm(perp_vec))
        ray_distances.append(max(t_i, 0.0))
        perp_residuals.append(miss_dist)

    if require_forward_rays and backward_stains:
        raise TrajectoryError(
            f"Estimated convergence lies behind the back-projection direction for stain(s): {', '.join(backward_stains)}. "
            "Trajectories diverge."
        )

    rms_residual = float(np.sqrt(np.mean(np.square(perp_residuals))))
    max_residual = float(np.max(perp_residuals))

    # Propagate azimuth uncertainty (sigma_gamma_deg) into a minimum physical radius of uncertainty
    angular_radii = []
    for idx, st in enumerate(stains):
        d_i = float(np.linalg.norm(c_est - points_2d[idx]))
        ang_rad = math.radians(max(st.sigma_gamma_deg, 0.5))
        angular_radii.append(d_i * math.tan(ang_rad))
    convergence_radius_mm = max(rms_residual, float(np.mean(angular_radii)))

    return {
        "convergence_x_mm": round(float(c_est[0]), 4),
        "convergence_y_mm": round(float(c_est[1]), 4),
        "point_2d": c_est,
        "stain_count": len(reconstructed),
        "rms_residual_mm": round(rms_residual, 4),
        "max_residual_mm": round(max_residual, 4),
        "convergence_radius_mm": round(convergence_radius_mm, 4),
        "per_stain_residuals_mm": [round(r, 4) for r in perp_residuals],
        "per_stain_planar_distances_mm": [round(float(np.linalg.norm(c_est - p)), 4) for p in points_2d],
        "condition_number": round(float(singular_vals[0] / singular_vals[-1]), 4),
        "disclaimer": FORENSIC_TRAJECTORY_DISCLAIMER,
    }


# ===========================================================================
# 8. 3D Area of Origin Estimation (Tangent Method + 3D Skew-Ray CPA)
# ===========================================================================


def estimate_area_of_origin_3d(
    stains: Sequence[StainTrajectoryInput],
    require_calibrated: bool = True,
    require_forward_rays: bool = True,
) -> Dict[str, Any]:
    """
    Estimate the 3D Area of Origin (X, Y, Z) above the target surface from N >= 2
    directional bloodstains using two complementary forensic methods:

    1. Classical BPA Tangent Method:
       Uses the 2D Area of Convergence (X_c, Y_c) to compute planar distance D_i for each
       stain and derives individual height estimates Z_i = D_i * tan(alpha_i), reporting
       mean Z, standard deviation, median, and [Z_min, Z_max] bounds.

    2. 3D Skew-Ray Least-Squares Closest Point of Approach (CPA):
       Finds the 3D point P* = [X*, Y*, Z*]^T minimizing the sum of squared perpendicular
       distances to all N 3D back-projected rays in R^3, refined with bounded SciPy
       least-squares optimization (Z >= 0).
    """
    conv_2d = estimate_area_of_convergence_2d(
        stains=stains,
        require_calibrated=require_calibrated,
        require_forward_rays=require_forward_rays,
    )
    c_2d = conv_2d["point_2d"]

    reconstructed = [reconstruct_single_stain_trajectory(st) for st in stains]

    # 1. Classical Tangent Method Heights: Z_i = D_i * tan(alpha_i)
    z_estimates: List[float] = []
    z_sigmas: List[float] = []
    per_stain_summary: List[Dict[str, Any]] = []

    a_3d = np.zeros((3, 3), dtype=np.float64)
    b_3d = np.zeros(3, dtype=np.float64)
    identity_3 = np.eye(3, dtype=np.float64)

    origins_3d: List[np.ndarray] = []
    dirs_3d: List[np.ndarray] = []

    for idx, rec in enumerate(reconstructed):
        st = stains[idx]
        p_2d = np.array([rec["surface_x_mm"], rec["surface_y_mm"]], dtype=np.float64)
        dist_i = float(np.linalg.norm(c_2d - p_2d))
        alpha_rad = rec["components"]["angle_radians"]
        z_i = dist_i * math.tan(alpha_rad)
        z_estimates.append(z_i)

        unc_i = propagate_uncertainty(
            width=st.width,
            length=st.length,
            sigma_width=st.sigma_width,
            sigma_length=st.sigma_length,
            planar_distance_mm=dist_i,
            sigma_distance_mm=max(conv_2d["convergence_radius_mm"], 1.0),
        )
        sigma_z_i = unc_i["sigma_height_mm"] or 0.0
        z_sigmas.append(sigma_z_i)

        per_stain_summary.append({
            "stain_id": rec["stain_id"],
            "angle_degrees": rec["components"]["angle_degrees"],
            "gamma_deg": rec["direction"]["gamma_deg"],
            "reverse_azimuth_deg": rec["direction"]["reverse_azimuth_deg"],
            "planar_distance_mm": round(dist_i, 4),
            "height_z_mm": round(z_i, 4),
            "sigma_z_mm": round(sigma_z_i, 4),
        })

        # Accumulate 3D skew-ray normal equations
        p3 = rec["origin_surface_mm"]
        d3 = rec["direction"]["vector_3d"]
        proj3 = identity_3 - np.outer(d3, d3)
        a_3d += proj3
        b_3d += proj3 @ p3
        origins_3d.append(p3)
        dirs_3d.append(d3)

    z_arr = np.asarray(z_estimates, dtype=np.float64)
    z_mean = float(np.mean(z_arr))
    z_std = float(np.std(z_arr, ddof=1)) if len(z_arr) > 1 else 0.0
    z_median = float(np.median(z_arr))
    z_min = float(np.min(z_arr))
    z_max = float(np.max(z_arr))

    # Combined vertical uncertainty (scatter across stains + propagated measurement uncertainty)
    mean_prop_sigma_z = float(np.sqrt(np.mean(np.square(z_sigmas))))
    combined_sigma_z = float(math.sqrt(z_std ** 2 + mean_prop_sigma_z ** 2))

    # 2. 3D Skew-Ray Closest Point of Approach (CPA) using SciPy
    svals_3d = sp_linalg.svdvals(a_3d)
    if svals_3d[-1] >= PARALLEL_DETERMINANT_TOL:
        cpa_init = sp_linalg.solve(a_3d, b_3d, assume_a="sym")
    else:
        cpa_init = np.array([c_2d[0], c_2d[1], z_mean], dtype=np.float64)

    def _skew_ray_residuals(pt: np.ndarray) -> np.ndarray:
        res_list = []
        for p_i, d_i in zip(origins_3d, dirs_3d):
            diff = pt - p_i
            t_proj = np.dot(diff, d_i)
            perp = diff - t_proj * d_i
            res_list.extend(perp.tolist())
        return np.asarray(res_list, dtype=np.float64)

    opt_res = sp_optimize.least_squares(
        _skew_ray_residuals,
        x0=np.array([cpa_init[0], cpa_init[1], max(cpa_init[2], 0.0)], dtype=np.float64),
        bounds=([-np.inf, -np.inf, 0.0], [np.inf, np.inf, np.inf]),
    )
    cpa_3d = opt_res.x

    # Compute 3D RMS miss distance at CPA
    miss_3d = []
    for p_i, d_i in zip(origins_3d, dirs_3d):
        diff = cpa_3d - p_i
        perp = diff - np.dot(diff, d_i) * d_i
        miss_3d.append(float(np.linalg.norm(perp)))
    rms_3d_mm = float(np.sqrt(np.mean(np.square(miss_3d))))

    return {
        "convergence_2d": conv_2d,
        "tangent_method_origin": {
            "x_mm": conv_2d["convergence_x_mm"],
            "y_mm": conv_2d["convergence_y_mm"],
            "z_mean_mm": round(z_mean, 4),
            "z_median_mm": round(z_median, 4),
            "z_std_mm": round(z_std, 4),
            "z_combined_uncertainty_mm": round(combined_sigma_z, 4),
            "z_range_mm": (round(z_min, 4), round(z_max, 4)),
        },
        "skew_ray_cpa_origin": {
            "x_mm": round(float(cpa_3d[0]), 4),
            "y_mm": round(float(cpa_3d[1]), 4),
            "z_mm": round(float(cpa_3d[2]), 4),
            "point_3d": cpa_3d,
            "rms_3d_miss_mm": round(rms_3d_mm, 4),
        },
        "stain_count": len(reconstructed),
        "per_stain_results": per_stain_summary,
        "disclaimer": FORENSIC_TRAJECTORY_DISCLAIMER,
    }

