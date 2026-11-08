"""
trajectory_integration.py
=========================
Integration and Scientific Validation Layer connecting Optical/Manual Stain Measurements
(SQLite + image_processing.py) to the Trajectory Reconstruction Engine (trajectory.py)
and 3D Visualization Suite (trajectory_visualization.py) (Phase 4.3).

Responsibilities:
1. Retrieve saved measurements, optical calibration metadata, surface positions, and
   examiner-supplied directional angles.
2. Enforce strict scientific eligibility before reconstruction:
   - Reject uncalibrated measurements (preventing pixel dimensions from being silently
     interpreted as physical millimeters).
   - Require explicit examiner-supplied direction of travel (0 <= gamma < 360 deg); never
     invent directionality from an undirected ellipse.
   - Reject 90° perpendicular stains from planar 2D ray intersection.
   - Require explicit receiving-surface coordinates (X_s, Y_s in mm).
   - Never silently discard problematic stains in a reconstruction workspace.
3. Pass validated StainTrajectoryInput datasets to trajectory.py / trajectory_visualization.py
   and persist structured results in SQLite.
"""

import json
import math
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

import database
from trajectory import (
    FORENSIC_TRAJECTORY_DISCLAIMER,
    StainTrajectoryInput,
    TrajectoryError,
    normalize_angle_360,
    propagate_uncertainty,
)
import trajectory_visualization


class ReconstructionValidationError(ValueError):
    """Exception raised when a reconstruction workspace or stain fails forensic validation."""
    pass


# ===========================================================================
# 1. Calibration, Directionality & Spatial Eligibility Validation
# ===========================================================================


def is_measurement_calibrated(record: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Verify whether a measurement record has a verified physical millimeter calibration.
    Prevents uncalibrated pixel dimensions from being silently treated as millimeters.
    """
    cal_status = str(record.get("calibration_status") or "").strip()
    img_id = record.get("image_id")
    img_is_cal = record.get("image_is_calibrated")
    img_scale = record.get("image_scale_factor")
    meas_scale = record.get("measurement_scale_factor") or record.get("scale_factor")

    if "uncalibrated" in cal_status.lower():
        return (
            False,
            f"Uncalibrated measurement ('{cal_status}'). Physical trajectory reconstruction requires millimeter calibration.",
        )

    if img_id:
        if img_is_calibrated_flag := (img_is_cal == 1):
            scale_val = float(img_scale or meas_scale or 0.0)
            if scale_val > 0.0:
                return True, f"Calibrated Optical Scale ({scale_val:.4f} mm/px)"
        if cal_status.lower().startswith("calibrated"):
            return True, cal_status
        return (
            False,
            f"Optical image '{img_id}' is not spatially calibrated in millimeters.",
        )

    # Manual measurement without an image: must be explicitly marked Calibrated (mm)
    if cal_status.lower().startswith("calibrated"):
        return True, cal_status

    return (
        False,
        "Measurement lacks verified millimeter calibration status.",
    )


def validate_direction_angle_input(direction_angle: Any) -> float:
    """
    Validate an examiner-supplied direction of travel angle gamma.
    Must be a finite number strictly in the interval [0.0, 360.0).
    """
    if direction_angle is None or (isinstance(direction_angle, str) and not direction_angle.strip()):
        raise ReconstructionValidationError(
            "Direction of travel (direction_angle) cannot be empty. "
            "An elliptical stain alone has a 180° ambiguity; direction must be supplied from tail/spine morphology."
        )
    try:
        val = float(direction_angle)
    except (ValueError, TypeError):
        raise ReconstructionValidationError(f"Invalid non-numeric direction angle: '{direction_angle}'.")

    if not math.isfinite(val):
        raise ReconstructionValidationError("Direction of travel angle must be a finite number.")

    if val < 0.0 or val >= 360.0:
        raise ReconstructionValidationError(
            f"Direction of travel angle must satisfy 0° <= gamma < 360°; received {val}°."
        )
    return float(val)


def evaluate_stain_eligibility(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Inspect a reconstruction stain or measurement record and return its complete
    calibration, directionality, coordinate, and uncertainty validation state.
    """
    issues: List[str] = []

    # 1. Calibration check
    is_cal, cal_reason = is_measurement_calibrated(record)
    if not is_cal:
        issues.append(cal_reason)

    # 2. Perpendicular / Impact angle check
    angle_deg = float(record.get("angle_degrees") or 0.0)
    is_perp = bool(record.get("is_perpendicular")) or (angle_deg >= 89.9)
    if is_perp:
        issues.append(
            f"Perpendicular impact ({angle_deg:.1f}°) has no horizontal surface direction vector for 2D convergence."
        )

    # 3. Direction of travel check (0 <= gamma < 360)
    raw_dir = record.get("direction_angle")
    has_valid_dir = False
    gamma_deg: Optional[float] = None
    reverse_bearing_deg: Optional[float] = None
    dir_reason = "Missing Direction of Travel"

    if raw_dir is not None and str(raw_dir).strip() != "":
        try:
            gamma_deg = validate_direction_angle_input(raw_dir)
            reverse_bearing_deg = normalize_angle_360(gamma_deg + 180.0)
            has_valid_dir = True
            dir_reason = f"γ = {gamma_deg:.1f}° (Back-Ray β = {reverse_bearing_deg:.1f}°)"
        except ReconstructionValidationError as err:
            issues.append(str(err))
            dir_reason = str(err)
    else:
        issues.append("Missing examiner-supplied direction of travel (0° <= γ < 360°).")

    # 4. Receiving surface coordinates (X_s, Y_s in mm)
    raw_x = record.get("surface_x_mm")
    raw_y = record.get("surface_y_mm")
    has_coords = False
    sx_mm: Optional[float] = None
    sy_mm: Optional[float] = None
    coords_reason = "Missing Surface Coordinates (X_s, Y_s)"

    if raw_x is not None and raw_y is not None and str(raw_x).strip() != "" and str(raw_y).strip() != "":
        try:
            sx_mm = float(raw_x)
            sy_mm = float(raw_y)
            if math.isfinite(sx_mm) and math.isfinite(sy_mm):
                has_coords = True
                coords_reason = f"({sx_mm:.1f}, {sy_mm:.1f}) mm"
            else:
                issues.append("Surface coordinates (X_s, Y_s) must be finite numbers.")
        except (ValueError, TypeError):
            issues.append("Surface coordinates (X_s, Y_s) must be valid numbers.")
    else:
        issues.append("Missing receiving-surface coordinates (surface_x_mm, surface_y_mm).")

    # 5. Propagate individual angle uncertainty
    sigma_alpha_deg = None
    try:
        unc = propagate_uncertainty(width=record["width"], length=record["length"])
        sigma_alpha_deg = unc["sigma_angle_degrees"]
    except Exception:
        sigma_alpha_deg = None

    is_eligible = is_cal and (not is_perp) and has_valid_dir and has_coords and len(issues) == 0

    return {
        "is_eligible": is_eligible,
        "is_calibrated": is_cal,
        "calibration_reason": cal_reason,
        "is_perpendicular": is_perp,
        "has_valid_direction": has_valid_dir,
        "gamma_deg": gamma_deg,
        "reverse_bearing_deg": reverse_bearing_deg,
        "direction_reason": dir_reason,
        "has_surface_coords": has_coords,
        "surface_x_mm": sx_mm,
        "surface_y_mm": sy_mm,
        "coords_reason": coords_reason,
        "sigma_angle_degrees": sigma_alpha_deg,
        "issues": issues,
    }


# ===========================================================================
# 2. Reconstruction Workspace Management
# ===========================================================================


def get_or_create_case_reconstruction(
    case_id: str,
    reconstruction_id: Optional[str] = None,
    reconstruction_name: str = "Primary Scene Reconstruction",
    surface_type: str = "Horizontal Floor",
) -> Dict[str, Any]:
    """
    Verify that case_id exists and return either the specified reconstruction_id
    or the active reconstruction for the case (creating one if none exists).
    """
    if not case_id or not str(case_id).strip():
        raise ReconstructionValidationError("A valid case_id is required.")

    case = database.get_case_by_id(case_id)
    if not case:
        raise ReconstructionValidationError(f"Forensic Case '{case_id}' was not found in the database.")

    if reconstruction_id:
        rec = database.get_reconstruction_by_id(reconstruction_id)
        if not rec or rec["case_id"] != case["case_id"]:
            raise ReconstructionValidationError(
                f"Reconstruction '{reconstruction_id}' was not found for Case '{case_id}'."
            )
        return rec

    existing = database.get_reconstructions_for_case(case["case_id"])
    if existing:
        return existing[0]

    return database.create_reconstruction(
        case_id=case["case_id"],
        reconstruction_name=reconstruction_name,
        surface_type=surface_type,
    )


def add_measurement_to_case_reconstruction(
    case_id: str,
    measurement_id: Any,
    reconstruction_id: Optional[str] = None,
    stain_identifier: Optional[str] = None,
    surface_x_mm: Any = None,
    surface_y_mm: Any = None,
    direction_angle: Any = None,
    surface_type: Optional[str] = None,
    direction_source: str = "Examiner Supplied (Tail/Spine Morphology)",
    examiner_notes: str = "",
) -> Dict[str, Any]:
    """
    Validate and add an existing measurement to a case's reconstruction workspace.
    Prevents duplicate measurement associations and validates directional/coordinate inputs if supplied.
    """
    rec = get_or_create_case_reconstruction(case_id, reconstruction_id=reconstruction_id)

    try:
        mid = int(measurement_id)
    except (ValueError, TypeError):
        raise ReconstructionValidationError(f"Invalid measurement_id: '{measurement_id}'.")

    meas = database.get_measurement_by_id(mid)
    if not meas:
        raise ReconstructionValidationError(f"Measurement ID {mid} does not exist.")

    if meas.get("case_id") and meas["case_id"] != rec["case_id"]:
        raise ReconstructionValidationError(
            f"Measurement ID {mid} belongs to Case '{meas['case_id']}', not '{rec['case_id']}'."
        )

    parsed_x = None
    if surface_x_mm is not None and str(surface_x_mm).strip() != "":
        try:
            parsed_x = float(surface_x_mm)
            if not math.isfinite(parsed_x):
                raise ValueError()
        except (ValueError, TypeError):
            raise ReconstructionValidationError(f"Invalid surface_x_mm coordinate: '{surface_x_mm}'.")

    parsed_y = None
    if surface_y_mm is not None and str(surface_y_mm).strip() != "":
        try:
            parsed_y = float(surface_y_mm)
            if not math.isfinite(parsed_y):
                raise ValueError()
        except (ValueError, TypeError):
            raise ReconstructionValidationError(f"Invalid surface_y_mm coordinate: '{surface_y_mm}'.")

    parsed_dir = None
    if direction_angle is not None and str(direction_angle).strip() != "":
        parsed_dir = validate_direction_angle_input(direction_angle)

    try:
        stain_row = database.add_stain_to_reconstruction(
            reconstruction_id=rec["reconstruction_id"],
            measurement_id=mid,
            stain_identifier=stain_identifier,
            surface_x_mm=parsed_x,
            surface_y_mm=parsed_y,
            direction_angle=parsed_dir,
            surface_type=surface_type or rec["surface_type"],
            direction_source=direction_source,
            examiner_notes=examiner_notes,
        )
    except sqlite3.IntegrityError as exc:
        raise ReconstructionValidationError(
            f"Measurement ID {mid} is already associated with reconstruction '{rec['reconstruction_id']}'."
        ) from exc

    return stain_row


def update_stain_in_case_reconstruction(
    case_id: str,
    measurement_id: Any,
    reconstruction_id: Optional[str] = None,
    stain_identifier: Optional[str] = None,
    surface_x_mm: Any = None,
    surface_y_mm: Any = None,
    direction_angle: Any = None,
    surface_type: Optional[str] = None,
    direction_source: Optional[str] = None,
    examiner_notes: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Update examiner-supplied directionality, surface position (X_s, Y_s), or surface type
    for a stain in the case's reconstruction workspace.
    """
    rec = get_or_create_case_reconstruction(case_id, reconstruction_id=reconstruction_id)
    try:
        mid = int(measurement_id)
    except (ValueError, TypeError):
        raise ReconstructionValidationError(f"Invalid measurement_id: '{measurement_id}'.")

    parsed_x = None
    if surface_x_mm is not None and str(surface_x_mm).strip() != "":
        try:
            parsed_x = float(surface_x_mm)
            if not math.isfinite(parsed_x):
                raise ValueError()
        except (ValueError, TypeError):
            raise ReconstructionValidationError(f"Invalid surface_x_mm coordinate: '{surface_x_mm}'.")

    parsed_y = None
    if surface_y_mm is not None and str(surface_y_mm).strip() != "":
        try:
            parsed_y = float(surface_y_mm)
            if not math.isfinite(parsed_y):
                raise ValueError()
        except (ValueError, TypeError):
            raise ReconstructionValidationError(f"Invalid surface_y_mm coordinate: '{surface_y_mm}'.")

    parsed_dir = None
    if direction_angle is not None and str(direction_angle).strip() != "":
        parsed_dir = validate_direction_angle_input(direction_angle)

    updated = database.update_reconstruction_stain(
        reconstruction_id=rec["reconstruction_id"],
        measurement_id=mid,
        stain_identifier=stain_identifier,
        surface_x_mm=parsed_x,
        surface_y_mm=parsed_y,
        direction_angle=parsed_dir,
        surface_type=surface_type,
        direction_source=direction_source,
        examiner_notes=examiner_notes,
    )
    if not updated:
        raise ReconstructionValidationError(
            f"Measurement ID {mid} is not part of reconstruction '{rec['reconstruction_id']}'."
        )

    stains = database.get_stains_for_reconstruction(rec["reconstruction_id"])
    for st in stains:
        if st["measurement_id"] == mid:
            return st
    raise ReconstructionValidationError(f"Failed to retrieve updated stain {mid}.")


def remove_measurement_from_case_reconstruction(
    case_id: str,
    measurement_id: Any,
    reconstruction_id: Optional[str] = None,
) -> bool:
    """Remove a stain measurement from a case's reconstruction workspace."""
    rec = get_or_create_case_reconstruction(case_id, reconstruction_id=reconstruction_id)
    try:
        mid = int(measurement_id)
    except (ValueError, TypeError):
        raise ReconstructionValidationError(f"Invalid measurement_id: '{measurement_id}'.")

    removed = database.remove_stain_from_reconstruction(rec["reconstruction_id"], mid)
    if not removed:
        raise ReconstructionValidationError(
            f"Measurement ID {mid} was not found in reconstruction '{rec['reconstruction_id']}'."
        )
    return True


# ===========================================================================
# 3. Scientific Validation & Automated Reconstruction Execution
# ===========================================================================


def get_workspace_state(case_id: str, reconstruction_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Assemble the complete state of a case's reconstruction workspace:
    - Case metadata
    - Active reconstruction record
    - Available case measurements (with eligibility flags & whether already added)
    - Selected reconstruction stains (with per-stain eligibility & validation details)
    - Overall readiness status and latest persisted reconstruction result
    """
    case = database.get_case_by_id(case_id)
    if not case:
        raise ReconstructionValidationError(f"Forensic Case '{case_id}' was not found.")

    rec = get_or_create_case_reconstruction(case_id, reconstruction_id=reconstruction_id)
    rec_stains = database.get_stains_for_reconstruction(rec["reconstruction_id"])
    added_mids = {s["measurement_id"] for s in rec_stains}

    available_meas = database.get_measurements_for_case(case_id)
    enriched_available = []
    for m in available_meas:
        elig = evaluate_stain_eligibility(m)
        m_copy = dict(m)
        m_copy["validation"] = elig
        m_copy["already_added"] = m["id"] in added_mids
        enriched_available.append(m_copy)

    enriched_selected = []
    ineligible_reasons: List[str] = []
    for s in rec_stains:
        elig = evaluate_stain_eligibility(s)
        s_copy = dict(s)
        s_copy["validation"] = elig
        enriched_selected.append(s_copy)
        if not elig["is_eligible"]:
            joined_issues = "; ".join(elig["issues"])
            ineligible_reasons.append(f"{s['stain_identifier']} (Measurement #{s['measurement_id']}): {joined_issues}")

    can_calculate = (len(enriched_selected) >= 2) and (len(ineligible_reasons) == 0)
    latest_result = database.get_latest_reconstruction_result(rec["reconstruction_id"])
    parsed_scene = None
    if latest_result and latest_result.get("result_json"):
        try:
            parsed_scene = json.loads(latest_result["result_json"])
        except Exception:
            parsed_scene = None

    return {
        "case": case,
        "reconstruction": rec,
        "available_measurements": enriched_available,
        "selected_stains": enriched_selected,
        "can_calculate": can_calculate,
        "ineligible_reasons": ineligible_reasons,
        "latest_result": latest_result,
        "latest_scene": parsed_scene,
    }


def execute_case_reconstruction(
    case_id: str,
    reconstruction_id: Optional[str] = None,
    surface_type: Optional[str] = None,
    incline_deg: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Validate all selected stains in a case's reconstruction workspace, execute 2D Area of
    Convergence and 3D Area of Origin reconstruction via trajectory.py / trajectory_visualization.py,
    and store the verified results in SQLite.

    Scientific Safeguards:
    - Requires at least 2 stains.
    - Never silently discards problematic stains: if ANY selected stain is uncalibrated,
      missing directionality, perpendicular (90°), or missing surface coordinates, raises
      ReconstructionValidationError with explicit per-stain diagnostics.
    - Reports geometric identifiability warnings (e.g., 2-ray minimum constraint vs >= 3 rays).
    """
    rec = get_or_create_case_reconstruction(case_id, reconstruction_id=reconstruction_id)

    eff_surface = (surface_type or rec["surface_type"] or "Horizontal Floor").strip()
    eff_incline = incline_deg if incline_deg is not None else rec.get("incline_deg")

    database.update_reconstruction_metadata(
        reconstruction_id=rec["reconstruction_id"],
        surface_type=eff_surface,
        incline_deg=eff_incline,
    )

    rec_stains = database.get_stains_for_reconstruction(rec["reconstruction_id"])
    if len(rec_stains) < 2:
        raise ReconstructionValidationError(
            f"At least 2 calibrated directional bloodstains are required for trajectory reconstruction "
            f"(currently {len(rec_stains)} selected)."
        )

    # Validate every single selected stain — NEVER silently discard problematic stains
    problem_stains: List[str] = []
    stain_inputs: List[StainTrajectoryInput] = []

    for s in rec_stains:
        elig = evaluate_stain_eligibility(s)
        if not elig["is_eligible"]:
            problem_stains.append(f"{s['stain_identifier']} (#{s['measurement_id']}): {'; '.join(elig['issues'])}")
        else:
            stain_inputs.append(
                StainTrajectoryInput(
                    stain_id=s["stain_identifier"],
                    width=float(s["width"]),
                    length=float(s["length"]),
                    surface_x_mm=float(elig["surface_x_mm"]),
                    surface_y_mm=float(elig["surface_y_mm"]),
                    gamma_deg=float(elig["gamma_deg"]),
                    sigma_gamma_deg=float(s.get("sigma_gamma_deg") or 2.0),
                    is_calibrated=True,
                )
            )

    if problem_stains:
        raise ReconstructionValidationError(
            "Cannot execute reconstruction because one or more selected stains failed forensic validation: "
            + " | ".join(problem_stains)
        )

    # Execute mathematical & visual reconstruction via trajectory_visualization.py -> trajectory.py
    try:
        vis_2d = trajectory_visualization.render_2d_convergence_plot(
            stains=stain_inputs,
            surface_type=eff_surface,
        )
        vis_3d = trajectory_visualization.render_3d_trajectory_plot(
            stains=stain_inputs,
            surface_type=eff_surface,
            incline_deg=eff_incline,
        )
    except TrajectoryError as traj_err:
        database.update_reconstruction_metadata(rec["reconstruction_id"], status="Invalid")
        raise ReconstructionValidationError(f"Geometric Trajectory Error: {traj_err}") from traj_err

    scene = vis_3d["scene"]
    conv_2d = scene["convergence_2d"]
    tan_3d = scene["tangent_origin_3d"]
    cpa_3d = scene["cpa_origin_3d"]

    # Geometric identifiability & conditioning warnings
    warnings: List[str] = []
    if len(stain_inputs) == 2:
        warnings.append(
            "Two-Stain Minimum Geometry Caution: A 2-ray reconstruction has zero redundancy in 2D "
            "planar intersection and cannot independently cross-validate angular measurement error. "
            "Include 3 or more well-separated stains whenever possible."
        )
    if conv_2d["condition_number"] > 15.0:
        warnings.append(
            f"High Geometric Condition Number ({conv_2d['condition_number']:.1f}): Stain trajectories "
            "intersect at a shallow angle, increasing spatial uncertainty along the ray direction."
        )
    scene["warnings"] = warnings

    saved_res = database.save_reconstruction_result(
        reconstruction_id=rec["reconstruction_id"],
        stain_count=scene["stain_count"],
        surface_type=scene["surface_basis"]["surface_type"],
        convergence_x_mm=conv_2d["x_surface_mm"],
        convergence_y_mm=conv_2d["y_surface_mm"],
        rms_residual_mm=conv_2d["rms_residual_mm"],
        max_residual_mm=conv_2d["max_residual_mm"],
        convergence_radius_mm=conv_2d["convergence_radius_mm"],
        condition_number=conv_2d["condition_number"],
        tangent_origin_x_mm=tan_3d["x_surface_mm"],
        tangent_origin_y_mm=tan_3d["y_surface_mm"],
        tangent_origin_z_mm=tan_3d["z_mean_surface_mm"],
        tangent_z_std_mm=tan_3d["z_std_surface_mm"],
        tangent_z_uncertainty_mm=tan_3d["z_combined_uncertainty_mm"],
        tangent_z_min_mm=tan_3d["z_range_surface_mm"][0],
        tangent_z_max_mm=tan_3d["z_range_surface_mm"][1],
        estimated_origin_x_mm=cpa_3d["point_lab_mm"][0],
        estimated_origin_y_mm=cpa_3d["point_lab_mm"][1],
        estimated_origin_z_mm=cpa_3d["point_lab_mm"][2],
        rms_3d_miss_mm=cpa_3d["rms_3d_miss_mm"],
        result_json=json.dumps(scene),
    )

    return {
        "reconstruction": database.get_reconstruction_by_id(rec["reconstruction_id"]),
        "result": saved_res,
        "scene": scene,
        "plot_2d_b64": vis_2d["base64_png"],
        "plot_3d_b64": vis_3d["base64_png"],
        "warnings": warnings,
        "disclaimer": FORENSIC_TRAJECTORY_DISCLAIMER,
    }

