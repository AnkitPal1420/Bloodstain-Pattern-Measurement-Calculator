"""
overlap_analysis.py
===================
Phase 5.2 — Advanced Overlapping & Complex Bloodstain Pattern Analysis Engine
Forensic Laboratory Suite

Responsibilities:
1. Complex Connected-Component Analysis & Morphological Topology Inspection:
   - Identifies Isolated Stains, Touching Stains, Overlapping Stains, Dense Clusters,
     Irregular Merged Regions, and Uncertain / Indeterminate Regions.
2. Distance-Transform Marker-Controlled Watershed Segmentation:
   - Uses distance-transform Euclidean peaks and concavity defect notches to propose
     candidate separation boundaries without indiscriminate over-splitting.
3. Examiner-Assisted Review & Decision Workflow:
   - Retains parent-child region tracking (`REG-XX` -> `REG-XX-S01`, `REG-XX-S02`).
   - Supports explicit statuses: `Clearly Separated`, `Candidate Separation`,
     `Examiner Reviewed`, `Accepted Segmentation`, `Rejected Segmentation`,
     `Indeterminate`, and `Unresolved Merged Region`.
4. Scientific Measurement Integrity & Unit Discipline:
   - Enforces strict separation of uncalibrated (`_px`, `_px2`) vs. calibrated (`_mm`, `_mm2`).
   - Area conservation tracking between parent region and proposed child segments.
   - Never manufactures artificial measurements or claims higher-layer occlusion without evidence.
"""

import base64
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

import image_processing
from image_processing import ImageProcessingError

OVERLAP_ENGINE_VERSION = "BPMS Overlap Analysis Engine v5.2 (Examiner-Assisted)"

# Provenance labels
PROVENANCE_AUTO_MEASURED = "Automatically Measured"
PROVENANCE_EXAMINER_OBSERVED = "Examiner Observed"
PROVENANCE_EXAMINER_SUPPLIED = "Examiner Supplied"
PROVENANCE_SOFTWARE_SUGGESTED = "Software Suggested"
PROVENANCE_EXAMINER_APPROVED = "Examiner Approved"

# Region classification types
REGION_TYPE_ISOLATED = "Isolated Stain"
REGION_TYPE_TOUCHING = "Touching Stains"
REGION_TYPE_OVERLAPPING = "Overlapping Stains"
REGION_TYPE_CLUSTER = "Dense Cluster"
REGION_TYPE_IRREGULAR = "Irregular Merged Region"
REGION_TYPE_INDETERMINATE = "Uncertain / Indeterminate Region"

# Segmentation statuses
STATUS_CLEARLY_SEPARATED = "Clearly Separated"
STATUS_CANDIDATE_SEPARATION = "Candidate Separation"
STATUS_EXAMINER_REVIEWED = "Examiner Reviewed"
STATUS_ACCEPTED = "Accepted Segmentation"
STATUS_REJECTED = "Rejected Segmentation"
STATUS_INDETERMINATE = "Indeterminate"
STATUS_UNRESOLVED_MERGED = "Unresolved Merged Region"


class OverlapAnalysisError(ValueError):
    """Raised when complex stain detection or segmentation encounters invalid geometry or domain failure."""
    pass


# ===========================================================================
# 1. Feature Helpers & Mathematical Safeguards
# ===========================================================================

def compute_circularity(area_px2: float, perimeter_px: float) -> float:
    """Circularity (isoperimetric quotient) = 4 * pi * Area / Perimeter^2 in (0, 1]."""
    if area_px2 <= 0.0 or perimeter_px <= 0.0:
        raise OverlapAnalysisError("Area and perimeter must be strictly positive.")
    raw = (4.0 * math.pi * float(area_px2)) / (float(perimeter_px) ** 2)
    return round(min(max(raw, 0.0), 1.0), 4)


def compute_aspect_ratio(major_px: float, minor_px: float) -> float:
    """Aspect ratio = Major Axis / Minor Axis >= 1.0."""
    if major_px <= 0.0 or minor_px <= 0.0:
        raise OverlapAnalysisError("Axes must be strictly positive.")
    eff_maj = max(float(major_px), float(minor_px))
    eff_min = min(float(major_px), float(minor_px))
    return round(eff_maj / eff_min, 4)


def compute_solidity(area_px2: float, hull_area_px2: float) -> float:
    """Solidity = Contour Area / Convex Hull Area in (0, 1]."""
    if area_px2 <= 0.0 or hull_area_px2 <= 0.0:
        raise OverlapAnalysisError("Contour area and hull area must be strictly positive.")
    return round(min(float(area_px2) / float(hull_area_px2), 1.0), 4)


def compute_eccentricity(major_px: float, minor_px: float) -> float:
    """First eccentricity e = sqrt(1 - (Minor / Major)^2) in [0, 1)."""
    if major_px <= 0.0 or minor_px <= 0.0:
        raise OverlapAnalysisError("Axes must be strictly positive.")
    eff_maj = max(float(major_px), float(minor_px))
    eff_min = min(float(major_px), float(minor_px))
    val = max(1.0 - (eff_min / eff_maj) ** 2, 0.0)
    return round(math.sqrt(val), 4)


# ===========================================================================
# 2. Multi-Peak Distance Transform Detection
# ===========================================================================

def find_distance_transform_peaks(
    dist_map: np.ndarray,
    min_distance: float = 6.0,
    threshold_ratio: float = 0.35,
) -> List[Tuple[int, int, float]]:
    """
    Locate significant local maxima in a Euclidean distance transform map.
    Filters out spurious micro-peaks caused by noisy digital boundaries.

    Returns list of (x, y, peak_val) sorted highest-peak first.
    """
    max_val = float(np.max(dist_map))
    if max_val < 2.0:
        return []

    cutoff = max_val * max(threshold_ratio, 0.20)
    # Binary mask of candidate peak regions
    cand_mask = (dist_map >= cutoff).astype(np.uint8) * 255

    # Dilate distance transform to find strict local maxima
    ksize = int(max(3, int(min_distance) // 2 * 2 + 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    dilated = cv2.dilate(dist_map, kernel)
    peaks_mask = (dist_map == dilated) & (dist_map >= cutoff)

    ys, xs = np.where(peaks_mask)
    if len(xs) == 0:
        # Fallback to absolute maximum centroid
        max_ys, max_xs = np.where(dist_map == max_val)
        return [(int(max_xs[0]), int(max_ys[0]), max_val)]

    raw_peaks: List[Tuple[int, int, float]] = []
    for x, y in zip(xs, ys):
        raw_peaks.append((int(x), int(y), float(dist_map[y, x])))

    # Sort descending by distance peak height
    raw_peaks.sort(key=lambda p: p[2], reverse=True)

    # Non-maximum suppression by Euclidean distance
    filtered_peaks: List[Tuple[int, int, float]] = []
    for px, py, pval in raw_peaks:
        too_close = False
        for fx, fy, _ in filtered_peaks:
            dist = math.hypot(px - fx, py - fy)
            if dist < min_distance:
                too_close = True
                break
        if not too_close:
            filtered_peaks.append((px, py, pval))

    return filtered_peaks


# ===========================================================================
# 3. Contour Topology & Concavity Notch Analysis
# ===========================================================================

def analyze_contour_concavity_notches(
    contour: np.ndarray,
    min_depth_px: float = 3.5,
) -> List[Dict[str, Any]]:
    """
    Detect inward concavity notches (constriction necks) using convexity defects.
    Notches indicate points where touching or overlapping stains coalesce.
    """
    notches: List[Dict[str, Any]] = []
    if len(contour) < 5:
        return notches

    try:
        hull_idx = cv2.convexHull(contour, returnPoints=False)
        if hull_idx is not None and len(hull_idx) >= 4:
            defects = cv2.convexityDefects(contour, hull_idx)
            if defects is not None:
                for d_row in defects:
                    flat = np.asarray(d_row).ravel()
                    if len(flat) >= 4:
                        start_idx = int(flat[0])
                        end_idx = int(flat[1])
                        far_idx = int(flat[2])
                        depth_px = float(flat[3]) / 256.0
                        if depth_px >= min_depth_px:
                            far_pt = contour[far_idx][0]
                            notches.append(
                                {
                                    "x": int(far_pt[0]),
                                    "y": int(far_pt[1]),
                                    "depth_px": round(depth_px, 2),
                                    "start_pt": [int(contour[start_idx][0][0]), int(contour[start_idx][0][1])],
                                    "end_pt": [int(contour[end_idx][0][0]), int(contour[end_idx][0][1])],
                                }
                            )
    except (cv2.error, IndexError, TypeError, ValueError):
        pass

    return notches


# ===========================================================================
# 4. Marker-Controlled Watershed Segmentation
# ===========================================================================

def execute_marker_watershed(
    sub_binary_mask: np.ndarray,
    peaks: List[Tuple[int, int, float]],
    padding: int = 5,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Perform controlled distance-transform watershed segmentation on a connected region mask.

    Returns:
    - watershed_markers: full 2D label matrix (-1 = boundary, 1..K = child segments)
    - boundary_mask: binary 1-pixel boundary line mask
    - child_contours: list of extracted child contours in sub_binary coordinates
    """
    h, w = sub_binary_mask.shape[:2]
    # Inverted distance transform for watershed topography
    dist = cv2.distanceTransform(sub_binary_mask, cv2.DIST_L2, 5)

    # Initialize markers: 0 = unknown, 1..K = seed regions
    markers = np.zeros((h, w), dtype=np.int32)
    for idx, (px, py, _) in enumerate(peaks, start=1):
        # Create a small circular seed around each peak
        seed_r = max(2, int(min(dist[py, px] * 0.25, 6)))
        cv2.circle(markers, (px, py), seed_r, idx, -1)

    # Prepare 3-channel inverted gradient representation
    dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    dist_inv = 255 - dist_norm
    # Keep background outside region as high ridge
    dist_inv[sub_binary_mask == 0] = 255
    ws_input = cv2.cvtColor(dist_inv, cv2.COLOR_GRAY2BGR)

    # Run OpenCV watershed
    cv2.watershed(ws_input, markers)

    # Boundaries where markers == -1
    boundary_mask = np.zeros((h, w), dtype=np.uint8)
    boundary_mask[(markers == -1) & (sub_binary_mask > 0)] = 255

    child_contours: List[np.ndarray] = []
    for k in range(1, len(peaks) + 1):
        child_mask = np.zeros((h, w), dtype=np.uint8)
        child_mask[(markers == k) & (sub_binary_mask > 0)] = 255
        # Clean small fragments
        cnts, _ = cv2.findContours(child_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            largest = max(cnts, key=lambda c: cv2.contourArea(c))
            if cv2.contourArea(largest) >= 8.0:
                child_contours.append(largest)

    return markers, boundary_mask, child_contours


# ===========================================================================
# 5. Connected Component Analysis & Complex Stain Detection
# ===========================================================================

def detect_and_segment_complex_regions(
    image_source: Union[str, bytes, np.ndarray],
    is_calibrated: bool = False,
    mm_per_pixel: Optional[float] = None,
    threshold_method: str = "otsu",
    min_area_px: float = 20.0,
    blur_kernel: int = 5,
    min_peak_distance_px: float = 8.0,
    peak_threshold_ratio: float = 0.35,
) -> Dict[str, Any]:
    """
    Comprehensive Phase 5.2 detection & segmentation pipeline for complex, touching,
    and overlapping bloodstains.

    CRITICAL SCIENTIFIC RULE ENFORCED:
    Never automatically declare a merged contour represents a specific number of
    individual stains unless topological evidence (peaks, concavity defects) supports it.
    Uncertain regions are preserved as `Unresolved Merged Region` or `Indeterminate`.
    """
    valid_cal = bool(is_calibrated and mm_per_pixel is not None and float(mm_per_pixel) > 0.0)
    scale_mm: Optional[float] = round(float(mm_per_pixel), 6) if valid_cal else None

    original_bgr = image_processing.load_and_validate_image(image_source)
    gray = image_processing.convert_to_grayscale(original_bgr)
    h_img, w_img = gray.shape[:2]

    blurred = image_processing.apply_gaussian_blur(gray, kernel_size=blur_kernel)
    binary = image_processing.apply_thresholding(blurred, method=threshold_method, invert=True)
    morphed = image_processing.apply_morphology(binary, open_kernel=3, close_kernel=3)

    raw_contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(raw_contours, key=lambda c: cv2.contourArea(c), reverse=True)

    regions: List[Dict[str, Any]] = []
    reg_counter = 1

    for c in sorted_contours:
        area_px2 = float(cv2.contourArea(c))
        if area_px2 < min_area_px:
            continue
        # Frame boundary envelope check (>94% image area)
        if area_px2 > 0.94 * float(h_img * w_img):
            continue

        perim_px = float(cv2.arcLength(c, True))
        bx, by, bw, bh = cv2.boundingRect(c)
        hull = cv2.convexHull(c)
        hull_area = float(cv2.contourArea(hull))

        solidity = compute_solidity(area_px2, hull_area) if hull_area > 0 else 1.0
        circularity = compute_circularity(area_px2, perim_px) if perim_px > 0 else 0.0

        # Centroid moments
        M = cv2.moments(c)
        if M["m00"] > 0:
            cx = float(M["m10"] / M["m00"])
            cy = float(M["m01"] / M["m00"])
        else:
            cx = float(bx + bw / 2.0)
            cy = float(by + bh / 2.0)

        # Ellipse fit on parent region
        parent_major_px = max(float(bw), float(bh))
        parent_minor_px = min(float(bw), float(bh))
        parent_angle_deg = 0.0
        if len(c) >= 5:
            try:
                el = cv2.fitEllipse(c)
                parent_minor_px = min(float(el[1][0]), float(el[1][1]))
                parent_major_px = max(float(el[1][0]), float(el[1][1]))
                parent_angle_deg = float(el[2])
            except cv2.error:
                pass

        aspect_ratio = compute_aspect_ratio(parent_major_px, parent_minor_px)
        eccentricity = compute_eccentricity(parent_major_px, parent_minor_px)

        # Sub-image mask for localized distance transform and concavity notch analysis
        pad = 6
        x1 = max(0, bx - pad)
        y1 = max(0, by - pad)
        x2 = min(w_img, bx + bw + pad)
        y2 = min(h_img, by + bh + pad)

        sub_mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        c_shifted = c - np.array([[[x1, y1]]], dtype=np.int32)
        cv2.drawContours(sub_mask, [c_shifted], -1, 255, -1)

        dist_map = cv2.distanceTransform(sub_mask, cv2.DIST_L2, 5)
        peaks_local = find_distance_transform_peaks(
            dist_map=dist_map,
            min_distance=min_peak_distance_px,
            threshold_ratio=peak_threshold_ratio,
        )

        notches_local = analyze_contour_concavity_notches(c_shifted, min_depth_px=max(3.0, 0.08 * parent_minor_px))
        peak_count = len(peaks_local)
        notch_count = len(notches_local)

        # Region classification logic
        if peak_count <= 1 and notch_count == 0 and solidity >= 0.88:
            region_type = REGION_TYPE_ISOLATED
            seg_status = STATUS_CLEARLY_SEPARATED
        elif peak_count == 2:
            region_type = REGION_TYPE_TOUCHING if (notch_count >= 2 and aspect_ratio > 1.35) else REGION_TYPE_OVERLAPPING
            seg_status = STATUS_CANDIDATE_SEPARATION
        elif peak_count >= 3:
            region_type = REGION_TYPE_CLUSTER if solidity >= 0.65 else REGION_TYPE_IRREGULAR
            seg_status = STATUS_CANDIDATE_SEPARATION
        elif solidity < 0.78 or notch_count >= 2:
            region_type = REGION_TYPE_IRREGULAR
            seg_status = STATUS_CANDIDATE_SEPARATION if peak_count >= 2 else STATUS_UNRESOLVED_MERGED
        else:
            region_type = REGION_TYPE_INDETERMINATE
            seg_status = STATUS_INDETERMINATE

        reg_id = f"REG-{reg_counter:02d}"
        candidate_children: List[Dict[str, Any]] = []
        boundary_pts_global: List[List[int]] = []
        area_diff_px2 = 0.0
        area_ratio = 1.0
        uncertainty_notes: List[str] = []

        # Execute marker-controlled watershed when multi-peak candidate exists
        if peak_count >= 2:
            try:
                markers_sub, boundary_sub, child_cnts = execute_marker_watershed(
                    sub_binary_mask=sub_mask,
                    peaks=peaks_local,
                    padding=pad,
                )
                b_ys, b_xs = np.where(boundary_sub > 0)
                for bx_loc, by_loc in zip(b_xs, b_ys):
                    boundary_pts_global.append([int(bx_loc + x1), int(by_loc + y1)])

                total_child_area = 0.0
                for c_idx, ch_c in enumerate(child_cnts, start=1):
                    ch_area_px2 = float(cv2.contourArea(ch_c))
                    total_child_area += ch_area_px2
                    ch_perim_px = float(cv2.arcLength(ch_c, True))
                    ch_bx, ch_by, ch_bw, ch_bh = cv2.boundingRect(ch_c)

                    ch_M = cv2.moments(ch_c)
                    if ch_M["m00"] > 0:
                        ch_cx = float(ch_M["m10"] / ch_M["m00"]) + x1
                        ch_cy = float(ch_M["m01"] / ch_M["m00"]) + y1
                    else:
                        ch_cx = float(ch_bx + ch_bw / 2.0) + x1
                        ch_cy = float(ch_by + ch_bh / 2.0) + y1

                    ch_maj_px = max(float(ch_bw), float(ch_bh))
                    ch_min_px = min(float(ch_bw), float(ch_bh))
                    ch_ang_deg = 0.0
                    if len(ch_c) >= 5:
                        try:
                            ch_el = cv2.fitEllipse(ch_c)
                            ch_min_px = min(float(ch_el[1][0]), float(ch_el[1][1]))
                            ch_maj_px = max(float(ch_el[1][0]), float(ch_el[1][1]))
                            ch_ang_deg = float(ch_el[2])
                        except cv2.error:
                            pass

                    ch_circ = compute_circularity(ch_area_px2, ch_perim_px) if ch_perim_px > 0 else 0.0
                    ch_ar = compute_aspect_ratio(ch_maj_px, ch_min_px)
                    ch_ecc = compute_eccentricity(ch_maj_px, ch_min_px)

                    # Calibrated metrics
                    ch_area_mm2 = round(ch_area_px2 * (scale_mm ** 2), 4) if valid_cal and scale_mm else None
                    ch_min_mm = round(ch_min_px * scale_mm, 3) if valid_cal and scale_mm else None
                    ch_maj_mm = round(ch_maj_px * scale_mm, 3) if valid_cal and scale_mm else None

                    # Global contour points
                    ch_global_pts = (ch_c + np.array([[[x1, y1]]], dtype=np.int32)).tolist()

                    candidate_children.append(
                        {
                            "child_id": f"{reg_id}-S{c_idx:02d}",
                            "parent_region_id": reg_id,
                            "provenance": PROVENANCE_SOFTWARE_SUGGESTED,
                            "area_px2": round(ch_area_px2, 2),
                            "area_mm2": ch_area_mm2,
                            "perimeter_px": round(ch_perim_px, 2),
                            "minor_axis_px": round(ch_min_px, 2),
                            "major_axis_px": round(ch_maj_px, 2),
                            "minor_axis_mm": ch_min_mm,
                            "major_axis_mm": ch_maj_mm,
                            "aspect_ratio": ch_ar,
                            "circularity": ch_circ,
                            "eccentricity": ch_ecc,
                            "orientation_deg": round(ch_ang_deg, 2),
                            "centroid_px": [round(ch_cx, 2), round(ch_cy, 2)],
                            "bbox_px": [int(ch_bx + x1), int(ch_by + y1), int(ch_bw), int(ch_bh)],
                            "contour_pts": ch_global_pts,
                        }
                    )

                area_diff_px2 = round(abs(area_px2 - total_child_area), 2)
                area_ratio = round(total_child_area / max(area_px2, 1e-4), 4)

                if area_ratio < 0.85:
                    uncertainty_notes.append(
                        f"Area conservation disparity ({area_ratio*100:.1f}% accounted for); watershed boundary loss occurred."
                    )
                if notch_count == 0:
                    uncertainty_notes.append(
                        "Multiple distance peaks detected without corroborating contour concavity notches; possible over-segmentation."
                    )
            except Exception as exc:
                seg_status = STATUS_INDETERMINATE
                uncertainty_notes.append(f"Watershed segmentation failed numerically: {exc}")
        elif peak_count == 1 and solidity < 0.75:
            uncertainty_notes.append(
                "Irregular perimeter exhibits low solidity, but distance transform reveals only one central peak; manual boundary review recommended."
            )

        parent_area_mm2 = round(area_px2 * (scale_mm ** 2), 4) if valid_cal and scale_mm else None
        parent_min_mm = round(parent_minor_px * scale_mm, 3) if valid_cal and scale_mm else None
        parent_maj_mm = round(parent_major_px * scale_mm, 3) if valid_cal and scale_mm else None

        regions.append(
            {
                "region_id": reg_id,
                "region_type": region_type,
                "provenance": PROVENANCE_AUTO_MEASURED,
                "parent_area_px2": round(area_px2, 2),
                "parent_area_mm2": parent_area_mm2,
                "parent_perimeter_px": round(perim_px, 2),
                "parent_minor_axis_px": round(parent_minor_px, 2),
                "parent_major_axis_px": round(parent_major_px, 2),
                "parent_minor_axis_mm": parent_min_mm,
                "parent_major_axis_mm": parent_maj_mm,
                "parent_solidity": solidity,
                "parent_circularity": circularity,
                "parent_aspect_ratio": aspect_ratio,
                "parent_eccentricity": eccentricity,
                "parent_orientation_deg": round(parent_angle_deg, 2),
                "parent_centroid_px": [round(cx, 2), round(cy, 2)],
                "parent_bbox_px": [int(bx), int(by), int(bw), int(bh)],
                "contour_pts": c.tolist(),
                "peak_count": peak_count,
                "notch_count": notch_count,
                "segmentation_status": seg_status,
                "examiner_review_status": "Pending Examiner Review",
                "examiner_decision": "Pending",
                "examiner_notes": "",
                "candidate_children": candidate_children,
                "separation_boundary_pts": boundary_pts_global,
                "area_conservation_diff_px2": area_diff_px2,
                "area_conservation_ratio": area_ratio,
                "uncertainty_notes": uncertainty_notes,
            }
        )
        reg_counter += 1

    # Quality check
    quality = image_processing.evaluate_image_quality(gray) if hasattr(image_processing, "evaluate_image_quality") else {
        "contrast_std": round(float(np.std(gray)), 2),
        "is_sufficient_quality": float(np.std(gray)) >= 18.0,
    }

    # Summary counts
    type_counts = {
        "isolated": sum(1 for r in regions if r["region_type"] == REGION_TYPE_ISOLATED),
        "touching": sum(1 for r in regions if r["region_type"] == REGION_TYPE_TOUCHING),
        "overlapping": sum(1 for r in regions if r["region_type"] == REGION_TYPE_OVERLAPPING),
        "cluster": sum(1 for r in regions if r["region_type"] == REGION_TYPE_CLUSTER),
        "irregular": sum(1 for r in regions if r["region_type"] == REGION_TYPE_IRREGULAR),
        "indeterminate": sum(1 for r in regions if r["region_type"] == REGION_TYPE_INDETERMINATE),
    }

    # Render forensic overlays
    overlays = render_overlap_overlays(
        original_bgr=original_bgr,
        regions=regions,
        is_calibrated=valid_cal,
        mm_per_pixel=scale_mm,
    )

    return {
        "engine_version": OVERLAP_ENGINE_VERSION,
        "provenance": PROVENANCE_SOFTWARE_SUGGESTED,
        "is_calibrated": valid_cal,
        "measurement_unit": "mm (Calibrated Metric)" if valid_cal else "px (Uncalibrated Pixel Units)",
        "mm_per_pixel": scale_mm,
        "image_width_px": w_img,
        "image_height_px": h_img,
        "image_quality": quality,
        "total_regions": len(regions),
        "type_counts": type_counts,
        "candidate_splits_count": sum(1 for r in regions if len(r["candidate_children"]) >= 2),
        "regions": regions,
        "detection_overlay_b64": overlays["detection_overlay_b64"],
        "segmentation_overlay_b64": overlays["segmentation_overlay_b64"],
        "review_overlay_b64": overlays["review_overlay_b64"],
        "mandatory_scientific_disclaimers": get_mandatory_overlap_disclaimers(),
    }


# ===========================================================================
# 6. Forensic Visual Overlay Generators
# ===========================================================================

def render_overlap_overlays(
    original_bgr: np.ndarray,
    regions: List[Dict[str, Any]],
    is_calibrated: bool,
    mm_per_pixel: Optional[float],
) -> Dict[str, str]:
    """
    Produce publication-grade forensic overlay views encoded as base64 PNGs:
    1. detection_overlay_b64: connected region boundaries & IDs colored by morphology.
    2. segmentation_overlay_b64: candidate watershed cuts & child stain boundaries.
    3. review_overlay_b64: reflects current examiner review states (accepted vs. rejected).
    """
    h, w = original_bgr.shape[:2]

    # Color palette (BGR)
    COLOR_ISOLATED = (80, 210, 50)       # Green
    COLOR_TOUCHING = (0, 215, 255)       # Amber/Gold
    COLOR_OVERLAPPING = (0, 140, 255)    # Orange
    COLOR_CLUSTER = (255, 105, 180)      # Hot Pink
    COLOR_IRREGULAR = (230, 80, 80)      # Crimson/Purple
    COLOR_INDETERMINATE = (160, 160, 160)# Slate Gray
    COLOR_WATERSHED_CUT = (0, 255, 255)  # Bright Yellow

    # View 1: Detection Overlay
    vis_det = original_bgr.copy()
    for reg in regions:
        pts = np.array(reg["contour_pts"], dtype=np.int32)
        rtype = reg["region_type"]
        if rtype == REGION_TYPE_ISOLATED:
            c_color = COLOR_ISOLATED
        elif rtype == REGION_TYPE_TOUCHING:
            c_color = COLOR_TOUCHING
        elif rtype == REGION_TYPE_OVERLAPPING:
            c_color = COLOR_OVERLAPPING
        elif rtype == REGION_TYPE_CLUSTER:
            c_color = COLOR_CLUSTER
        elif rtype == REGION_TYPE_IRREGULAR:
            c_color = COLOR_IRREGULAR
        else:
            c_color = COLOR_INDETERMINATE

        cv2.drawContours(vis_det, [pts], -1, c_color, 2)
        cx = int(reg["parent_centroid_px"][0])
        cy = int(reg["parent_centroid_px"][1])
        cv2.putText(
            vis_det,
            reg["region_id"],
            (cx - 15, max(14, cy - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # View 2: Proposed Segmentation Overlay
    vis_seg = original_bgr.copy()
    for reg in regions:
        pts = np.array(reg["contour_pts"], dtype=np.int32)
        # Draw outer parent boundary in subtle gray
        cv2.drawContours(vis_seg, [pts], -1, (120, 120, 120), 1)

        children = reg.get("candidate_children", [])
        if len(children) >= 2:
            # Draw child segments in alternating vibrant colors
            child_palette = [(50, 220, 90), (0, 165, 255), (255, 90, 180), (255, 215, 0), (180, 105, 255)]
            for idx, ch in enumerate(children):
                ch_pts = np.array(ch["contour_pts"], dtype=np.int32)
                ch_col = child_palette[idx % len(child_palette)]
                cv2.drawContours(vis_seg, [ch_pts], -1, ch_col, 2)
                ccx = int(ch["centroid_px"][0])
                ccy = int(ch["centroid_px"][1])
                cv2.circle(vis_seg, (ccx, ccy), 3, ch_col, -1)
                cv2.putText(
                    vis_seg,
                    ch["child_id"],
                    (ccx + 4, ccy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            # Draw watershed separation cut points in bright yellow
            for bx, by in reg.get("separation_boundary_pts", []):
                if 0 <= bx < w and 0 <= by < h:
                    vis_seg[by, bx] = COLOR_WATERSHED_CUT
        else:
            # Single isolated stain or unresolved region
            c_col = (50, 210, 80) if reg["region_type"] == REGION_TYPE_ISOLATED else (0, 140, 255)
            cv2.drawContours(vis_seg, [pts], -1, c_col, 2)

    # View 3: Review Overlay (Accepted vs. Rejected)
    vis_rev = original_bgr.copy()
    for reg in regions:
        dec = reg.get("examiner_decision", "Pending")
        children = reg.get("candidate_children", [])
        pts = np.array(reg["contour_pts"], dtype=np.int32)

        if dec == "Accepted" and len(children) >= 2:
            # Draw approved child stains in emerald green
            for ch in children:
                ch_pts = np.array(ch["contour_pts"], dtype=np.int32)
                cv2.drawContours(vis_rev, [ch_pts], -1, (40, 210, 70), 2)
                ccx = int(ch["centroid_px"][0])
                ccy = int(ch["centroid_px"][1])
                cv2.putText(
                    vis_rev,
                    f"✓ {ch['child_id']}",
                    (ccx - 12, ccy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
        elif dec == "Rejected":
            # Rejection retains parent region as unbroken entity with amber hash
            cv2.drawContours(vis_rev, [pts], -1, (0, 140, 255), 2)
            cx = int(reg["parent_centroid_px"][0])
            cy = int(reg["parent_centroid_px"][1])
            cv2.putText(
                vis_rev,
                f"✗ {reg['region_id']} (Merged)",
                (cx - 25, cy - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (0, 215, 255),
                1,
                cv2.LINE_AA,
            )
        elif dec == "Indeterminate":
            cv2.drawContours(vis_rev, [pts], -1, (180, 180, 180), 2)
            cx = int(reg["parent_centroid_px"][0])
            cy = int(reg["parent_centroid_px"][1])
            cv2.putText(
                vis_rev,
                f"? {reg['region_id']} (Indet)",
                (cx - 20, cy - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )
        else:
            # Pending review
            cv2.drawContours(vis_rev, [pts], -1, (160, 160, 200), 1)

    def _to_b64(img_arr: np.ndarray) -> str:
        ok, buf = cv2.imencode(".png", img_arr)
        if not ok:
            return ""
        return f"data:image/png;base64,{base64.b64encode(buf.tobytes()).decode('ascii')}"

    return {
        "detection_overlay_b64": _to_b64(vis_det),
        "segmentation_overlay_b64": _to_b64(vis_seg),
        "review_overlay_b64": _to_b64(vis_rev),
    }


# ===========================================================================
# 7. Examiner Review & Decision Engine
# ===========================================================================

def apply_examiner_review_actions(
    analysis_record: Dict[str, Any],
    review_actions: Dict[str, Dict[str, Any]],
    examiner_name: Optional[str] = None,
    examiner_notes: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Apply examiner decisions on connected regions and candidate splits:
    - `accept`: marks proposed child stains as approved individual stains.
    - `reject`: rejects the split; retains the parent region as an unresolved/merged stain.
    - `indeterminate`: marks region as unresolved/indeterminate pending scene context.
    - `notes`: attaches examiner observation rationale.

    Never destroys the original automated segmentation result.
    """
    updated_regions: List[Dict[str, Any]] = []
    orig_regions = analysis_record.get("regions", [])

    accepted_stains: List[Dict[str, Any]] = []
    unresolved_regions: List[Dict[str, Any]] = []

    for reg in orig_regions:
        reg_copy = dict(reg)
        rid = reg_copy["region_id"]
        action_data = review_actions.get(rid, {})

        dec = action_data.get("decision", reg_copy.get("examiner_decision", "Pending")).strip()
        user_notes = action_data.get("notes", reg_copy.get("examiner_notes", "")).strip()

        reg_copy["examiner_decision"] = dec
        reg_copy["examiner_notes"] = user_notes

        if dec == "Accepted" and len(reg_copy.get("candidate_children", [])) >= 2:
            reg_copy["segmentation_status"] = STATUS_ACCEPTED
            reg_copy["examiner_review_status"] = "Examiner Approved"
            for ch in reg_copy["candidate_children"]:
                ch_copy = dict(ch)
                ch_copy["provenance"] = PROVENANCE_EXAMINER_APPROVED
                accepted_stains.append(ch_copy)
        elif dec == "Rejected":
            reg_copy["segmentation_status"] = STATUS_REJECTED
            reg_copy["examiner_review_status"] = "Examiner Rejected — Kept as Merged"
            unresolved_regions.append(reg_copy)
        elif dec == "Indeterminate":
            reg_copy["segmentation_status"] = STATUS_INDETERMINATE
            reg_copy["examiner_review_status"] = "Examiner Reviewed — Indeterminate"
            unresolved_regions.append(reg_copy)
        else:
            # If isolated stain and not explicitly rejected, it is naturally accepted
            if reg_copy["region_type"] == REGION_TYPE_ISOLATED:
                reg_copy["examiner_review_status"] = "Examiner Verified (Isolated)"
                accepted_stains.append(
                    {
                        "child_id": f"{rid}-S01",
                        "parent_region_id": rid,
                        "provenance": PROVENANCE_EXAMINER_APPROVED,
                        "area_px2": reg_copy["parent_area_px2"],
                        "area_mm2": reg_copy["parent_area_mm2"],
                        "perimeter_px": reg_copy["parent_perimeter_px"],
                        "minor_axis_px": reg_copy["parent_minor_axis_px"],
                        "major_axis_px": reg_copy["parent_major_axis_px"],
                        "minor_axis_mm": reg_copy["parent_minor_axis_mm"],
                        "major_axis_mm": reg_copy["parent_major_axis_mm"],
                        "aspect_ratio": reg_copy["parent_aspect_ratio"],
                        "circularity": reg_copy["parent_circularity"],
                        "eccentricity": reg_copy["parent_eccentricity"],
                        "orientation_deg": reg_copy["parent_orientation_deg"],
                        "centroid_px": reg_copy["parent_centroid_px"],
                        "bbox_px": reg_copy["parent_bbox_px"],
                        "contour_pts": reg_copy["contour_pts"],
                    }
                )
            else:
                unresolved_regions.append(reg_copy)

        updated_regions.append(reg_copy)

    # Re-render updated review overlay
    orig_bgr = None
    if "original_image_path" in analysis_record:
        try:
            orig_bgr = image_processing.load_and_validate_image(analysis_record["original_image_path"])
        except Exception:
            pass

    review_b64 = analysis_record.get("review_overlay_b64", "")
    if orig_bgr is not None:
        new_overlays = render_overlap_overlays(
            original_bgr=orig_bgr,
            regions=updated_regions,
            is_calibrated=analysis_record.get("is_calibrated", False),
            mm_per_pixel=analysis_record.get("mm_per_pixel"),
        )
        review_b64 = new_overlays["review_overlay_b64"]

    type_counts = {
        "isolated": sum(1 for r in updated_regions if r["region_type"] == REGION_TYPE_ISOLATED),
        "touching": sum(1 for r in updated_regions if r["region_type"] == REGION_TYPE_TOUCHING),
        "overlapping": sum(1 for r in updated_regions if r["region_type"] == REGION_TYPE_OVERLAPPING),
        "cluster": sum(1 for r in updated_regions if r["region_type"] == REGION_TYPE_CLUSTER),
        "irregular": sum(1 for r in updated_regions if r["region_type"] == REGION_TYPE_IRREGULAR),
        "indeterminate": sum(1 for r in updated_regions if r["region_type"] == REGION_TYPE_INDETERMINATE),
    }

    return {
        "engine_version": OVERLAP_ENGINE_VERSION,
        "provenance": PROVENANCE_EXAMINER_APPROVED,
        "is_calibrated": analysis_record.get("is_calibrated", False),
        "measurement_unit": analysis_record.get("measurement_unit", "px"),
        "mm_per_pixel": analysis_record.get("mm_per_pixel"),
        "total_regions": len(updated_regions),
        "type_counts": type_counts,
        "candidate_splits_count": sum(1 for r in updated_regions if len(r.get("candidate_children", [])) >= 2),
        "regions": updated_regions,
        "accepted_individual_stains": accepted_stains,
        "unresolved_merged_regions": unresolved_regions,
        "accepted_stains_count": len(accepted_stains),
        "unresolved_regions_count": len(unresolved_regions),
        "examiner_name": examiner_name or analysis_record.get("examiner_name"),
        "examiner_notes": examiner_notes or analysis_record.get("examiner_notes", ""),
        "detection_overlay_b64": analysis_record.get("detection_overlay_b64", ""),
        "segmentation_overlay_b64": analysis_record.get("segmentation_overlay_b64", ""),
        "review_overlay_b64": review_b64,
        "mandatory_scientific_disclaimers": get_mandatory_overlap_disclaimers(),
    }


def get_mandatory_overlap_disclaimers() -> List[str]:
    """Mandatory scientific limitations on overlapping bloodstain segmentation."""
    return [
        "1. Examiner-Assisted Decision Support Only: Proposed watershed separation boundaries and child stain dimensions are candidate suggestions requiring visual and physical verification by a qualified examiner.",
        "2. Absence of 3D Temporal Occlusion Evidence: 2D photograph segmentation cannot determine droplet deposition sequence or physical vertical layering (which stain struck first/second) without macroscopic drying rims, differential clotting, or biochemical evidence.",
        "3. Merged Region Integrity: Regions where automated separation is uncertain or rejected by the examiner MUST be documented as indeterminate or unresolved merged regions. Individual stain dimensions must never be calculated from manufactured boundaries.",
        "4. Area Conservation & Edge Erosion Disparity: Watershed algorithms partition contiguous pixels along a 1-pixel boundary line; slight discrepancies between parent area and the sum of child areas are tracked and must not be masked.",
    ]
