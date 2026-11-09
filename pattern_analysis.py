"""
pattern_analysis.py
===================
Phase 5.1 — Advanced Bloodstain Pattern Classification & Analytical Decision Support Engine.

SCIENTIFIC & FORENSIC DESIGN PRINCIPLES:
1. Examiner-Assisted Decision Support Only:
   This module extracts quantitative morphological features from bloodstain imagery and
   organizes candidate pattern interpretations by comparing measured geometry and
   examiner-recorded observations against established BPA morphological characteristics.
2. No Automated Mechanism Declaration:
   The software NEVER declares a definitive deposition mechanism, weapon type, impact
   velocity, force, or event sequence from image appearance alone.
3. Strict Separation of Provenance Labels:
   Every datum managed by this module is explicitly tagged with one of five provenance labels:
   - "Automatically Measured" (computed deterministically from image contours/pixels)
   - "Examiner Observed" (visual scene/substrate traits recorded by the examiner)
   - "Examiner Supplied" (calibration scale, surface orientation, contextual metadata)
   - "Software Suggested" (rule-based preliminary candidate interpretations)
   - "Examiner Approved" (final interpretation explicitly reviewed and signed off by examiner)
4. Pixel vs. Metric Discipline:
   Millimeter (mm) and square-millimeter (mm²) values are computed ONLY when a verified
   positive scale calibration (mm_per_pixel > 0 and is_calibrated=True) is provided.
   Uncalibrated analyses report pixel (px / px²) and dimensionless ratios exclusively.
5. Transparent Contours & Non-Silent Rejection:
   Every detected external contour is accounted for. Contours that cannot be fitted or
   are flagged as potential speckle noise, image-border truncations, or coalesced/overlapping
   regions are preserved in `rejected_or_uncertain_detections` with explicit reasons.
"""

import base64
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

import image_processing
from image_processing import ImageProcessingError

PATTERN_ENGINE_VERSION = "BPMS Pattern Analysis Decision-Support Engine v5.1 (Experimental Rule-Based)"

# Provenance constants required by Section 6
PROVENANCE_AUTO_MEASURED = "Automatically Measured"
PROVENANCE_EXAMINER_OBSERVED = "Examiner Observed"
PROVENANCE_EXAMINER_SUPPLIED = "Examiner Supplied"
PROVENANCE_SOFTWARE_SUGGESTED = "Software Suggested"
PROVENANCE_EXAMINER_APPROVED = "Examiner Approved"

# Explicit system determination outcomes required by Section 5
OUTCOME_INSUFFICIENT = "Insufficient Information"
OUTCOME_INDETERMINATE = "Indeterminate — Examiner Review Required"
OUTCOME_MULTIPLE = "Multiple Possible Interpretations — Examiner Review Required"
OUTCOME_REVIEW_REQUIRED = "Examiner Review Required"


class PatternAnalysisError(ValueError):
    """Exception raised for invalid inputs to the pattern analysis module."""
    pass


# ===========================================================================
# 1. Structured BPA Pattern Taxonomy (Section 3)
# ===========================================================================

PATTERN_TAXONOMY: Dict[str, Dict[str, Any]] = {
    # A. Passive bloodstain patterns
    "Passive Drop": {
        "group_code": "A",
        "group_name": "Passive Bloodstain Patterns",
        "category_name": "Passive Drop",
        "description": (
            "Individual bloodstain formed by a droplet falling Primarily under the influence "
            "of gravity without significant external horizontal projection."
        ),
        "typical_characteristics": [
            "Single or sparse isolated stain(s)",
            "Moderate to high circularity on horizontal non-porous surfaces (Circularity > 0.70, Aspect Ratio near 1.0–1.3)",
            "High solidity (smooth perimeter on non-porous substrate; scalloping/spines increase on rough/porous surfaces)",
        ],
        "distinguishing_limitations": (
            "A single circular or near-circular drop cannot by itself establish drop height without "
            "known droplet volume and substrate calibration, nor distinguish passive dripping from "
            "a perpendicular low-velocity projection."
        ),
    },
    "Drip Pattern": {
        "group_code": "A",
        "group_name": "Passive Bloodstain Patterns",
        "category_name": "Drip Pattern",
        "description": (
            "Bloodstain pattern resulting from liquid blood dripping into liquid blood or "
            "forming a sequential linear/curved drip trail from a moving source under gravity."
        ),
        "typical_characteristics": [
            "Multiple spherical/elliptical stains arranged in a linear trail or clustered around a parent drip pool",
            "Secondary satellite spatter radiating around a central parent stain (drip into blood)",
            "Relatively uniform stain sizes along a linear path (drip trail)",
        ],
        "distinguishing_limitations": (
            "Linear drip trails can share spatial linearity with cast-off patterns; distinguishing "
            "them requires evaluating stain elongation, progressive aspect ratio change, and scene geometry."
        ),
    },
    "Pool": {
        "group_code": "A",
        "group_name": "Passive Bloodstain Patterns",
        "category_name": "Pool",
        "description": (
            "Accumulation of liquid blood on a surface that contours to surface topography "
            "and gravity rather than single-droplet impact dynamics."
        ),
        "typical_characteristics": [
            "Large contiguous stain area relative to spatter droplets",
            "Often accompanied by serum separation margins or peripheral drying rings",
            "Low stain count dominated by one or few large contiguous regions",
        ],
        "distinguishing_limitations": (
            "Image segmentation alone cannot distinguish a primary blood pool from a large "
            "saturation stain without substrate absorption inspection."
        ),
    },
    "Flow Pattern": {
        "group_code": "A",
        "group_name": "Passive Bloodstain Patterns",
        "category_name": "Flow Pattern",
        "description": (
            "Directional movement of liquid blood across a surface primarily under the influence "
            "of gravity or surface inclination after deposition."
        ),
        "typical_characteristics": [
            "High aspect ratio (elongated stream/channel morphology, Aspect Ratio > 2.5)",
            "Non-elliptical ribbon or tapering stream contour with variable width",
            "Orientation aligned with gravity or substrate slope",
        ],
        "distinguishing_limitations": (
            "Post-deposition flows may originate from impact spatter, wounds, or pools; "
            "orientation reflects gravity/surface tilt at the time of flow, not airborne trajectory."
        ),
    },
    "Saturation Pattern": {
        "group_code": "A",
        "group_name": "Passive Bloodstain Patterns",
        "category_name": "Saturation Pattern",
        "description": (
            "Accumulation of liquid blood absorbed into a porous or permeable substrate "
            "(such as fabric, carpet, or unsealed wood) via capillary wicking."
        ),
        "typical_characteristics": [
            "Large irregular stain boundary with diffuse or wicked margins",
            "Occurs on absorbent/porous substrate material",
            "Irregular contour perimeter (lower circularity and solidity due to weave/wicking)",
        ],
        "distinguishing_limitations": (
            "Requires examiner verification of substrate porosity/absorbency; optical contour shape "
            "is governed by capillary wicking rather than impact angle trigonometry."
        ),
    },
    # B. Impact-related patterns
    "Impact Spatter": {
        "group_code": "B",
        "group_name": "Impact-Related Patterns",
        "category_name": "Impact Spatter",
        "description": (
            "Bloodstain pattern resulting from an external force applied to a liquid blood source, "
            "dispersing multiple airborne droplets that strike a receiving surface at varying angles."
        ),
        "typical_characteristics": [
            "Multiple discrete elliptical/circular stains distributed across a radiating fan or cluster",
            "Wide distribution of droplet sizes (fine mist to millimeter-scale elliptical drops)",
            "Directional major axes tending to radiate from a common convergence region",
        ],
        "distinguishing_limitations": (
            "Stain size and spatial distribution alone CANNOT establish the specific mechanism "
            "(e.g., blunt trauma vs. gunshot vs. machinery) or impact velocity without independent "
            "physical and pathological evidence. Expirated blood and secondary satellite spatter "
            "can produce overlapping droplet size distributions."
        ),
    },
    "Cast-Off Pattern": {
        "group_code": "B",
        "group_name": "Impact-Related Patterns",
        "category_name": "Cast-Off Pattern",
        "description": (
            "Bloodstain pattern resulting from blood droplets released from an object in motion "
            "(centrifugal ejection along an arc)."
        ),
        "typical_characteristics": [
            "Multiple stains arranged in a linear or gently curving trail (high centroid linearity R²)",
            "Consistent orientation of major axes aligned along the linear path",
            "Progressive transition in aspect ratio (from more circular to more elongated) along the trail",
        ],
        "distinguishing_limitations": (
            "Linear arrangements of stains can also arise from drip trails or arterial projection; "
            "object dimensions or swing count cannot be inferred from a single linear trail without caution."
        ),
    },
    "Projected Blood Pattern": {
        "group_code": "B",
        "group_name": "Impact-Related Patterns",
        "category_name": "Projected Blood Pattern",
        "description": (
            "Bloodstain pattern produced by blood released under pressure or volume ejection "
            "(e.g., arterial breach, pressurized stream, or large-volume projection) rather than point impact."
        ),
        "typical_characteristics": [
            "Large-volume stains often accompanied by downward gravity spines/flows on vertical surfaces",
            "Arcuate, serpentine, or wave-like spatial distribution when source is moving or pulsatile",
            "Prominent spines, satellites, or connected volume deposits",
        ],
        "distinguishing_limitations": (
            "Cannot be attributed to arterial breach from photograph geometry alone without "
            "medical/autopsy correlation of vascular injury."
        ),
    },
    # C. Contact and transfer patterns
    "Contact Transfer": {
        "group_code": "C",
        "group_name": "Contact and Transfer Patterns",
        "category_name": "Contact Transfer",
        "description": (
            "Bloodstain resulting from direct physical contact between a blood-bearing surface "
            "and a second surface without significant lateral motion."
        ),
        "typical_characteristics": [
            "Irregular non-elliptical contour (low solidity or irregular boundary)",
            "Absence of radiating satellite spatter or airborne impact spines",
            "Often exhibits substrate texture dropout or irregular interior density",
        ],
        "distinguishing_limitations": (
            "Requires examiner evaluation of margin sharpness and absence of lateral smearing "
            "to distinguish simple contact from swipe/wipe."
        ),
    },
    "Swipe Pattern": {
        "group_code": "C",
        "group_name": "Contact and Transfer Patterns",
        "category_name": "Swipe Pattern",
        "description": (
            "Transfer of blood onto an unstained surface by a moving blood-bearing object."
        ),
        "typical_characteristics": [
            "Irregular elongated smear or streak boundary with feathered/striated termination margin",
            "Low solidity or non-elliptical contour geometry",
            "Examiner observation of lateral smearing/feathered edges without a pre-existing dried stain rim",
        ],
        "distinguishing_limitations": (
            "Distinguishing a swipe (moving bloody object) from a wipe (object moving through "
            "pre-existing wet blood) requires inspecting perimeter drying rims/skeletonization."
        ),
    },
    "Wipe Pattern": {
        "group_code": "C",
        "group_name": "Contact and Transfer Patterns",
        "category_name": "Wipe Pattern",
        "description": (
            "Alteration of a pre-existing bloodstain on a surface caused by an object moving through it."
        ),
        "typical_characteristics": [
            "Disrupted or smeared interior with a preserved outer skeletonized perimeter ring",
            "Irregular contour with visible smear striations or displacement",
            "Examiner observation of pre-existing stain alteration / skeletonized rim",
        ],
        "distinguishing_limitations": (
            "Requires visual confirmation of an altered pre-existing stain boundary (skeletonized rim) "
            "that automated contour thresholding alone cannot definitively verify."
        ),
    },
    "Patterned Transfer": {
        "group_code": "C",
        "group_name": "Contact and Transfer Patterns",
        "category_name": "Patterned Transfer",
        "description": (
            "Contact bloodstain that reproduces recognizable geometric elements (ridge detail, "
            "fabric weave, footwear tread, or tool geometry) of the blood-bearing object."
        ),
        "typical_characteristics": [
            "Repetitive geometric sub-elements, parallel ridges, or regular grid/tread periodicity",
            "Complex multi-component contour array or textured boundary",
            "Examiner observation of repeating object geometry or ridge/weave impression",
        ],
        "distinguishing_limitations": (
            "Source object comparison requires calibrated 1:1 scale photography and formal "
            "comparative examination."
        ),
    },
    # D. Altered or associated patterns
    "Expirated Blood Pattern": {
        "group_code": "D",
        "group_name": "Altered or Associated Patterns",
        "category_name": "Expirated Blood Pattern",
        "description": (
            "Bloodstain pattern resulting from blood forced by airflow out of the nose, mouth, "
            "or respiratory system."
        ),
        "typical_characteristics": [
            "Multiple small circular/elliptical droplets that may closely resemble impact spatter in size and shape",
            "Presence of air bubbles, bubble rings, central vacuoles, mucus strands, or saliva dilution",
        ],
        "distinguishing_limitations": (
            "CRITICAL: Expirated blood droplets can be morphologically indistinguishable from "
            "impact spatter based on contour dimensions alone. Confirmation requires observing "
            "air bubble rings/vacuoles, mucous strands, biochemical testing (e.g., amylase), or airway injury context."
        ),
    },
    "Void Pattern": {
        "group_code": "D",
        "group_name": "Altered or Associated Patterns",
        "category_name": "Void Pattern",
        "description": (
            "An absence of bloodstains in an otherwise continuous bloodstain pattern, indicating "
            "an intermediate object or body blocked droplet deposition at the time of the event."
        ),
        "typical_characteristics": [
            "Surrounding spatter or projected stains terminating abruptly along a demarked unstained boundary",
            "Spatial gap or low-density interior zone bounded by higher-density spatter",
            "Examiner observation of an unstained shadow/void boundary within a continuous pattern",
        ],
        "distinguishing_limitations": (
            "A void indicates the presence and approximate outline of an intercepting object during "
            "deposition, but rarely identifies the exact object without scene correlation."
        ),
    },
    "Environmentally / Surface-Altered Pattern": {
        "group_code": "D",
        "group_name": "Altered or Associated Patterns",
        "category_name": "Environmentally / Surface-Altered Pattern",
        "description": (
            "Bloodstain pattern whose appearance has been modified during or after formation by "
            "surface texture, porosity, moisture, dilution, insect activity, drying/clotting, or temperature."
        ),
        "typical_characteristics": [
            "Irregular or distorted stain perimeters (low circularity/solidity) on rough, textured, or wet surfaces",
            "Diluted halo rims, clotting inhomogeneity, or satellite artifacting from surface roughness",
            "Examiner observation of surface roughness, porosity, moisture, clotting, or insect artifacts",
        ],
        "distinguishing_limitations": (
            "Stains distorted by rough/porous substrates or environmental alteration MUST NOT be used "
            "for trigonometric impact angle (arcsin W/L) or 3D trajectory back-projection unless "
            "an undistorted elliptical boundary can be independently verified."
        ),
    },
}


# ===========================================================================
# 2. Core Mathematical Feature Calculations (Section 4)
# ===========================================================================


def compute_circularity(area: float, perimeter: float) -> float:
    """
    Compute dimensionless contour circularity (isoperimetric quotient):
        Circularity = (4 * pi * Area) / (Perimeter^2)

    For an ideal Euclidean circle, Circularity = 1.0.
    For elongated ellipses or irregular/spined contours, 0 < Circularity < 1.0.
    Note: Discrete pixel-grid perimeters (8-connected chain codes) on very small
    digital circles can slightly underestimate Euclidean circumference, yielding
    raw values up to ~1.05; values > 1.0 are clamped to 1.0.

    Raises:
        PatternAnalysisError: If area < 0 or perimeter <= 0 or inputs are non-finite.
    """
    if area is None or perimeter is None:
        raise PatternAnalysisError("Area and perimeter must not be None.")
    try:
        a = float(area)
        p = float(perimeter)
    except (TypeError, ValueError) as exc:
        raise PatternAnalysisError(f"Non-numeric input to compute_circularity: {exc}") from exc

    if not (math.isfinite(a) and math.isfinite(p)):
        raise PatternAnalysisError("Area and perimeter must be finite numbers.")
    if a <= 0.0:
        raise PatternAnalysisError(f"Area must be strictly positive, got {a}.")
    if p <= 0.0:
        raise PatternAnalysisError(f"Perimeter must be strictly positive, got {p}.")

    raw = (4.0 * math.pi * a) / (p * p)
    return round(min(max(raw, 0.0), 1.0), 6)


def compute_aspect_ratio(major_axis: float, minor_axis: float) -> float:
    """
    Compute dimensionless ellipse aspect ratio:
        Aspect Ratio = Major Axis (L) / Minor Axis (W) >= 1.0

    Raises:
        PatternAnalysisError: If minor_axis <= 0, major_axis <= 0, or major_axis < minor_axis.
    """
    try:
        maj = float(major_axis)
        min_ax = float(minor_axis)
    except (TypeError, ValueError) as exc:
        raise PatternAnalysisError(f"Non-numeric axis input to compute_aspect_ratio: {exc}") from exc

    if not (math.isfinite(maj) and math.isfinite(min_ax)):
        raise PatternAnalysisError("Major and minor axes must be finite numbers.")
    if min_ax <= 0.0 or maj <= 0.0:
        raise PatternAnalysisError(f"Ellipse axes must be strictly positive (got major={maj}, minor={min_ax}).")
    if maj + 1e-9 < min_ax:
        raise PatternAnalysisError(
            f"Major axis ({maj}) cannot be smaller than minor axis ({min_ax})."
        )

    eff_maj = max(maj, min_ax)
    return round(eff_maj / min_ax, 6)


def compute_solidity(contour_area: float, convex_hull_area: float) -> float:
    """
    Compute dimensionless contour solidity:
        Solidity = Contour Area / Convex Hull Area  in (0, 1]

    A convex stain has Solidity = 1.0. Stains with prominent spines, scallops,
    concavities, or coalesced overlaps exhibit Solidity < 1.0.

    Raises:
        PatternAnalysisError: If contour_area < 0, convex_hull_area <= 0, or contour_area > convex_hull_area.
    """
    try:
        ca = float(contour_area)
        ha = float(convex_hull_area)
    except (TypeError, ValueError) as exc:
        raise PatternAnalysisError(f"Non-numeric area input to compute_solidity: {exc}") from exc

    if not (math.isfinite(ca) and math.isfinite(ha)):
        raise PatternAnalysisError("Contour area and convex hull area must be finite.")
    if ca <= 0.0:
        raise PatternAnalysisError(f"Contour area must be strictly positive, got {ca}.")
    if ha <= 0.0:
        raise PatternAnalysisError(f"Convex hull area must be strictly positive, got {ha}.")
    if ca > ha * 1.0001:
        raise PatternAnalysisError(
            f"Contour area ({ca}) cannot exceed its convex hull area ({ha})."
        )

    return round(min(ca / ha, 1.0), 6)


def compute_eccentricity(major_axis: float, minor_axis: float) -> float:
    """
    Compute the first eccentricity (e) of an ellipse with major axis L and minor axis W:
        e = sqrt(1 - (W / L)^2)  in [0, 1)

    - For a circle (W == L): e = 0.0
    - For a 30-degree impact ellipse (W/L = 0.5): e = sqrt(1 - 0.25) = sqrt(3)/2 ≈ 0.866025
    - For a 45-degree impact ellipse (W/L = sqrt(2)/2): e = sqrt(1 - 0.5) ≈ 0.707107

    Raises:
        PatternAnalysisError: If minor_axis <= 0, major_axis <= 0, or minor_axis > major_axis.
    """
    try:
        maj = float(major_axis)
        min_ax = float(minor_axis)
    except (TypeError, ValueError) as exc:
        raise PatternAnalysisError(f"Non-numeric axis input to compute_eccentricity: {exc}") from exc

    if not (math.isfinite(maj) and math.isfinite(min_ax)):
        raise PatternAnalysisError("Major and minor axes must be finite numbers.")
    if min_ax <= 0.0 or maj <= 0.0:
        raise PatternAnalysisError(f"Ellipse axes must be strictly positive (got major={maj}, minor={min_ax}).")
    if min_ax > maj * 1.0001:
        raise PatternAnalysisError(
            f"Minor axis ({min_ax}) cannot exceed major axis ({maj})."
        )

    ratio = min(min_ax / maj, 1.0)
    ecc = math.sqrt(max(0.0, 1.0 - ratio * ratio))
    return round(ecc, 6)


# ===========================================================================
# 3. Image Quality Assessment & Multi-Stain Feature Extraction (Section 4)
# ===========================================================================


def evaluate_image_quality(gray_image: np.ndarray) -> Dict[str, Any]:
    """
    Assess grayscale evidence photograph quality prior to contour interpretation:
    - Resolution (width_px, height_px)
    - Contrast (standard deviation of pixel intensities)
    - Dynamic range (p99 - p1 intensity span)
    - Sharpness / focus proxy (variance of Laplacian)
    - High-frequency background noise estimate (MAD of 3x3 Laplacian)
    """
    if gray_image is None or not isinstance(gray_image, np.ndarray) or gray_image.size == 0:
        raise PatternAnalysisError("Cannot evaluate quality of an empty image.")

    h, w = gray_image.shape[:2]
    mean_intensity = float(np.mean(gray_image))
    contrast_std = float(np.std(gray_image))
    p1, p99 = np.percentile(gray_image, [1, 99])
    dynamic_range = float(p99 - p1)

    lap = cv2.Laplacian(gray_image, cv2.CV_64F)
    laplacian_var = float(np.var(lap))
    noise_mad = float(np.median(np.abs(lap - np.median(lap))))

    warnings: List[str] = []
    is_sufficient = True

    if min(h, w) < 80:
        warnings.append(
            f"Low spatial resolution ({w}x{h} px): fine spatter droplets and contour margins may suffer pixel discretization."
        )
        if min(h, w) < 40:
            is_sufficient = False

    if contrast_std < 8.0 or dynamic_range < 25.0:
        warnings.append(
            f"Low image contrast (std={contrast_std:.1f}, dynamic range={dynamic_range:.1f}): "
            "threshold segmentation between bloodstains and substrate is unreliable."
        )
        is_sufficient = False

    if noise_mad > 18.0:
        warnings.append(
            f"Elevated high-frequency background noise/texture (MAD={noise_mad:.1f}): "
            "substrate grain may introduce false micro-contours or distort stain perimeters."
        )

    return {
        "provenance": PROVENANCE_AUTO_MEASURED,
        "width_px": int(w),
        "height_px": int(h),
        "mean_intensity": round(mean_intensity, 2),
        "contrast_std": round(contrast_std, 2),
        "dynamic_range": round(dynamic_range, 2),
        "laplacian_variance": round(laplacian_var, 2),
        "noise_mad": round(noise_mad, 2),
        "is_sufficient_quality": is_sufficient,
        "quality_warnings": warnings,
    }


def _compute_spatial_and_cluster_metrics(
    accepted_stains: List[Dict[str, Any]],
    image_width_px: int,
    image_height_px: int,
    is_calibrated: bool,
    mm_per_pixel: Optional[float],
) -> Dict[str, Any]:
    """
    Compute pattern-wide spatial distribution, stain density, relative size distribution,
    shape variation, orientation dispersion, and centroid linearity R².
    """
    n = len(accepted_stains)
    if n == 0:
        return {
            "provenance": PROVENANCE_AUTO_MEASURED,
            "stain_count": 0,
            "spatial_distribution_type": "None Detected",
            "pattern_bounding_box_px": None,
            "pattern_envelope_area_px2": 0.0,
            "pattern_envelope_area_mm2": None,
            "stain_density_per_10k_px2": 0.0,
            "stain_density_per_cm2": None,
            "nearest_neighbor_mean_px": None,
            "nearest_neighbor_std_px": None,
            "nearest_neighbor_mean_mm": None,
            "centroid_linearity_r2": None,
            "mean_area_px2": None,
            "std_area_px2": None,
            "area_cv": None,
            "mean_area_mm2": None,
            "mean_circularity": None,
            "std_circularity": None,
            "mean_aspect_ratio": None,
            "std_aspect_ratio": None,
            "mean_solidity": None,
            "std_solidity": None,
            "mean_eccentricity": None,
            "orientation_dispersion_deg": None,
            "size_distribution_classes": {"fine_or_small": 0, "medium": 0, "large": 0},
        }

    areas_px = np.array([s["area_px2"] for s in accepted_stains], dtype=float)
    circs = np.array([s["circularity"] for s in accepted_stains], dtype=float)
    ars = np.array([s["aspect_ratio"] for s in accepted_stains], dtype=float)
    sols = np.array([s["solidity"] for s in accepted_stains], dtype=float)
    eccs = np.array([s["eccentricity"] for s in accepted_stains], dtype=float)
    centroids = np.array([[s["centroid_x_px"], s["centroid_y_px"]] for s in accepted_stains], dtype=float)

    min_x, min_y = np.min(centroids, axis=0)
    max_x, max_y = np.max(centroids, axis=0)
    span_w = max(float(max_x - min_x), 1.0)
    span_h = max(float(max_y - min_y), 1.0)
    envelope_area_px2 = max(span_w * span_h, float(np.sum(areas_px)))
    image_area_px2 = float(max(image_width_px * image_height_px, 1))

    density_per_10k_px2 = (n / image_area_px2) * 10000.0
    envelope_area_mm2: Optional[float] = None
    density_per_cm2: Optional[float] = None
    mean_area_mm2: Optional[float] = None

    if is_calibrated and mm_per_pixel and mm_per_pixel > 0:
        scale_sq = float(mm_per_pixel) ** 2
        envelope_area_mm2 = round(envelope_area_px2 * scale_sq, 3)
        image_area_cm2 = (image_area_px2 * scale_sq) / 100.0
        if image_area_cm2 > 0:
            density_per_cm2 = round(n / image_area_cm2, 4)
        mean_area_mm2 = round(float(np.mean(areas_px)) * scale_sq, 4)

    # Nearest-neighbor distances & spatial linearity R^2
    nn_mean_px: Optional[float] = None
    nn_std_px: Optional[float] = None
    nn_mean_mm: Optional[float] = None
    linearity_r2: Optional[float] = None

    if n >= 2:
        diff = centroids[:, np.newaxis, :] - centroids[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(dist_matrix, np.inf)
        nn_dists = np.min(dist_matrix, axis=1)
        nn_mean_px = round(float(np.mean(nn_dists)), 2)
        nn_std_px = round(float(np.std(nn_dists)), 2)
        if is_calibrated and mm_per_pixel and mm_per_pixel > 0:
            nn_mean_mm = round(nn_mean_px * float(mm_per_pixel), 3)

    if n >= 3:
        centered = centroids - np.mean(centroids, axis=0)
        cov = np.cov(centered, rowvar=False)
        eigvals = np.linalg.eigvalsh(cov)
        total_var = float(np.sum(eigvals))
        if total_var > 1e-9:
            linearity_r2 = round(float(np.max(eigvals) / total_var), 4)

    # Orientation dispersion (axial data: angles modulo 180 deg)
    orient_angles = [
        s["orientation_deg"] for s in accepted_stains if s.get("orientation_deg") is not None
    ]
    orient_dispersion_deg: Optional[float] = None
    if len(orient_angles) >= 2:
        doubled_rads = np.radians(np.array(orient_angles, dtype=float) * 2.0)
        r_bar = math.hypot(float(np.mean(np.cos(doubled_rads))), float(np.mean(np.sin(doubled_rads))))
        r_bar = min(max(r_bar, 1e-9), 1.0)
        circular_std_rad = math.sqrt(-2.0 * math.log(r_bar)) / 2.0
        orient_dispersion_deg = round(math.degrees(circular_std_rad), 2)

    mean_area = float(np.mean(areas_px))
    std_area = float(np.std(areas_px))
    area_cv = round(std_area / mean_area, 4) if mean_area > 0 else 0.0

    # Relative size distribution buckets (relative to median area or calibrated mm²)
    small_cnt = 0
    med_cnt = 0
    large_cnt = 0
    median_area = float(np.median(areas_px))
    for a_px in areas_px:
        if is_calibrated and mm_per_pixel and mm_per_pixel > 0:
            equiv_diam_mm = 2.0 * math.sqrt((a_px * (mm_per_pixel ** 2)) / math.pi)
            if equiv_diam_mm < 1.5:
                small_cnt += 1
            elif equiv_diam_mm <= 4.0:
                med_cnt += 1
            else:
                large_cnt += 1
        else:
            if a_px < 0.5 * median_area:
                small_cnt += 1
            elif a_px <= 2.0 * median_area:
                med_cnt += 1
            else:
                large_cnt += 1

    # Descriptive spatial distribution label
    if n == 1:
        spatial_type = "Single Isolated Stain"
    elif linearity_r2 is not None and linearity_r2 >= 0.88 and n >= 3:
        spatial_type = "Linear / Arcuate Alignment"
    elif nn_mean_px is not None and nn_mean_px < max(image_width_px, image_height_px) * 0.08 and n >= 6:
        spatial_type = "Dense Clustered Distribution"
    elif n >= 4:
        spatial_type = "Dispersed Multi-Stain Distribution"
    else:
        spatial_type = "Sparse Multi-Stain Group"

    return {
        "provenance": PROVENANCE_AUTO_MEASURED,
        "stain_count": int(n),
        "spatial_distribution_type": spatial_type,
        "pattern_bounding_box_px": [
            round(float(min_x), 1),
            round(float(min_y), 1),
            round(float(max_x - min_x), 1),
            round(float(max_y - min_y), 1),
        ],
        "pattern_envelope_area_px2": round(envelope_area_px2, 2),
        "pattern_envelope_area_mm2": envelope_area_mm2,
        "stain_density_per_10k_px2": round(density_per_10k_px2, 4),
        "stain_density_per_cm2": density_per_cm2,
        "nearest_neighbor_mean_px": nn_mean_px,
        "nearest_neighbor_std_px": nn_std_px,
        "nearest_neighbor_mean_mm": nn_mean_mm,
        "centroid_linearity_r2": linearity_r2,
        "mean_area_px2": round(mean_area, 2),
        "std_area_px2": round(std_area, 2),
        "area_cv": area_cv,
        "mean_area_mm2": mean_area_mm2,
        "mean_circularity": round(float(np.mean(circs)), 4),
        "std_circularity": round(float(np.std(circs)), 4),
        "mean_aspect_ratio": round(float(np.mean(ars)), 4),
        "std_aspect_ratio": round(float(np.std(ars)), 4),
        "mean_solidity": round(float(np.mean(sols)), 4),
        "std_solidity": round(float(np.std(sols)), 4),
        "mean_eccentricity": round(float(np.mean(eccs)), 4),
        "orientation_dispersion_deg": orient_dispersion_deg,
        "size_distribution_classes": {
            "fine_or_small": int(small_cnt),
            "medium": int(med_cnt),
            "large": int(large_cnt),
        },
    }


def _render_pattern_overlay_b64(
    original_bgr: np.ndarray,
    accepted_stains: List[Dict[str, Any]],
    rejected_contours_with_meta: List[Tuple[np.ndarray, Dict[str, Any]]],
    is_calibrated: bool,
    mm_per_pixel: Optional[float],
) -> str:
    """
    Render a multi-stain diagnostic overlay image encoded as a data:image/png;base64 URI:
    - Accepted stains: Emerald contour + Crimson fitted ellipse + Stain ID label
    - Overlapping/Irregular flagged stains: Amber contour
    - Rejected micro-contours / border truncations: Muted cyan/gray outline
    Never modifies `original_bgr`.
    """
    vis = original_bgr.copy()

    # Draw rejected/uncertain contours in subtle slate/amber
    for c_arr, r_meta in rejected_contours_with_meta:
        color = (0, 165, 255) if "overlap" in r_meta["reason"].lower() else (140, 140, 140)
        cv2.drawContours(vis, [c_arr], -1, color, 1)

    # Draw accepted stains
    for st in accepted_stains:
        c_arr = st.get("_contour")
        if c_arr is not None:
            contour_color = (0, 190, 255) if st.get("overlap_or_irregular_flag") else (50, 210, 80)
            cv2.drawContours(vis, [c_arr], -1, contour_color, 2)

        box = st.get("_box")
        if box is not None:
            try:
                cv2.ellipse(vis, box, (35, 35, 230), 2)
            except cv2.error:
                pass

        cx = int(round(st["centroid_x_px"]))
        cy = int(round(st["centroid_y_px"]))
        cv2.putText(
            vis,
            f"#{st['stain_index']}",
            (cx + 5, max(14, cy - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    hud_text = (
        f"Accepted Stains: {len(accepted_stains)} | "
        f"Rejected/Uncertain: {len(rejected_contours_with_meta)} | "
        + (f"Calibrated ({mm_per_pixel:.4f} mm/px)" if is_calibrated and mm_per_pixel else "Uncalibrated (px only)")
    )
    cv2.rectangle(vis, (6, 6), (min(vis.shape[1] - 6, 540), 32), (15, 23, 42), -1)
    cv2.putText(
        vis,
        hud_text,
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (226, 232, 240),
        1,
        cv2.LINE_AA,
    )

    ok, buf = cv2.imencode(".png", vis)
    if not ok:
        return ""
    b64_str = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/png;base64,{b64_str}"


def extract_pattern_features(
    image_source: Union[str, bytes, np.ndarray],
    is_calibrated: bool = False,
    mm_per_pixel: Optional[float] = None,
    threshold_method: str = "otsu",
    invert_binary: bool = True,
    blur_kernel: int = 5,
    min_area_px: float = 20.0,
) -> Dict[str, Any]:
    """
    Extract comprehensive bloodstain morphological and spatial pattern features
    without modifying the original image.

    Requirements satisfied:
    - Distinguishes pixel measurements (`_px`, `_px2`) from calibrated metric measurements (`_mm`, `_mm2`).
    - Never reports `mm` or `mm²` unless `is_calibrated=True` and `mm_per_pixel > 0`.
    - Never silently discards detected contours: all rejected or uncertain contours are
      recorded in `rejected_or_uncertain_detections` with explicit reasons.
    - Flags irregular stains, overlapping/coalesced contours, border-truncated contours,
      and insufficient image quality.
    """
    valid_calibration = bool(is_calibrated and mm_per_pixel is not None and float(mm_per_pixel) > 0.0)
    scale_mm_per_px: Optional[float] = round(float(mm_per_pixel), 6) if valid_calibration else None

    if is_calibrated and (mm_per_pixel is None or float(mm_per_pixel) <= 0.0):
        raise PatternAnalysisError(
            "When is_calibrated=True, a strictly positive mm_per_pixel scale factor must be provided."
        )

    original_bgr = image_processing.load_and_validate_image(image_source)
    gray = image_processing.convert_to_grayscale(original_bgr)
    quality = evaluate_image_quality(gray)

    h, w = gray.shape[:2]
    total_img_area = float(h * w)
    max_area_px = total_img_area * 0.92

    blurred = image_processing.apply_gaussian_blur(gray, kernel_size=blur_kernel)
    binary = image_processing.apply_thresholding(blurred, method=threshold_method, invert=invert_binary)
    morphed = image_processing.apply_morphology(binary, open_kernel=3, close_kernel=3)

    raw_contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    accepted_stains: List[Dict[str, Any]] = []
    rejected_or_uncertain: List[Dict[str, Any]] = []
    rejected_contours_for_overlay: List[Tuple[np.ndarray, Dict[str, Any]]] = []

    # Sort raw contours largest-first for deterministic indexing
    raw_contours_sorted = sorted(raw_contours, key=lambda c: cv2.contourArea(c), reverse=True)

    border_margin = 2
    for raw_idx, c in enumerate(raw_contours_sorted, start=1):
        area_px2 = float(cv2.contourArea(c))
        perim_px = float(cv2.arcLength(c, True))
        pt_count = int(len(c))
        bx, by, bw, bh = cv2.boundingRect(c)

        touches_border = (
            bx <= border_margin
            or by <= border_margin
            or (bx + bw) >= (w - border_margin)
            or (by + bh) >= (h - border_margin)
        )

        # 1. Sub-threshold micro-contours / sensor speckle
        if area_px2 < min_area_px:
            r_entry = {
                "raw_contour_id": raw_idx,
                "area_px2": round(area_px2, 2),
                "perimeter_px": round(perim_px, 2),
                "bbox_px": [int(bx), int(by), int(bw), int(bh)],
                "status": "Excluded — Sub-Threshold Speckle / Micro-Contour",
                "reason": (
                    f"Contour area ({area_px2:.1f} px²) is below minimum analytical threshold "
                    f"({min_area_px:.1f} px²); excluded from ellipse fitting to prevent sensor noise bias."
                ),
            }
            rejected_or_uncertain.append(r_entry)
            rejected_contours_for_overlay.append((c, r_entry))
            continue

        # 2. Full-frame background inversion artifact
        if area_px2 > max_area_px:
            r_entry = {
                "raw_contour_id": raw_idx,
                "area_px2": round(area_px2, 2),
                "perimeter_px": round(perim_px, 2),
                "bbox_px": [int(bx), int(by), int(bw), int(bh)],
                "status": "Excluded — Background Frame Envelope",
                "reason": "Contour spans >92% of image frame area (background boundary artifact).",
            }
            rejected_or_uncertain.append(r_entry)
            continue

        # 3. Insufficient vertices for 5-parameter conic section fit
        if pt_count < 5 or perim_px <= 0.0:
            r_entry = {
                "raw_contour_id": raw_idx,
                "area_px2": round(area_px2, 2),
                "perimeter_px": round(perim_px, 2),
                "bbox_px": [int(bx), int(by), int(bw), int(bh)],
                "status": "Excluded — Insufficient Contour Vertices",
                "reason": f"Contour has {pt_count} vertices (< 5 required for least-squares ellipse fit).",
            }
            rejected_or_uncertain.append(r_entry)
            rejected_contours_for_overlay.append((c, r_entry))
            continue

        hull = cv2.convexHull(c)
        hull_area_px2 = float(cv2.contourArea(hull))
        if hull_area_px2 <= 0.0:
            r_entry = {
                "raw_contour_id": raw_idx,
                "area_px2": round(area_px2, 2),
                "perimeter_px": round(perim_px, 2),
                "bbox_px": [int(bx), int(by), int(bw), int(bh)],
                "status": "Excluded — Degenerate Collinear Hull",
                "reason": "Convex hull area is zero (collinear contour points).",
            }
            rejected_or_uncertain.append(r_entry)
            continue

        # Fit ellipse via image_processing.fit_stain_ellipse
        try:
            el = image_processing.fit_stain_ellipse(c)
        except ImageProcessingError as exc:
            r_entry = {
                "raw_contour_id": raw_idx,
                "area_px2": round(area_px2, 2),
                "perimeter_px": round(perim_px, 2),
                "bbox_px": [int(bx), int(by), int(bw), int(bh)],
                "status": "Excluded — Ellipse Fit Numerical Failure",
                "reason": str(exc),
            }
            rejected_or_uncertain.append(r_entry)
            rejected_contours_for_overlay.append((c, r_entry))
            continue

        minor_px = float(el["minor_axis_px"])
        major_px = float(el["major_axis_px"])
        circularity = compute_circularity(area_px2, perim_px)
        aspect_ratio = compute_aspect_ratio(major_px, minor_px)
        solidity = compute_solidity(area_px2, hull_area_px2)
        eccentricity = compute_eccentricity(major_px, minor_px)

        # Compute moments for true area centroid
        moments = cv2.moments(c)
        if moments["m00"] > 0:
            cx_px = float(moments["m10"] / moments["m00"])
            cy_px = float(moments["m01"] / moments["m00"])
        else:
            cx_px = float(el["center_x"])
            cy_px = float(el["center_y"])

        # Detect overlapping/coalesced or non-elliptical irregular contours
        # Compare actual contour area against idealized fitted ellipse area (pi * a * b)
        ideal_ellipse_area = math.pi * (major_px / 2.0) * (minor_px / 2.0)
        ellipse_area_discrepancy = (
            abs(area_px2 - ideal_ellipse_area) / max(ideal_ellipse_area, 1.0)
        )

        # Check convexity defects for multi-lobe / overlapping constriction necks
        concavity_defect_count = 0
        try:
            hull_indices = cv2.convexHull(c, returnPoints=False)
            if hull_indices is not None and len(hull_indices) >= 4:
                defects = cv2.convexityDefects(c, hull_indices)
                if defects is not None:
                    for d_row in defects:
                        flat_d = np.asarray(d_row).ravel()
                        if len(flat_d) >= 4:
                            depth_px = float(flat_d[3]) / 256.0
                            if depth_px > max(3.5, 0.12 * minor_px):
                                concavity_defect_count += 1
        except (cv2.error, IndexError, TypeError, ValueError):
            pass

        caution_flags: List[str] = []
        if touches_border:
            caution_flags.append("Touches Image Frame Border (Potentially Truncated)")
        if solidity < 0.82 or concavity_defect_count >= 2:
            caution_flags.append(
                f"Low Solidity ({solidity:.2f}) / Concavity Pinch ({concavity_defect_count} deep defects) — "
                "Possible Overlapping/Coalesced Stains, Prominent Spines, or Irregular Transfer"
            )
        if ellipse_area_discrepancy > 0.35:
            caution_flags.append(
                f"Non-Elliptical Contour Geometry (Area vs. Fitted Ellipse discrepancy = {ellipse_area_discrepancy*100:.1f}%)"
            )

        if caution_flags:
            rejected_or_uncertain.append(
                {
                    "raw_contour_id": raw_idx,
                    "area_px2": round(area_px2, 2),
                    "perimeter_px": round(perim_px, 2),
                    "bbox_px": [int(bx), int(by), int(bw), int(bh)],
                    "status": "Flagged for Examiner Review — Uncertain / Irregular / Possible Overlap",
                    "reason": "; ".join(caution_flags),
                }
            )

        stain_idx = len(accepted_stains) + 1

        # Metric values computed ONLY if valid_calibration is True
        area_mm2: Optional[float] = None
        perim_mm: Optional[float] = None
        minor_mm: Optional[float] = None
        major_mm: Optional[float] = None
        if valid_calibration and scale_mm_per_px is not None:
            area_mm2 = round(area_px2 * (scale_mm_per_px ** 2), 4)
            perim_mm = round(perim_px * scale_mm_per_px, 3)
            minor_mm = round(minor_px * scale_mm_per_px, 3)
            major_mm = round(major_px * scale_mm_per_px, 3)

        # Morphological shape descriptor for transparent rule evaluation
        if solidity < 0.78 or ellipse_area_discrepancy > 0.40:
            shape_class = "Irregular / Non-Elliptical / Possible Overlap"
        elif aspect_ratio < 1.20 and circularity >= 0.75:
            shape_class = "Circular / Near-Circular"
        elif aspect_ratio <= 2.20:
            shape_class = "Moderately Elliptical"
        else:
            shape_class = "Highly Elongated / Linear"

        apparent_angle_deg = round(
            math.degrees(math.asin(min(max(minor_px / major_px, 0.0), 1.0))), 2
        )

        accepted_stains.append(
            {
                "stain_index": stain_idx,
                "raw_contour_id": raw_idx,
                "provenance": PROVENANCE_AUTO_MEASURED,
                "centroid_x_px": round(cx_px, 2),
                "centroid_y_px": round(cy_px, 2),
                "bbox_px": [int(bx), int(by), int(bw), int(bh)],
                "area_px2": round(area_px2, 2),
                "area_mm2": area_mm2,
                "perimeter_px": round(perim_px, 2),
                "perimeter_mm": perim_mm,
                "convex_hull_area_px2": round(hull_area_px2, 2),
                "minor_axis_px": round(minor_px, 2),
                "major_axis_px": round(major_px, 2),
                "minor_axis_mm": minor_mm,
                "major_axis_mm": major_mm,
                "width_to_length_ratio": round(minor_px / major_px, 4),
                "apparent_impact_angle_deg": apparent_angle_deg,
                "aspect_ratio": round(aspect_ratio, 4),
                "circularity": round(circularity, 4),
                "solidity": round(solidity, 4),
                "eccentricity": round(eccentricity, 4),
                "orientation_deg": round(float(el["orientation_deg"]), 2),
                "concavity_defect_count": concavity_defect_count,
                "touches_border": touches_border,
                "overlap_or_irregular_flag": bool(caution_flags),
                "caution_notes": caution_flags,
                "shape_classification": shape_class,
                "_contour": c,
                "_box": el["box"],
            }
        )

    overlay_b64 = _render_pattern_overlay_b64(
        original_bgr=original_bgr,
        accepted_stains=accepted_stains,
        rejected_contours_with_meta=rejected_contours_for_overlay,
        is_calibrated=valid_calibration,
        mm_per_pixel=scale_mm_per_px,
    )

    # Strip internal numpy/OpenCV objects before returning JSON-serializable stain dicts
    serializable_stains: List[Dict[str, Any]] = []
    for st in accepted_stains:
        clean_st = {k: v for k, v in st.items() if not k.startswith("_")}
        serializable_stains.append(clean_st)

    summary_metrics = _compute_spatial_and_cluster_metrics(
        accepted_stains=serializable_stains,
        image_width_px=w,
        image_height_px=h,
        is_calibrated=valid_calibration,
        mm_per_pixel=scale_mm_per_px,
    )

    return {
        "engine_version": PATTERN_ENGINE_VERSION,
        "provenance": PROVENANCE_AUTO_MEASURED,
        "is_calibrated": valid_calibration,
        "measurement_unit": "mm (Calibrated Metric)" if valid_calibration else "px (Uncalibrated Pixel Units)",
        "mm_per_pixel": scale_mm_per_px,
        "image_quality": quality,
        "total_raw_contours_detected": len(raw_contours_sorted),
        "accepted_stain_count": len(serializable_stains),
        "rejected_or_uncertain_count": len(rejected_or_uncertain),
        "stains": serializable_stains,
        "rejected_or_uncertain_detections": rejected_or_uncertain,
        "pattern_summary": summary_metrics,
        "overlay_b64": overlay_b64,
    }


# ===========================================================================
# 4. Examiner Observation Normalization (Section 6)
# ===========================================================================


def normalize_examiner_observations(raw_obs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Structure and tag examiner-recorded observations and contextual parameters,
    clearly distinguishing Examiner Observed vs. Examiner Supplied fields.
    """
    obs = raw_obs or {}

    def _clean(key: str, default: str = "Not Recorded") -> str:
        val = obs.get(key)
        if val is None or not str(val).strip():
            return default
        return str(val).strip()

    return {
        "suspected_category": {
            "value": _clean("suspected_category", "Unspecified"),
            "provenance": PROVENANCE_EXAMINER_OBSERVED,
        },
        "observed_morphology": {
            "value": _clean("observed_morphology", "Not Recorded"),
            "provenance": PROVENANCE_EXAMINER_OBSERVED,
        },
        "surface_material": {
            "value": _clean("surface_material", "Not Recorded"),
            "provenance": PROVENANCE_EXAMINER_SUPPLIED,
        },
        "surface_orientation": {
            "value": _clean("surface_orientation", "Not Recorded"),
            "provenance": PROVENANCE_EXAMINER_SUPPLIED,
        },
        "surface_texture": {
            "value": _clean("surface_texture", "Not Recorded"),
            "provenance": PROVENANCE_EXAMINER_OBSERVED,
        },
        "visible_distortion": {
            "value": _clean("visible_distortion", "None Observed"),
            "provenance": PROVENANCE_EXAMINER_OBSERVED,
        },
        "overlapping_stains": {
            "value": _clean("overlapping_stains", "Not Recorded"),
            "provenance": PROVENANCE_EXAMINER_OBSERVED,
        },
        "air_bubbles_or_vacuoles": {
            "value": _clean("air_bubbles_or_vacuoles", "Not Observed"),
            "provenance": PROVENANCE_EXAMINER_OBSERVED,
        },
        "feathered_or_skeletonized_margins": {
            "value": _clean("feathered_or_skeletonized_margins", "Not Observed"),
            "provenance": PROVENANCE_EXAMINER_OBSERVED,
        },
        "void_shadow_observed": {
            "value": _clean("void_shadow_observed", "Not Observed"),
            "provenance": PROVENANCE_EXAMINER_OBSERVED,
        },
        "environmental_observations": {
            "value": _clean("environmental_observations", "None Recorded"),
            "provenance": PROVENANCE_EXAMINER_OBSERVED,
        },
        "examination_notes": {
            "value": _clean("examination_notes", ""),
            "provenance": PROVENANCE_EXAMINER_OBSERVED,
        },
    }


# ===========================================================================
# 5. Explainable Rule-Based Classification-Support Engine (Section 5)
# ===========================================================================


def generate_classification_suggestions(
    feature_result: Dict[str, Any],
    examiner_observations: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Evaluate extracted morphological features (`feature_result`) alongside structured
    examiner observations (`examiner_observations`) to produce transparent, explainable
    preliminary pattern suggestions.

    SCIENTIFIC SAFEGUARDS:
    1. Never declares a single definitive deposition mechanism.
    2. Does not output fabricated probabilities or unvalidated ML scores.
    3. Explicitly returns `Insufficient Information`, `Indeterminate — Examiner Review Required`,
       or `Multiple Possible Interpretations — Examiner Review Required`.
    4. Lists specific supporting features, contradicting features, and missing information
       for every candidate category.
    """
    if not feature_result or "pattern_summary" not in feature_result:
        raise PatternAnalysisError("Valid feature extraction results are required for classification support.")

    norm_obs = normalize_examiner_observations(examiner_observations)
    summary = feature_result["pattern_summary"]
    quality = feature_result.get("image_quality", {})
    stains = feature_result.get("stains", [])
    rejected = feature_result.get("rejected_or_uncertain_detections", [])
    is_cal = bool(feature_result.get("is_calibrated"))

    n = int(summary.get("stain_count", 0))
    mean_circ = summary.get("mean_circularity")
    mean_ar = summary.get("mean_aspect_ratio")
    mean_sol = summary.get("mean_solidity")
    area_cv = summary.get("area_cv")
    lin_r2 = summary.get("centroid_linearity_r2")
    orient_disp = summary.get("orientation_dispersion_deg")

    irregular_count = sum(1 for s in stains if s.get("overlap_or_irregular_flag"))
    irregular_ratio = (irregular_count / n) if n > 0 else 0.0

    # Extract examiner observation strings in lowercase for transparent rule matching
    obs_morph = norm_obs["observed_morphology"]["value"].lower()
    obs_mat = norm_obs["surface_material"]["value"].lower()
    obs_orient = norm_obs["surface_orientation"]["value"].lower()
    obs_tex = norm_obs["surface_texture"]["value"].lower()
    obs_dist = norm_obs["visible_distortion"]["value"].lower()
    obs_overlap = norm_obs["overlapping_stains"]["value"].lower()
    obs_bubbles = norm_obs["air_bubbles_or_vacuoles"]["value"].lower()
    obs_margins = norm_obs["feathered_or_skeletonized_margins"]["value"].lower()
    obs_void = norm_obs["void_shadow_observed"]["value"].lower()
    obs_env = norm_obs["environmental_observations"]["value"].lower()
    obs_susp = norm_obs["suspected_category"]["value"]

    # Check global missing information
    global_missing: List[str] = []
    if not is_cal:
        global_missing.append(
            "Metric scale calibration (mm/px) is not applied; absolute droplet diameters (mm) and areas (mm²) remain unknown."
        )
    if norm_obs["surface_material"]["value"] == "Not Recorded":
        global_missing.append(
            "Receiving surface material/porosity is not recorded (porosity and roughness strongly alter stain margins)."
        )
    if norm_obs["surface_orientation"]["value"] == "Not Recorded":
        global_missing.append(
            "Receiving surface orientation (floor vs. vertical wall vs. inclined) is not recorded."
        )

    # Immediate Insufficient Information check when image quality fails or zero stains exist
    if not quality.get("is_sufficient_quality", True) or n == 0:
        reason_str = (
            "Zero valid bloodstain contours met analytical fitting criteria."
            if n == 0
            else "; ".join(quality.get("quality_warnings", ["Insufficient image contrast/resolution."]))
        )
        return {
            "engine_version": PATTERN_ENGINE_VERSION,
            "provenance": PROVENANCE_SOFTWARE_SUGGESTED,
            "overall_determination": OUTCOME_INSUFFICIENT,
            "determination_summary": (
                f"Analytical classification support cannot suggest candidate patterns due to insufficient "
                f"evidence data: {reason_str}"
            ),
            "candidate_suggestions": [],
            "excluded_or_contradicted_categories": [],
            "global_missing_information": global_missing,
            "mandatory_scientific_disclaimers": _get_mandatory_disclaimers(),
        }

    candidates: List[Dict[str, Any]] = []

    def _add_candidate(
        cat_key: str,
        supporting: List[str],
        contradicting: List[str],
        missing: List[str],
        priority_rank: int,
    ) -> None:
        if not supporting:
            return
        tax = PATTERN_TAXONOMY[cat_key]
        if len(supporting) >= 2 and len(contradicting) == 0:
            strength = "Consistent Morphological & Observational Characteristics"
        elif len(supporting) >= len(contradicting):
            strength = "Plausible Candidate — Additional Context or Verification Required"
        else:
            strength = "Partial Consistency — Significant Contradicting/Ambiguous Features Present"

        candidates.append(
            {
                "category_name": tax["category_name"],
                "group_code": tax["group_code"],
                "group_name": tax["group_name"],
                "description": tax["description"],
                "provenance": PROVENANCE_SOFTWARE_SUGGESTED,
                "evidence_indicator": strength,
                "supporting_features": supporting,
                "contradicting_features": contradicting,
                "missing_information": missing,
                "distinguishing_limitations": tax["distinguishing_limitations"],
                "_rank": priority_rank + (len(supporting) * 2) - (len(contradicting) * 2),
            }
        )

    # --- Evaluate Category 1: Passive Drop ---
    p_supp, p_contra, p_miss = [], [], list(global_missing)
    if n <= 3 and mean_circ is not None and mean_circ >= 0.72 and mean_ar is not None and mean_ar <= 1.30:
        p_supp.append(
            f"Automatically Measured: {n} isolated circular/near-circular stain(s) "
            f"(mean circularity = {mean_circ:.3f}, mean aspect ratio = {mean_ar:.2f})."
        )
    if mean_sol is not None and mean_sol >= 0.88 and n <= 3:
        p_supp.append(f"Automatically Measured: High contour solidity ({mean_sol:.3f}) indicating smooth perimeter.")
    if "passive" in obs_morph or "circular drop" in obs_morph:
        p_supp.append(f"Examiner Observed: Morphology recorded as '{norm_obs['observed_morphology']['value']}'.")
    if n > 6:
        p_contra.append(f"Automatically Measured: Multi-stain count ({n} stains) is atypical for a single isolated passive drop.")
    if mean_ar is not None and mean_ar > 1.50:
        p_contra.append(f"Automatically Measured: Elongated mean aspect ratio ({mean_ar:.2f}) indicates oblique strike or directional motion.")
    _add_candidate("Passive Drop", p_supp, p_contra, p_miss, priority_rank=5)

    # --- Evaluate Category 2: Drip Pattern ---
    d_supp, d_contra, d_miss = [], [], list(global_missing)
    if n >= 3 and lin_r2 is not None and lin_r2 >= 0.85 and mean_ar is not None and mean_ar <= 1.45:
        d_supp.append(
            f"Automatically Measured: {n} stains arranged along a linear alignment "
            f"(centroid linearity R² = {lin_r2:.3f}) with near-circular/moderately elliptical shapes (mean AR = {mean_ar:.2f})."
        )
    if n >= 3 and area_cv is not None and area_cv > 1.1 and mean_circ is not None and mean_circ >= 0.65:
        d_supp.append(
            f"Automatically Measured: Central parent stain with surrounding smaller satellite droplets (area CV = {area_cv:.2f})."
        )
    if "drip" in obs_morph or "satellite" in obs_morph:
        d_supp.append(f"Examiner Observed: Morphology notes '{norm_obs['observed_morphology']['value']}'.")
    if mean_ar is not None and mean_ar > 2.2:
        d_contra.append(f"Automatically Measured: High elongation (mean AR = {mean_ar:.2f}) is more typical of low-angle spatter, cast-off, or flow.")
    _add_candidate("Drip Pattern", d_supp, d_contra, d_miss, priority_rank=5)

    # --- Evaluate Category 3: Impact Spatter ---
    i_supp, i_contra, i_miss = [], [], list(global_missing)
    if n >= 3:
        i_supp.append(f"Automatically Measured: Multiple discrete bloodstain contours detected (n = {n}).")
    if n >= 3 and mean_sol is not None and mean_sol >= 0.82:
        i_supp.append(
            f"Automatically Measured: Well-formed elliptical/circular droplet contours (mean solidity = {mean_sol:.3f}, mean AR = {mean_ar:.2f})."
        )
    if n >= 3 and area_cv is not None and area_cv >= 0.20:
        i_supp.append(
            f"Automatically Measured: Variation in droplet sizes across pattern (area CV = {area_cv:.2f})."
        )
    if "spatter" in obs_morph or "radiating" in obs_morph or "elliptical" in obs_morph:
        i_supp.append(f"Examiner Observed: Morphology recorded as '{norm_obs['observed_morphology']['value']}'.")
    if lin_r2 is not None and lin_r2 >= 0.94 and n >= 4:
        i_contra.append(
            f"Automatically Measured: High single-line collinearity (R² = {lin_r2:.3f}) warrants comparing against Cast-Off or Drip Trail."
        )
    if "bubble" in obs_bubbles or "vacuole" in obs_bubbles or "yes" in obs_bubbles:
        i_contra.append(
            "Examiner Observed: Air bubbles / vacuoles noted — Expirated Blood Pattern must be actively evaluated."
        )
    i_miss.append(
        "Inspection for air bubbles/mucus strands and medical/scene context to differentiate impact spatter from expirated or projected blood."
    )
    _add_candidate("Impact Spatter", i_supp, i_contra, i_miss, priority_rank=6)

    # --- Evaluate Category 4: Cast-Off Pattern ---
    c_supp, c_contra, c_miss = [], [], list(global_missing)
    if n >= 3 and lin_r2 is not None and lin_r2 >= 0.85:
        c_supp.append(
            f"Automatically Measured: Linear/arcuate spatial alignment across {n} stains (centroid linearity R² = {lin_r2:.3f})."
        )
    if n >= 3 and orient_disp is not None and orient_disp <= 28.0:
        c_supp.append(
            f"Automatically Measured: Coherent major-axis orientation alignment along trail (angular dispersion = {orient_disp:.1f}°)."
        )
    if "cast-off" in obs_morph or "linear trail" in obs_morph or "arc" in obs_morph:
        c_supp.append(f"Examiner Observed: Morphology recorded as '{norm_obs['observed_morphology']['value']}'.")
    if n >= 4 and lin_r2 is not None and lin_r2 < 0.60:
        c_contra.append(
            f"Automatically Measured: Wide 2D spatial dispersion (linearity R² = {lin_r2:.3f}) contradicts a single linear cast-off arc."
        )
    _add_candidate("Cast-Off Pattern", c_supp, c_contra, c_miss, priority_rank=5)

    # --- Evaluate Category 5: Flow Pattern / Pool / Saturation ---
    if mean_ar is not None and mean_ar >= 2.60:
        f_supp = [
            f"Automatically Measured: Highly elongated contour geometry (mean aspect ratio = {mean_ar:.2f}, eccentricity = {summary.get('mean_eccentricity'):.3f})."
        ]
        f_contra = []
        if "vertical" in obs_orient or "incline" in obs_orient or "flow" in obs_morph:
            f_supp.append(
                f"Examiner Supplied/Observed: Surface orientation ('{norm_obs['surface_orientation']['value']}') or flow morphology supports gravity-driven flow."
            )
        if n > 10 and mean_sol is not None and mean_sol > 0.90:
            f_contra.append("Automatically Measured: High count of regular ellipses may represent acute-angle spatter rather than continuous liquid flow.")
        _add_candidate("Flow Pattern", f_supp, f_contra, list(global_missing), priority_rank=4)

    if "porous" in obs_mat or "fabric" in obs_mat or "carpet" in obs_mat or "absorb" in obs_tex or "saturat" in obs_morph:
        sat_supp = [
            f"Examiner Supplied/Observed: Absorbent/porous substrate ('{norm_obs['surface_material']['value']}', texture='{norm_obs['surface_texture']['value']}')."
        ]
        if irregular_ratio > 0.3 or (mean_sol is not None and mean_sol < 0.85):
            sat_supp.append(f"Automatically Measured: Irregular contour perimeter (mean solidity = {mean_sol:.3f}) consistent with capillary wicking.")
        _add_candidate("Saturation Pattern", sat_supp, [], list(global_missing), priority_rank=5)

    if (n <= 2 and summary.get("mean_area_px2", 0) > (0.12 * quality["width_px"] * quality["height_px"])) or "pool" in obs_morph:
        pool_supp = ["Large contiguous stain area dominating the field of view."]
        if "pool" in obs_morph or "serum" in obs_morph:
            pool_supp.append(f"Examiner Observed: '{norm_obs['observed_morphology']['value']}'.")
        _add_candidate("Pool", pool_supp, [], list(global_missing), priority_rank=4)

    # --- Evaluate Category 6: Contact / Swipe / Wipe / Patterned Transfer ---
    if irregular_ratio >= 0.35 or (mean_sol is not None and mean_sol < 0.82) or any(
        kw in obs_morph or kw in obs_margins
        for kw in ("transfer", "swipe", "wipe", "smear", "feather", "skeleton", "ridge", "weave", "tread")
    ):
        ct_supp = []
        if irregular_ratio >= 0.35 or (mean_sol is not None and mean_sol < 0.82):
            ct_supp.append(
                f"Automatically Measured: Non-elliptical/irregular contour geometry (mean solidity = {mean_sol:.3f}, "
                f"{irregular_count}/{n} contours flagged as non-elliptical/irregular)."
            )
        if "transfer" in obs_morph or "contact" in obs_morph:
            ct_supp.append(f"Examiner Observed: Contact/transfer morphology noted ('{norm_obs['observed_morphology']['value']}').")
        _add_candidate(
            "Contact Transfer",
            ct_supp,
            [],
            list(global_missing) + ["Verify whether lateral motion (feathered edge vs. skeletonized rim) occurred during contact."],
            priority_rank=4,
        )

        if "feather" in obs_margins or "swipe" in obs_morph or "smear" in obs_morph or (mean_ar is not None and mean_ar >= 1.8 and mean_sol is not None and mean_sol < 0.82):
            sw_supp = list(ct_supp)
            if "feather" in obs_margins or "swipe" in obs_morph:
                sw_supp.append(
                    f"Examiner Observed: Feathered/striated margin or swipe morphology ('{norm_obs['feathered_or_skeletonized_margins']['value']}')."
                )
            _add_candidate(
                "Swipe Pattern",
                sw_supp,
                ["Cannot rule out Wipe Pattern unless absence of a pre-existing skeletonized stain rim is verified."],
                list(global_missing),
                priority_rank=5,
            )

        if "skeleton" in obs_margins or "wipe" in obs_morph or "altered rim" in obs_margins:
            wp_supp = list(ct_supp) + [
                f"Examiner Observed: Skeletonized perimeter rim or wipe alteration ('{norm_obs['feathered_or_skeletonized_margins']['value']}')."
            ]
            _add_candidate("Wipe Pattern", wp_supp, [], list(global_missing), priority_rank=6)

        if any(kw in obs_morph for kw in ("patterned", "ridge", "fabric", "weave", "tread", "impression")):
            pt_supp = list(ct_supp) + [
                f"Examiner Observed: Repetitive geometric/object impression noted ('{norm_obs['observed_morphology']['value']}')."
            ]
            _add_candidate("Patterned Transfer", pt_supp, [], list(global_missing), priority_rank=6)

    # --- Evaluate Category 7: Expirated Blood, Void, and Environmentally/Surface-Altered ---
    if n >= 3 or "bubble" in obs_bubbles or "vacuole" in obs_bubbles or "expir" in obs_morph or "mucus" in obs_morph:
        exp_supp = []
        exp_contra = []
        if "bubble" in obs_bubbles or "vacuole" in obs_bubbles or "yes" in obs_bubbles or "expir" in obs_morph:
            exp_supp.append(
                f"Examiner Observed: Air bubble rings / vacuoles or respiratory indicators noted ('{norm_obs['air_bubbles_or_vacuoles']['value']}')."
            )
        elif n >= 3 and mean_circ is not None and mean_circ >= 0.55:
            exp_supp.append(
                f"Automatically Measured: Multi-droplet distribution (n = {n}) shares geometric overlap with fine/medium spatter droplets."
            )
            exp_contra.append(
                "No air bubble vacuoles or mucus strands have been recorded by the examiner."
            )
        _add_candidate(
            "Expirated Blood Pattern",
            exp_supp,
            exp_contra,
            list(global_missing) + ["Microscopic/macroscopic inspection for air-bubble vacuoles, dilution halos, or amylase/airway evidence."],
            priority_rank=6 if ("bubble" in obs_bubbles or "yes" in obs_bubbles) else 3,
        )

    if "void" in obs_void or "yes" in obs_void or "shadow" in obs_void or "void" in obs_morph:
        _add_candidate(
            "Void Pattern",
            [f"Examiner Observed: Unstained void/interception boundary recorded ('{norm_obs['void_shadow_observed']['value']}')."],
            [],
            list(global_missing),
            priority_rank=6,
        )

    if (
        len(rejected) > 0
        or irregular_ratio > 0.25
        or any(kw in obs_tex or kw in obs_dist or kw in obs_env for kw in ("rough", "textured", "porous", "wet", "dilut", "clot", "insect", "distort", "overlap"))
        or "yes" in obs_overlap
    ):
        alt_supp = []
        if len(rejected) > 0 or irregular_ratio > 0.25:
            alt_supp.append(
                f"Automatically Measured: {len(rejected)} contour(s) flagged as uncertain/rejected and "
                f"{irregular_count}/{n} accepted stain(s) exhibit irregular/non-elliptical or overlapping boundaries."
            )
        if norm_obs["visible_distortion"]["value"] != "None Observed":
            alt_supp.append(f"Examiner Observed: Visible distortion recorded ('{norm_obs['visible_distortion']['value']}').")
        if norm_obs["environmental_observations"]["value"] != "None Recorded":
            alt_supp.append(f"Examiner Observed: Environmental/surface condition ('{norm_obs['environmental_observations']['value']}').")
        _add_candidate(
            "Environmentally / Surface-Altered Pattern",
            alt_supp,
            [],
            list(global_missing),
            priority_rank=5,
        )

    # If the examiner explicitly recorded a suspected category in the taxonomy that wasn't already added, include it for review
    if obs_susp in PATTERN_TAXONOMY and not any(c["category_name"] == obs_susp for c in candidates):
        _add_candidate(
            obs_susp,
            [f"Examiner Observed: Examiner recorded suspected category '{obs_susp}' for analytical comparison."],
            ["Automated contour metrics alone do not uniquely isolate this category without examiner contextual verification."],
            list(global_missing),
            priority_rank=4,
        )

    # Sort candidates by rank descending
    candidates.sort(key=lambda item: item["_rank"], reverse=True)
    for c in candidates:
        c.pop("_rank", None)

    # Determine overall outcome (NEVER forces a single definitive classification)
    if len(candidates) == 0:
        overall_outcome = OUTCOME_INDETERMINATE
        summary_text = (
            "Extracted contour metrics do not clearly align with a specific pattern archetype. "
            "Outcome is Indeterminate — Examiner Review Required."
        )
    elif len(candidates) == 1:
        overall_outcome = OUTCOME_REVIEW_REQUIRED
        summary_text = (
            f"Preliminary analytical comparison identified 1 candidate category ({candidates[0]['category_name']}), "
            "subject to mandatory examiner verification of substrate and scene context."
        )
    else:
        overall_outcome = OUTCOME_MULTIPLE
        top_names = ", ".join(c["category_name"] for c in candidates[:4])
        summary_text = (
            f"Multiple Possible Interpretations identified ({len(candidates)} plausible candidates: {top_names}). "
            "Morphological overlap between categories requires independent examiner evaluation of physical substrate, "
            "biochemical, and scene context."
        )

    return {
        "engine_version": PATTERN_ENGINE_VERSION,
        "provenance": PROVENANCE_SOFTWARE_SUGGESTED,
        "overall_determination": overall_outcome,
        "determination_summary": summary_text,
        "candidate_suggestions": candidates,
        "global_missing_information": global_missing,
        "mandatory_scientific_disclaimers": _get_mandatory_disclaimers(),
    }


def _get_mandatory_disclaimers() -> List[str]:
    """Return mandatory scientific limitation statements required by Sections 1, 5, and 10."""
    return [
        "1. Examiner-Assisted Decision Support Only: Candidate pattern classifications are preliminary software suggestions generated by an experimental rule-based engine and DO NOT constitute a forensic conclusion until reviewed and approved by a qualified examiner.",
        "2. Mechanism & Velocity Non-Specificity: Bloodstain contour morphology, droplet size, and spatial density alone CANNOT establish the specific deposition mechanism (e.g., blunt impact vs. firearm vs. machinery), impact velocity, applied force, weapon type, or exact chronological sequence of events.",
        "3. Morphological Overlap Across Categories: Different physical mechanisms (such as impact spatter, expirated blood, and secondary satellite spatter; or cast-off trails and drip trails; or swipe and wipe transfers) can produce overlapping 2D contour geometries.",
        "4. Substrate & Calibration Dependence: Rough, porous, absorbent, or inclined receiving surfaces distort stain perimeters and invalidate ideal elliptical impact-angle assumptions. Uncalibrated photographs provide dimensionless ratios and pixel dimensions only.",
    ]
