"""
image_processing.py
===================
OpenCV Computer Vision and Optical Image Processing pipeline for Bloodstain Pattern Analysis (Phase 3).

This module handles:
1. Secure image acquisition and integrity validation.
2. Forensic image preprocessing (grayscale, Gaussian blur, Otsu vs Adaptive thresholding).
3. Morphological noise reduction and contour segmentation.
4. Direct least-squares ellipse fitting (cv2.fitEllipse) with strict axis validation (minor <= major).
5. Spatial pixel-to-millimeter calibration via reference scale bar.
6. Multi-layer visual overlay generation (contour, fitted ellipse, minor/major axes).
"""

import math
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
MAX_IMAGE_DIMENSION = 4096  # Max width or height in pixels to avoid decompression bombs
MAX_FILE_SIZE_BYTES = 16 * 1024 * 1024  # 16 MB


class ImageProcessingError(ValueError):
    """Exception raised for invalid, corrupted, or unprocessable image inputs."""
    pass


# ===========================================================================
# 1. Image Acquisition & Integrity Validation
# ===========================================================================


def is_allowed_file(filename: str) -> bool:
    """Validate that filename has an approved forensic image extension (jpg, jpeg, png)."""
    if not filename or "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def load_and_validate_image(source: Union[str, bytes, np.ndarray]) -> np.ndarray:
    """
    Read and validate an image from a filesystem path, raw bytes, or existing numpy array.

    Raises:
        ImageProcessingError: If the file is missing, empty, corrupted, unreadable,
                              or exceeds size bounds.
    """
    img: Optional[np.ndarray] = None

    if isinstance(source, np.ndarray):
        img = source
    elif isinstance(source, bytes):
        if len(source) == 0:
            raise ImageProcessingError("Uploaded image byte stream is empty.")
        if len(source) > MAX_FILE_SIZE_BYTES:
            raise ImageProcessingError(f"Image file size exceeds maximum permitted limit ({MAX_FILE_SIZE_BYTES // (1024*1024)} MB).")
        nparr = np.frombuffer(source, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    elif isinstance(source, str):
        if not os.path.exists(source):
            raise ImageProcessingError(f"Image file not found at path: {source}")
        if os.path.getsize(source) > MAX_FILE_SIZE_BYTES:
            raise ImageProcessingError(f"Image file size exceeds maximum permitted limit ({MAX_FILE_SIZE_BYTES // (1024*1024)} MB).")
        img = cv2.imread(source, cv2.IMREAD_COLOR)
    else:
        raise ImageProcessingError(f"Unsupported image source type: {type(source)}")

    if img is None or not isinstance(img, np.ndarray) or img.size == 0:
        raise ImageProcessingError("Corrupted or unreadable image file. Unable to decode pixel data.")

    h, w = img.shape[:2]
    if h == 0 or w == 0:
        raise ImageProcessingError("Image dimensions must be greater than zero.")
    if h > MAX_IMAGE_DIMENSION or w > MAX_IMAGE_DIMENSION:
        raise ImageProcessingError(f"Image dimensions ({w}x{h}) exceed maximum permitted size ({MAX_IMAGE_DIMENSION}px).")

    return img


# ===========================================================================
# 2. Image Preprocessing (Grayscale, Blur, Thresholding)
# ===========================================================================


def convert_to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert an input BGR or single-channel image into 8-bit grayscale."""
    if img is None or img.size == 0:
        raise ImageProcessingError("Cannot convert empty image to grayscale.")
    if len(img.shape) == 2:
        return img.copy()
    if len(img.shape) == 3:
        if img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    raise ImageProcessingError(f"Unexpected image shape: {img.shape}")


def apply_gaussian_blur(gray: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Apply Gaussian smoothing to reduce sensor noise and substrate grain.
    Kernel size must be an odd positive integer.
    """
    if kernel_size <= 0:
        raise ImageProcessingError(f"Gaussian blur kernel size must be positive, got {kernel_size}.")
    if kernel_size % 2 == 0:
        kernel_size += 1  # Force odd integer for OpenCV GaussianBlur
    return cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)


def apply_thresholding(
    blurred: np.ndarray,
    method: str = "otsu",
    invert: bool = True,
    block_size: int = 15,
    c_value: int = 2,
) -> np.ndarray:
    """
    Perform binary binarization using either Otsu's optimal global thresholding
    or adaptive Gaussian window thresholding.

    Parameters:
        blurred: 8-bit grayscale smoothed image.
        method: 'otsu' or 'adaptive'.
        invert: If True, foreground bloodstains become white (255) and background black (0).
        block_size: Neighborhood size for adaptive thresholding (odd int >= 3).
        c_value: Constant subtracted from weighted mean in adaptive thresholding.
    """
    if blurred is None or len(blurred.shape) != 2:
        raise ImageProcessingError("Input to thresholding must be a 2D grayscale image.")

    method_clean = method.strip().lower()

    if method_clean == "otsu":
        thresh_type = (cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY) + cv2.THRESH_OTSU
        _, binary = cv2.threshold(blurred, 0, 255, thresh_type)
        return binary

    elif method_clean == "adaptive":
        if block_size < 3:
            block_size = 3
        if block_size % 2 == 0:
            block_size += 1
        thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresh_type,
            block_size,
            c_value,
        )
        return binary

    else:
        raise ImageProcessingError(f"Unknown thresholding method '{method}'. Choose 'otsu' or 'adaptive'.")


# ===========================================================================
# 3. Morphological Filtering & Stain Segmentation
# ===========================================================================


def apply_morphology(
    binary: np.ndarray,
    open_kernel: int = 3,
    close_kernel: int = 3,
) -> np.ndarray:
    """
    Apply morphological opening (to suppress isolated pixel noise) and closing
    (to bridge minute internal voids), preserving core elliptical morphology.
    """
    result = binary.copy()
    if open_kernel > 1:
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, k_open)

    if close_kernel > 1:
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, k_close)

    return result


def detect_stain_contours(
    mask: np.ndarray,
    min_area: float = 30.0,
    max_area: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Identify and filter external stain contours from the binary segmentation mask.

    Returns a list of candidate dictionaries sorted by area descending:
        - 'index': int
        - 'area': float
        - 'perimeter': float
        - 'bbox': (x, y, w, h)
        - 'point_count': int
        - 'can_fit_ellipse': bool (True if point_count >= 5)
        - 'contour': np.ndarray
    """
    if mask is None or len(mask.shape) != 2:
        raise ImageProcessingError("Segmentation mask must be a 2D binary image.")

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: List[Dict[str, Any]] = []

    h, w = mask.shape
    total_image_area = float(h * w)
    effective_max_area = max_area if max_area is not None else (total_image_area * 0.95)

    for idx, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area < min_area or area > effective_max_area:
            continue

        point_count = len(c)
        bbox = cv2.boundingRect(c)
        perimeter = cv2.arcLength(c, True)

        candidates.append({
            "index": idx,
            "area": round(float(area), 2),
            "perimeter": round(float(perimeter), 2),
            "bbox": bbox,
            "point_count": point_count,
            "can_fit_ellipse": point_count >= 5,
            "contour": c,
        })

    # Sort descending by area (primary stain candidate first)
    candidates.sort(key=lambda item: item["area"], reverse=True)

    # Re-index neatly for the examiner
    for i, item in enumerate(candidates):
        item["candidate_id"] = i + 1

    return candidates


# ===========================================================================
# 4. Direct Least-Squares Ellipse Fitting (cv2.fitEllipse)
# ===========================================================================


def fit_stain_ellipse(contour: np.ndarray) -> Dict[str, Any]:
    """
    Fit an idealized mathematical ellipse to a bloodstain contour using OpenCV's
    direct least-squares formulation (cv2.fitEllipse).

    Mathematical & Forensic Constraint Handling:
    - Requires at least 5 non-collinear contour coordinates.
    - Guarantees minor axis (Width, W) <= major axis (Length, L).
    - Preserves ellipse center, major axis orientation, and aspect ratio.

    Raises:
        ImageProcessingError: If the contour has < 5 points or fails numerical fitting.
    """
    if contour is None or len(contour) < 5:
        raise ImageProcessingError(
            f"Ellipse fitting requires a minimum of 5 non-collinear contour points; received {len(contour) if contour is not None else 0}."
        )

    try:
        box = cv2.fitEllipse(contour)
    except cv2.error as cv_err:
        raise ImageProcessingError(f"OpenCV numerical ellipse fitting failed: {cv_err}")
    except Exception as exc:
        raise ImageProcessingError(f"Unexpected ellipse fitting failure: {exc}")

    (center_x, center_y), (axis_dim1, axis_dim2), angle_deg = box

    # OpenCV fitEllipse returns full axis lengths (diameters)
    minor_axis = min(axis_dim1, axis_dim2)
    major_axis = max(axis_dim1, axis_dim2)

    if minor_axis <= 0 or major_axis <= 0:
        raise ImageProcessingError("Fitted ellipse yielded non-positive geometric dimensions.")

    # Guard IEEE-754 precision boundary
    ratio = min(minor_axis / major_axis, 1.0)

    return {
        "center_x": round(float(center_x), 2),
        "center_y": round(float(center_y), 2),
        "minor_axis_px": round(float(minor_axis), 2),
        "major_axis_px": round(float(major_axis), 2),
        "ratio": round(float(ratio), 4),
        "orientation_deg": round(float(angle_deg), 2),
        "box": box,
        "point_count": len(contour),
        "contour_area": round(float(cv2.contourArea(contour)), 2),
    }


# ===========================================================================
# 5. Spatial Pixel-to-Millimeter Calibration
# ===========================================================================


def calculate_scale_factor(
    point1: Tuple[float, float],
    point2: Tuple[float, float],
    known_length_mm: float,
) -> Dict[str, Any]:
    """
    Derive the optical pixel-to-millimeter scale factor from two reference points
    on a calibrated scale bar within the target surface plane.

    Formula:
        pixel_distance = sqrt((x2 - x1)^2 + (y2 - y1)^2)
        mm_per_pixel = known_length_mm / pixel_distance
        pixels_per_mm = pixel_distance / known_length_mm
    """
    if known_length_mm <= 0:
        raise ImageProcessingError(f"Known scale bar length must be positive; received {known_length_mm} mm.")

    x1, y1 = point1
    x2, y2 = point2
    pixel_distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    if pixel_distance <= 0:
        raise ImageProcessingError("Calibration points are identical or have zero pixel distance.")

    mm_per_pixel = known_length_mm / pixel_distance
    pixels_per_mm = pixel_distance / known_length_mm

    warning: Optional[str] = None
    if pixel_distance < 20.0:
        warning = "Caution: Short calibration reference (< 20px) increases pixel discretization uncertainty."

    return {
        "pixel_distance": round(float(pixel_distance), 2),
        "known_length_mm": float(known_length_mm),
        "mm_per_pixel": round(float(mm_per_pixel), 6),
        "pixels_per_mm": round(float(pixels_per_mm), 4),
        "point1": (float(x1), float(y1)),
        "point2": (float(x2), float(y2)),
        "is_calibrated": True,
        "warning": warning,
    }


def apply_calibration(
    minor_axis_px: float,
    major_axis_px: float,
    mm_per_pixel: float,
) -> Dict[str, float]:
    """Convert pixel minor (Width) and major (Length) dimensions into physical millimeters."""
    if mm_per_pixel <= 0:
        raise ImageProcessingError("Scale factor (mm_per_pixel) must be positive.")
    width_mm = minor_axis_px * mm_per_pixel
    length_mm = major_axis_px * mm_per_pixel
    return {
        "width_mm": round(float(width_mm), 2),
        "length_mm": round(float(length_mm), 2),
    }


# ===========================================================================
# 6. Visual Overlays & Examiner Annotations
# ===========================================================================


def generate_ellipse_overlay(
    image: np.ndarray,
    contour: np.ndarray,
    ellipse_data: Dict[str, Any],
    calibration_data: Optional[Dict[str, Any]] = None,
    stain_label: str = "Stain-01",
    direction_angle_deg: Optional[float] = None,
) -> np.ndarray:
    """
    Render a forensic visualization overlay featuring:
    - Primary stain contour (Emerald Green).
    - Fitted least-squares ellipse (Crimson Red).
    - Major and minor axis crosslines with orientation vector.
    - Ellipse center crosshair.
    - Calibration scale bar line if active.
    - Information HUD card showing dimensions and impact angle.
    """
    vis = image.copy()
    box = ellipse_data.get("box")

    # 1. Draw outer contour in green
    cv2.drawContours(vis, [contour], -1, (40, 200, 40), 2)

    # 2. Draw fitted ellipse in crimson red
    if box:
        cv2.ellipse(vis, box, (20, 20, 220), 2)

        # 3. Calculate and draw Major and Minor Axis lines
        (cx, cy), (axis_w, axis_h), angle = box
        rad = math.radians(angle)

        # OpenCV fitEllipse angle convention:
        # axis_h is rotated by angle from vertical; axis_w is perpendicular
        # Major axis is the longer of axis_w and axis_h
        if axis_h >= axis_w:
            # Major axis along angle
            major_r = axis_h / 2.0
            minor_r = axis_w / 2.0
            maj_dx = -major_r * math.sin(rad)
            maj_dy = major_r * math.cos(rad)
            min_dx = minor_r * math.cos(rad)
            min_dy = minor_r * math.sin(rad)
        else:
            major_r = axis_w / 2.0
            minor_r = axis_h / 2.0
            maj_dx = major_r * math.cos(rad)
            maj_dy = major_r * math.sin(rad)
            min_dx = minor_r * math.sin(rad)
            min_dy = -minor_r * math.cos(rad)

        # Major axis line (Yellow)
        p_maj1 = (int(round(cx - maj_dx)), int(round(cy - maj_dy)))
        p_maj2 = (int(round(cx + maj_dx)), int(round(cy + maj_dy)))
        cv2.line(vis, p_maj1, p_maj2, (0, 240, 240), 2)

        # Minor axis line (Cyan)
        p_min1 = (int(round(cx - min_dx)), int(round(cy - min_dy)))
        p_min2 = (int(round(cx + min_dx)), int(round(cy + min_dy)))
        cv2.line(vis, p_min1, p_min2, (240, 240, 0), 2)

        # Center crosshair (White)
        cv2.drawMarker(
            vis,
            (int(round(cx)), int(round(cy))),
            (255, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=10,
            thickness=2,
        )

        # Examiner-supplied direction of travel vector (gamma in [0, 360) deg, +Y up -> -dy in image)
        if direction_angle_deg is not None:
            gamma_rad = math.radians(float(direction_angle_deg))
            arrow_r = max(axis_w, axis_h) * 0.72
            end_x = int(round(cx + arrow_r * math.cos(gamma_rad)))
            end_y = int(round(cy - arrow_r * math.sin(gamma_rad)))
            cv2.arrowedLine(
                vis,
                (int(round(cx)), int(round(cy))),
                (end_x, end_y),
                (255, 100, 255),
                2,
                tipLength=0.22,
            )

    # 4. Draw calibration reference if provided
    if calibration_data and calibration_data.get("is_calibrated"):
        p1 = tuple(map(int, map(round, calibration_data["point1"])))
        p2 = tuple(map(int, map(round, calibration_data["point2"])))
        cv2.line(vis, p1, p2, (255, 120, 0), 3)  # Bright blue line
        cv2.circle(vis, p1, 4, (0, 0, 255), -1)
        cv2.circle(vis, p2, 4, (0, 0, 255), -1)

    # 5. Overlay forensic HUD metadata box
    hud_x, hud_y = 12, 28
    hud_lines = [
        f"Stain: {stain_label}",
        f"Minor (W): {ellipse_data['minor_axis_px']:.1f} px",
        f"Major (L): {ellipse_data['major_axis_px']:.1f} px",
        f"Ratio (W/L): {ellipse_data['ratio']:.4f}",
    ]
    if calibration_data and calibration_data.get("is_calibrated"):
        mm_w = ellipse_data['minor_axis_px'] * calibration_data['mm_per_pixel']
        mm_l = ellipse_data['major_axis_px'] * calibration_data['mm_per_pixel']
        hud_lines.append(f"Calibrated W: {mm_w:.2f} mm")
        hud_lines.append(f"Calibrated L: {mm_l:.2f} mm")
    else:
        hud_lines.append("Optical Calibration: UNCALIBRATED (px mode)")

    if direction_angle_deg is not None:
        hud_lines.append(f"Supplied Travel Dir: {float(direction_angle_deg):.1f} deg")

    # Draw semi-transparent background for HUD
    box_w, box_h = 240, 20 + len(hud_lines) * 18
    overlay_box = vis.copy()
    cv2.rectangle(overlay_box, (hud_x - 6, hud_y - 18), (hud_x + box_w, hud_y + box_h - 18), (14, 21, 38), -1)
    cv2.addWeighted(overlay_box, 0.75, vis, 0.25, 0, vis)
    cv2.rectangle(vis, (hud_x - 6, hud_y - 18), (hud_x + box_w, hud_y + box_h - 18), (40, 55, 85), 1)

    for i, line in enumerate(hud_lines):
        color = (255, 255, 255)
        if "Minor" in line:
            color = (240, 240, 0)  # Cyan
        elif "Major" in line:
            color = (0, 240, 240)  # Yellow
        elif "Calibrated" in line:
            color = (50, 220, 50)  # Green
        elif "UNCALIBRATED" in line:
            color = (0, 180, 255)  # Orange
        cv2.putText(
            vis,
            line,
            (hud_x, hud_y + i * 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )

    return vis


# ===========================================================================
# 7. End-to-End Image Processing Pipeline
# ===========================================================================


def process_bloodstain_pipeline(
    image_source: Union[str, bytes, np.ndarray],
    threshold_method: str = "otsu",
    invert_binary: bool = True,
    blur_kernel: int = 5,
    min_area: float = 30.0,
    selected_candidate_id: int = 1,
    scale_bar_points: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    known_length_mm: Optional[float] = None,
    stain_label: str = "Stain-01",
    direction_angle_deg: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Execute the comprehensive forensic computer vision pipeline:
    1. Load and validate image.
    2. Convert to grayscale and Gaussian blur.
    3. Segment foreground stain using Otsu or Adaptive thresholding.
    4. Apply morphological filtering.
    5. Detect all candidate contours.
    6. Select target stain contour and fit least-squares ellipse.
    7. Optionally calculate spatial scale factor from reference points.
    8. Generate visual inspection overlay.

    Returns a rich results dictionary containing matrices, coordinates, and metrics.
    """
    # 1. Load & validate
    original_bgr = load_and_validate_image(image_source)

    # 2. Preprocess
    gray = convert_to_grayscale(original_bgr)
    blurred = apply_gaussian_blur(gray, kernel_size=blur_kernel)
    otsu_img = apply_thresholding(blurred, method="otsu", invert=invert_binary)
    adaptive_img = apply_thresholding(blurred, method="adaptive", invert=invert_binary)
    thresholded = apply_thresholding(blurred, method=threshold_method, invert=invert_binary)
    morphed = apply_morphology(thresholded, open_kernel=3, close_kernel=3)

    # 3. Detect contours
    candidates = detect_stain_contours(morphed, min_area=min_area)
    if not candidates:
        raise ImageProcessingError(
            "No stain contours detected above the minimum area threshold. "
            "Try adjusting the thresholding method or lowering the minimum area."
        )

    # 4. Select candidate (default: largest / candidate 1)
    target_candidate = None
    for c in candidates:
        if c.get("candidate_id") == selected_candidate_id:
            target_candidate = c
            break
    if target_candidate is None:
        target_candidate = candidates[0]

    if not target_candidate["can_fit_ellipse"]:
        raise ImageProcessingError(
            f"Selected candidate contour has only {target_candidate['point_count']} points; "
            "at least 5 points are required to fit an ellipse."
        )

    # 5. Fit ellipse
    ellipse_data = fit_stain_ellipse(target_candidate["contour"])

    # 6. Spatial Calibration (if reference provided)
    calibration_data = None
    if scale_bar_points and known_length_mm and known_length_mm > 0:
        p1, p2 = scale_bar_points
        calibration_data = calculate_scale_factor(p1, p2, known_length_mm)
        calibrated_axes = apply_calibration(
            ellipse_data["minor_axis_px"],
            ellipse_data["major_axis_px"],
            calibration_data["mm_per_pixel"],
        )
        ellipse_data["width_mm"] = calibrated_axes["width_mm"]
        ellipse_data["length_mm"] = calibrated_axes["length_mm"]
        ellipse_data["is_calibrated"] = True
    else:
        ellipse_data["width_mm"] = None
        ellipse_data["length_mm"] = None
        ellipse_data["is_calibrated"] = False

    # 7. Render Overlay
    overlay_bgr = generate_ellipse_overlay(
        image=original_bgr,
        contour=target_candidate["contour"],
        ellipse_data=ellipse_data,
        calibration_data=calibration_data,
        stain_label=stain_label,
        direction_angle_deg=direction_angle_deg,
    )

    return {
        "original_image": original_bgr,
        "grayscale_image": gray,
        "otsu_image": otsu_img,
        "adaptive_image": adaptive_img,
        "thresholded_image": thresholded,
        "morphed_mask": morphed,
        "overlay_image": overlay_bgr,
        "candidates": candidates,
        "selected_candidate": target_candidate,
        "ellipse": ellipse_data,
        "calibration": calibration_data,
        "image_shape": original_bgr.shape,
    }


def save_processed_views(
    pipeline_result: Dict[str, Any],
    output_dir: str,
    image_id: str,
) -> Dict[str, str]:
    """
    Save the intermediate and final processed images for a given image_id
    to the processed output directory, returning a dict of filenames.
    """
    os.makedirs(output_dir, exist_ok=True)
    safe_id = "".join(c for c in image_id if c.isalnum() or c in ("-", "_"))

    views = {
        "original": f"{safe_id}_original.png",
        "grayscale": f"{safe_id}_grayscale.png",
        "otsu": f"{safe_id}_otsu.png",
        "adaptive": f"{safe_id}_adaptive.png",
        "thresholded": f"{safe_id}_thresholded.png",
        "morphed": f"{safe_id}_morphed.png",
        "overlay": f"{safe_id}_overlay.png",
    }

    cv2.imwrite(os.path.join(output_dir, views["original"]), pipeline_result["original_image"])
    cv2.imwrite(os.path.join(output_dir, views["grayscale"]), pipeline_result["grayscale_image"])
    cv2.imwrite(os.path.join(output_dir, views["otsu"]), pipeline_result["otsu_image"])
    cv2.imwrite(os.path.join(output_dir, views["adaptive"]), pipeline_result["adaptive_image"])
    cv2.imwrite(os.path.join(output_dir, views["thresholded"]), pipeline_result["thresholded_image"])
    cv2.imwrite(os.path.join(output_dir, views["morphed"]), pipeline_result["morphed_mask"])
    cv2.imwrite(os.path.join(output_dir, views["overlay"]), pipeline_result["overlay_image"])

    return views

