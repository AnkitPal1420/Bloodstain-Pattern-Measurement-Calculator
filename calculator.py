"""
calculator.py
=============
Core calculation engine for Bloodstain Pattern Analysis (BPA).

This module calculates the approximate angle of impact of an idealized
elliptical bloodstain using its width and length.

Mathematical Principle:
-----------------------
When a spherical blood droplet strikes a flat surface at an angle, it forms an
ellipse. The angle of impact (alpha) is derived from right-triangle trigonometry:

    sin(alpha) = Width / Length
    alpha = arcsin(Width / Length)

Where:
    - Width  = Minor axis of the ellipse (transverse dimension)
    - Length = Major axis of the ellipse (direction of travel, excluding spines/tails)
    - alpha  = Angle of impact in degrees (0 deg < alpha <= 90 deg)

Forensic Note:
--------------
This calculation assumes an idealized ellipse on a smooth, non-porous surface.
It is an educational approximation and does not account for surface texture,
droplet volume, or fluid dynamic distortion.
"""

import math
from typing import Any, Dict, Tuple


class InvalidMeasurementError(ValueError):
    """Exception raised when stain measurements violate physical or geometric constraints."""
    pass


def validate_measurements(width: Any, length: Any) -> Tuple[float, float]:
    """
    Validate that width and length are valid numerical values and adhere
    to physical and geometric constraints of an elliptical stain.

    Parameters:
        width: The measured minor axis (width) of the bloodstain.
        length: The measured major axis (length) of the bloodstain.

    Returns:
        Tuple[float, float]: Validated (width, length) converted to floats.

    Raises:
        InvalidMeasurementError: If inputs are empty, non-numeric, zero,
                                 negative, NaN/infinite, or if width > length.
    """
    # 1. Check for empty or None inputs
    if width is None or length is None:
        raise InvalidMeasurementError("Measurements cannot be empty. Please provide both width and length.")

    # 2. Check for empty string inputs
    if isinstance(width, str) and not width.strip():
        raise InvalidMeasurementError("Width cannot be blank. Please enter a valid number.")
    if isinstance(length, str) and not length.strip():
        raise InvalidMeasurementError("Length cannot be blank. Please enter a valid number.")

    # 3. Convert to float and check for invalid numerical input
    try:
        w = float(width)
        l = float(length)
    except (ValueError, TypeError):
        raise InvalidMeasurementError(
            f"Invalid numerical input. Received width='{width}', length='{length}'. "
            "Please enter valid numeric values."
        )

    # 4. Check for NaN or infinite values
    if math.isnan(w) or math.isnan(l):
        raise InvalidMeasurementError("Measurements cannot be NaN (Not a Number).")
    if math.isinf(w) or math.isinf(l):
        raise InvalidMeasurementError("Measurements cannot be infinite.")

    # 5. Check for zero measurements
    if w == 0.0:
        raise InvalidMeasurementError("Width cannot be zero. A bloodstain must have a measurable width.")
    if l == 0.0:
        raise InvalidMeasurementError("Length cannot be zero. A bloodstain must have a measurable length.")

    # 6. Check for negative values
    if w < 0.0 or l < 0.0:
        raise InvalidMeasurementError(
            f"Measurements must be positive numbers. Received width={w}, length={l}."
        )

    # 7. Check geometric constraint: Width cannot exceed Length
    # In an elliptical bloodstain, the minor axis (width) is at most equal to
    # the major axis (length). A ratio > 1 would yield an impossible sine value (|sin| > 1).
    if w > l:
        raise InvalidMeasurementError(
            f"Width ({w}) cannot be greater than length ({l}). "
            "In an elliptical stain, the width corresponds to the minor axis and cannot exceed the major axis."
        )

    return w, l


def calculate_impact_angle(width: Any, length: Any) -> Dict[str, Any]:
    """
    Calculate the approximate impact angle (in degrees and radians) of an
    idealized elliptical bloodstain.

    Formula:
        angle_radians = arcsin(width / length)
        angle_degrees = degrees(angle_radians)

    Parameters:
        width: Minor axis dimension (e.g., in millimeters).
        length: Major axis dimension (e.g., in millimeters).

    Returns:
        Dict[str, Any]: Dictionary containing:
            - 'width': float
            - 'length': float
            - 'ratio': float (width / length, rounded to 4 decimals)
            - 'angle_degrees': float (impact angle in degrees, rounded to 2 decimals)
            - 'angle_radians': float (impact angle in radians, rounded to 4 decimals)
            - 'is_perpendicular': bool (True if angle is 90 deg, i.e., circular stain)
            - 'interpretation': str (educational summary of stain trajectory)
            - 'disclaimer': str (educational context warning)

    Raises:
        InvalidMeasurementError: If inputs fail validation.
    """
    w, l = validate_measurements(width, length)

    # Compute width-to-length ratio (W / L)
    ratio = w / l

    # arcsin requires ratio <= 1.0; floating-point tolerance handling:
    # If ratio is slightly above 1 due to IEEE 754 precision, clamp to 1.0.
    ratio_clamped = min(ratio, 1.0)

    # Calculate angle in radians and convert to degrees
    angle_rad = math.asin(ratio_clamped)
    angle_deg = math.degrees(angle_rad)

    # Determine if stain is circular (perpendicular impact at 90 degrees)
    is_perpendicular = math.isclose(angle_deg, 90.0, abs_tol=1e-5)

    if is_perpendicular:
        interpretation = (
            "A 90.0° impact angle indicates a perpendicular drop. "
            "The droplet fell straight down onto the surface, forming a circular stain (Width ≈ Length)."
        )
    elif angle_deg < 30.0:
        interpretation = (
            f"An acute impact angle of {angle_deg:.2f}° indicates an elongated, glancing impact. "
            "The droplet struck the surface at a shallow angle."
        )
    else:
        interpretation = (
            f"An estimated impact angle of {angle_deg:.2f}° indicates an oblique strike. "
            "The droplet struck at an intermediate incline."
        )

    return {
        "width": w,
        "length": l,
        "ratio": round(ratio, 4),
        "angle_degrees": round(angle_deg, 2),
        "angle_radians": round(angle_rad, 4),
        "is_perpendicular": is_perpendicular,
        "interpretation": interpretation,
        "disclaimer": (
            "Educational Estimate: This calculation assumes an idealized ellipse on a flat, "
            "smooth surface. Real-world physical factors (surface tension, spatter spines, "
            "porosity, velocity) are not modeled in this simple geometric formula."
        ),
    }
