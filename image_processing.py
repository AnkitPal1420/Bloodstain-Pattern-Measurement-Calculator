"""
image_processing.py
===================
Computer Vision module for Bloodstain Pattern Analysis (Phase 2 Milestone).

This module will handle:
1. Loading stain images (JPEG, PNG).
2. Grayscale conversion, Gaussian blurring, and Otsu/adaptive thresholding.
3. Morphological operations to separate spines/satellite spatter from the core stain.
4. Contour detection and direct least-squares ellipse fitting (cv2.fitEllipse).
5. Extracting minor axis (Width) and major axis (Length).

Current Status:
---------------
Planned for Milestone 2.
This placeholder ensures clean architectural modularity while the initial
calculator engine is verified and tested.
"""

from typing import Any, Dict, Optional, Tuple


def preprocess_image(image_path: str) -> Any:
    """
    (Phase 2) Preprocess an input bloodstain image: grayscale, blur, threshold.

    Parameters:
        image_path: Path to the target image file.

    Raises:
        NotImplementedError: Reserved for Milestone 2.
    """
    raise NotImplementedError(
        "Image preprocessing is part of Milestone 2 (OpenCV integration). "
        "Complete and verify the basic calculator before activating image processing."
    )


def extract_stain_dimensions(image_array: Any) -> Tuple[float, float]:
    """
    (Phase 2) Fit an ellipse to the primary bloodstain and extract (width, length).

    Parameters:
        image_array: Preprocessed binary or grayscale image.

    Returns:
        Tuple[float, float]: (measured_width, measured_length) in pixels or calibrated units.

    Raises:
        NotImplementedError: Reserved for Milestone 2.
    """
    raise NotImplementedError(
        "Ellipse dimension extraction is part of Milestone 2 (OpenCV integration)."
    )
