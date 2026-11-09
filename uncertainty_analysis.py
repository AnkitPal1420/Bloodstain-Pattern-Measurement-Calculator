"""
uncertainty_analysis.py
======================
Core Mathematical Engine for Phase 5.3: Advanced Uncertainty & Sensitivity Analysis.
Forensic Laboratory Suite — Bloodstain Pattern Measurement Calculator.

This module provides:
1. Measurement uncertainty assessment and validation (length, width, calibration, ellipse fit).
2. Analytical first-order error propagation for impact angle:
       theta = arcsin(W / L)
   using exact partial derivatives and covariance handling.
3. High-performance, vectorized Monte Carlo uncertainty simulation with reproducible seed,
   customizable distributions (Gaussian / Uniform), empirical 95% simulation intervals,
   and invalid-draw frequency tracking.
4. One-At-A-Time (OAT) parameter perturbation sensitivity analysis, normalized elasticity
   indices (S_W, S_L), and parameter influence ranking.
5. Scientific Matplotlib figure rendering for Monte Carlo histograms and Tornado sensitivity charts.

Scientific & Forensic Principles:
---------------------------------
- Never silently correct invalid measurements (W > L, W <= 0, L <= 0).
- Explicitly identify missing uncertainty values as "Unavailable / Not Provided" rather
  than fabricating numbers.
- Scale Invariance Note: When both axes are scaled by the same isotropic pixel-to-mm
  calibration factor s, s cancels in the dimensionless ratio (W_mm / L_mm = W_px / L_px).
  Consequently, isotropic scale uncertainty does NOT propagate into the calculated impact angle.
- Simulation Disclaimer: A simulation-derived 95% interval reflects parameter dispersion
  under the chosen distributional assumptions; it is NOT an absolute probability that a
  specific physical deposition mechanism occurred.
"""

import io
import base64
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class UncertaintyValidationError(ValueError):
    """Raised when uncertainty analysis inputs fail geometric, physical, or numerical constraints."""
    pass


def validate_uncertainty_inputs(
    width: Any,
    length: Any,
    width_uncertainty: Optional[Any] = None,
    length_uncertainty: Optional[Any] = None,
    correlation: Optional[Any] = 0.0,
    scale_factor: Optional[Any] = 1.0,
    scale_uncertainty: Optional[Any] = 0.0,
) -> Dict[str, Any]:
    """
    Validate primary stain measurements and associated uncertainty parameters.

    Returns:
        Dict[str, Any] with sanitized float values and availability flags.

    Raises:
        UncertaintyValidationError: If primary measurements violate physical or geometric constraints.
    """
    # 1. Primary measurement validation
    if width is None or length is None:
        raise UncertaintyValidationError("Measurements cannot be empty. Please provide both width and length.")

    if isinstance(width, str) and not width.strip():
        raise UncertaintyValidationError("Width cannot be blank.")
    if isinstance(length, str) and not length.strip():
        raise UncertaintyValidationError("Length cannot be blank.")

    try:
        w = float(width)
        l = float(length)
    except (ValueError, TypeError):
        raise UncertaintyValidationError(f"Invalid numerical input. Received width='{width}', length='{length}'.")

    if math.isnan(w) or math.isnan(l) or math.isinf(w) or math.isinf(l):
        raise UncertaintyValidationError("Measurements cannot be NaN or infinite.")

    if w <= 0.0:
        raise UncertaintyValidationError(f"Width must be strictly positive (> 0). Received width={w}.")
    if l <= 0.0:
        raise UncertaintyValidationError(f"Length must be strictly positive (> 0). Received length={l}.")

    if w > l:
        raise UncertaintyValidationError(
            f"Width ({w}) cannot exceed length ({l}). In an elliptical bloodstain, the minor axis "
            "cannot be larger than the major axis. Ratio > 1 violates the arcsin domain."
        )

    # 2. Scale factor validation
    try:
        s = float(scale_factor) if scale_factor is not None else 1.0
    except (ValueError, TypeError):
        s = 1.0
    if s <= 0.0 or math.isnan(s) or math.isinf(s):
        s = 1.0

    # 3. Uncertainty parameters (may be None / omitted)
    w_unc: Optional[float] = None
    if width_uncertainty is not None and str(width_uncertainty).strip() != "":
        try:
            val = float(width_uncertainty)
        except (ValueError, TypeError):
            raise UncertaintyValidationError(f"Invalid width uncertainty value: '{width_uncertainty}'.")
        if not (math.isnan(val) or math.isinf(val)):
            if val < 0.0:
                raise UncertaintyValidationError(f"Width uncertainty cannot be negative. Received {val}.")
            w_unc = val

    l_unc: Optional[float] = None
    if length_uncertainty is not None and str(length_uncertainty).strip() != "":
        try:
            val = float(length_uncertainty)
        except (ValueError, TypeError):
            raise UncertaintyValidationError(f"Invalid length uncertainty value: '{length_uncertainty}'.")
        if not (math.isnan(val) or math.isinf(val)):
            if val < 0.0:
                raise UncertaintyValidationError(f"Length uncertainty cannot be negative. Received {val}.")
            l_unc = val

    s_unc: float = 0.0
    if scale_uncertainty is not None and str(scale_uncertainty).strip() != "":
        try:
            val = float(scale_uncertainty)
            if not (math.isnan(val) or math.isinf(val)) and val >= 0.0:
                s_unc = val
        except (ValueError, TypeError):
            s_unc = 0.0

    # 4. Correlation coefficient validation (-1.0 <= rho <= 1.0)
    rho = 0.0
    if correlation is not None and str(correlation).strip() != "":
        try:
            val = float(correlation)
        except (ValueError, TypeError):
            raise UncertaintyValidationError(f"Invalid correlation coefficient: '{correlation}'.")
        if not (math.isnan(val) or math.isinf(val)):
            if not (-1.0 <= val <= 1.0):
                raise UncertaintyValidationError(f"Correlation coefficient must be in [-1.0, 1.0]. Received {val}.")
            rho = val

    return {
        "width": w,
        "length": l,
        "width_uncertainty": w_unc,
        "length_uncertainty": l_unc,
        "has_uncertainty": (w_unc is not None and l_unc is not None),
        "correlation": rho,
        "scale_factor": s,
        "scale_uncertainty": s_unc,
    }


def propagate_impact_angle_uncertainty(
    width: float,
    length: float,
    width_uncertainty: Optional[float],
    length_uncertainty: Optional[float],
    correlation: float = 0.0,
    scale_factor: float = 1.0,
    scale_uncertainty: float = 0.0,
) -> Dict[str, Any]:
    """
    Perform analytical first-order error propagation on impact angle:
        theta = arcsin(W / L)

    Partial derivatives:
        d(theta)/dW = 1 / sqrt(L^2 - W^2)
        d(theta)/dL = -W / (L * sqrt(L^2 - W^2))

    Variance formula:
        sigma_theta^2 = (d_theta/dW)^2 * sigma_W^2 + (d_theta/dL)^2 * sigma_L^2
                        + 2 * (d_theta/dW) * (d_theta/dL) * cov(W, L)

    Returns:
        Dict[str, Any] with baseline angle, derivatives, propagated standard uncertainty (deg),
        and analytical 95.4% (2-sigma) bounds.
    """
    validated = validate_uncertainty_inputs(
        width=width,
        length=length,
        width_uncertainty=width_uncertainty,
        length_uncertainty=length_uncertainty,
        correlation=correlation,
        scale_factor=scale_factor,
        scale_uncertainty=scale_uncertainty,
    )
    w = validated["width"]
    l = validated["length"]
    w_unc = validated["width_uncertainty"]
    l_unc = validated["length_uncertainty"]
    rho = validated["correlation"]
    s = validated["scale_factor"]
    s_unc = validated["scale_uncertainty"]

    ratio = w / l
    ratio_clamped = min(ratio, 1.0)
    theta_rad = math.asin(ratio_clamped)
    theta_deg = math.degrees(theta_rad)

    # Physical dimensions in calibrated units
    w_phys = w * s
    l_phys = l * s

    # Physical dimension uncertainties (accounting for scale uncertainty)
    if w_unc is not None:
        w_phys_unc = math.sqrt((s * w_unc) ** 2 + (w * s_unc) ** 2)
    else:
        w_phys_unc = None

    if l_unc is not None:
        l_phys_unc = math.sqrt((s * l_unc) ** 2 + (l * s_unc) ** 2)
    else:
        l_phys_unc = None

    # Check for missing uncertainty information
    if w_unc is None or l_unc is None:
        return {
            "status": "unavailable",
            "message": "Measurement uncertainties for width and/or length were not provided.",
            "width": w,
            "length": l,
            "ratio": round(ratio, 4),
            "angle_degrees": round(theta_deg, 2),
            "angle_radians": round(theta_rad, 4),
            "d_theta_dw": None,
            "d_theta_dl": None,
            "sigma_theta_deg": None,
            "lower_bound_1sigma_deg": None,
            "upper_bound_1sigma_deg": None,
            "lower_bound_2sigma_deg": None,
            "upper_bound_2sigma_deg": None,
            "scale_invariance_note": (
                "Isotropic scale factor cancels in the W/L ratio; scale uncertainty does not affect "
                "the dimensionless impact angle."
            ),
            "physical_dimensions": {
                "width_mm": round(w_phys, 4),
                "length_mm": round(l_phys, 4),
                "width_uncertainty_mm": None,
                "length_uncertainty_mm": None,
            },
        }

    # Evaluate partial derivatives
    # Handle singularity at W == L (theta = 90 deg)
    diff_sq = l * l - w * w
    if diff_sq <= 1e-9:
        # Near-perpendicular impact (circular stain)
        # Numerical safeguard to prevent division by zero
        sqrt_term = math.sqrt(max(diff_sq, 1e-9))
        d_theta_dw = 1.0 / sqrt_term
        d_theta_dl = -w / (l * sqrt_term)
        is_near_perpendicular = True
    else:
        sqrt_term = math.sqrt(diff_sq)
        d_theta_dw = 1.0 / sqrt_term
        d_theta_dl = -w / (l * sqrt_term)
        is_near_perpendicular = False

    # Analytical variance calculation
    cov_wl = rho * w_unc * l_unc
    var_theta_rad = (
        (d_theta_dw ** 2) * (w_unc ** 2)
        + (d_theta_dl ** 2) * (l_unc ** 2)
        + 2.0 * d_theta_dw * d_theta_dl * cov_wl
    )
    # Ensure non-negative variance
    var_theta_rad = max(0.0, var_theta_rad)
    sigma_theta_rad = math.sqrt(var_theta_rad)
    sigma_theta_deg = math.degrees(sigma_theta_rad)

    # 1-sigma bounds
    lower_1sigma = max(0.0, theta_deg - sigma_theta_deg)
    upper_1sigma = min(90.0, theta_deg + sigma_theta_deg)

    # 2-sigma (approx 95.4% coverage) bounds
    lower_2sigma = max(0.0, theta_deg - 1.96 * sigma_theta_deg)
    upper_2sigma = min(90.0, theta_deg + 1.96 * sigma_theta_deg)

    return {
        "status": "valid",
        "width": round(w, 4),
        "length": round(l, 4),
        "width_uncertainty": round(w_unc, 4),
        "length_uncertainty": round(l_unc, 4),
        "correlation": round(rho, 4),
        "ratio": round(ratio, 4),
        "angle_degrees": round(theta_deg, 2),
        "angle_radians": round(theta_rad, 4),
        "d_theta_dw": round(d_theta_dw, 6),
        "d_theta_dl": round(d_theta_dl, 6),
        "sigma_theta_deg": round(sigma_theta_deg, 2),
        "sigma_theta_rad": round(sigma_theta_rad, 4),
        "lower_bound_1sigma_deg": round(lower_1sigma, 2),
        "upper_bound_1sigma_deg": round(upper_1sigma, 2),
        "lower_bound_2sigma_deg": round(lower_2sigma, 2),
        "upper_bound_2sigma_deg": round(upper_2sigma, 2),
        "is_near_perpendicular": is_near_perpendicular,
        "scale_invariance_note": (
            "Isotropic scale factor cancels in the W/L ratio; scale uncertainty does not affect "
            "the dimensionless impact angle."
        ),
        "physical_dimensions": {
            "width_mm": round(w_phys, 4),
            "length_mm": round(l_phys, 4),
            "width_uncertainty_mm": round(w_phys_unc, 4) if w_phys_unc is not None else None,
            "length_uncertainty_mm": round(l_phys_unc, 4) if l_phys_unc is not None else None,
            "scale_factor": s,
            "scale_uncertainty": s_unc,
        },
    }


def run_monte_carlo_simulation(
    width: float,
    length: float,
    width_uncertainty: float,
    length_uncertainty: float,
    iterations: int = 10000,
    seed: Optional[int] = 42,
    distribution: str = "normal",
    correlation: float = 0.0,
) -> Dict[str, Any]:
    """
    Execute high-performance vectorized Monte Carlo uncertainty simulation.

    Generates N pairs of (W, L) based on the specified distribution and correlation,
    evaluates valid samples (W > 0, L > 0, W <= L), and derives empirical statistics.

    Parameters:
        width: Nominal minor axis dimension.
        length: Nominal major axis dimension.
        width_uncertainty: Standard uncertainty sigma_W.
        length_uncertainty: Standard uncertainty sigma_L.
        iterations: Number of simulation draws (default 10,000, clamped to [100, 100000]).
        seed: Random seed for exact reproducibility.
        distribution: 'normal' (Gaussian) or 'uniform'.
        correlation: Correlation coefficient rho between W and L in [-1.0, 1.0].

    Returns:
        Dict[str, Any] with summary statistics, quantiles, and histogram data.
    """
    # Sanitize iteration count
    n_iter = int(iterations)
    if n_iter < 100:
        n_iter = 100
    elif n_iter > 100000:
        n_iter = 100000

    # Initialize NumPy random generator with seed
    rng = np.random.default_rng(seed)

    dist_type = distribution.lower().strip()
    rho = float(correlation)
    rho = max(-1.0, min(1.0, rho))

    if dist_type == "uniform":
        # Uniform distribution with mean mu and standard deviation sigma:
        # half-width a = sqrt(3) * sigma
        half_w = math.sqrt(3.0) * width_uncertainty
        half_l = math.sqrt(3.0) * length_uncertainty
        w_samples = rng.uniform(width - half_w, width + half_w, size=n_iter)
        if abs(rho) < 1e-6:
            l_samples = rng.uniform(length - half_l, length + half_l, size=n_iter)
        else:
            # Correlated uniform via Gaussian copula
            cov_mat = [[1.0, rho], [rho, 1.0]]
            z = rng.multivariate_normal([0.0, 0.0], cov_mat, size=n_iter)
            from scipy.stats import norm
            u = norm.cdf(z)
            w_samples = width - half_w + 2.0 * half_w * u[:, 0]
            l_samples = length - half_l + 2.0 * half_l * u[:, 1]
    else:
        # Normal (Gaussian) distribution
        if abs(rho) < 1e-6:
            w_samples = rng.normal(loc=width, scale=width_uncertainty, size=n_iter)
            l_samples = rng.normal(loc=length, scale=length_uncertainty, size=n_iter)
        else:
            # Bivariate normal sampling
            cov = rho * width_uncertainty * length_uncertainty
            cov_matrix = [
                [width_uncertainty ** 2, cov],
                [cov, length_uncertainty ** 2]
            ]
            samples = rng.multivariate_normal([width, length], cov_matrix, size=n_iter)
            w_samples = samples[:, 0]
            l_samples = samples[:, 1]

    # Filter valid physical draws
    # A valid stain requires: W > 0, L > 0, W <= L
    valid_mask = (w_samples > 0.0) & (l_samples > 0.0) & (w_samples <= l_samples)
    valid_count = int(np.sum(valid_mask))
    invalid_count = n_iter - valid_count
    invalid_pct = (invalid_count / n_iter) * 100.0

    if valid_count == 0:
        return {
            "status": "error",
            "message": "All Monte Carlo simulation iterations violated physical constraints (W > L or non-positive).",
            "iterations": n_iter,
            "valid_count": 0,
            "invalid_count": invalid_count,
            "invalid_percentage": 100.0,
            "seed": seed,
            "distribution": dist_type,
        }

    valid_w = w_samples[valid_mask]
    valid_l = l_samples[valid_mask]
    ratios = valid_w / valid_l
    ratios_clamped = np.clip(ratios, 0.0, 1.0)
    simulated_angles_deg = np.degrees(np.arcsin(ratios_clamped))

    mean_angle = float(np.mean(simulated_angles_deg))
    std_angle = float(np.std(simulated_angles_deg, ddof=1)) if valid_count > 1 else 0.0
    median_angle = float(np.median(simulated_angles_deg))
    min_angle = float(np.min(simulated_angles_deg))
    max_angle = float(np.max(simulated_angles_deg))

    # Empirical 95% simulation interval (2.5th and 97.5th percentiles)
    p2_5 = float(np.percentile(simulated_angles_deg, 2.5))
    p97_5 = float(np.percentile(simulated_angles_deg, 97.5))

    # Compute histogram bins for visualization
    bin_counts, bin_edges = np.histogram(simulated_angles_deg, bins=30)

    return {
        "status": "valid",
        "iterations": n_iter,
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "invalid_percentage": round(invalid_pct, 2),
        "seed": seed,
        "distribution": dist_type,
        "correlation": round(rho, 4),
        "mean_angle_deg": round(mean_angle, 2),
        "std_angle_deg": round(std_angle, 2),
        "median_angle_deg": round(median_angle, 2),
        "min_angle_deg": round(min_angle, 2),
        "max_angle_deg": round(max_angle, 2),
        "p2_5_deg": round(p2_5, 2),
        "p97_5_deg": round(p97_5, 2),
        "interval_width_95_deg": round(p97_5 - p2_5, 2),
        "histogram": {
            "counts": bin_counts.tolist(),
            "bin_edges": [round(float(b), 2) for b in bin_edges],
        },
        "forensic_disclaimer": (
            "The 95% simulation interval reflects parameter dispersion under the chosen "
            "distributional assumptions. It does not represent an absolute physical probability "
            "of event occurrence."
        ),
    }


def run_sensitivity_analysis(
    width: float,
    length: float,
    width_uncertainty: Optional[float] = None,
    length_uncertainty: Optional[float] = None,
    scale_factor: float = 1.0,
    scale_uncertainty: float = 0.0,
    perturbation_pct: float = 5.0,
) -> Dict[str, Any]:
    """
    Execute sensitivity analysis on bloodstain impact angle calculation.

    Methods:
    1. One-At-A-Time (OAT) perturbation (+/- perturbation_pct %).
    2. Normalized Sensitivity Indices (Elasticities):
           S_W = (d_theta / dW) * (W / theta)
           S_L = (d_theta / dL) * (L / theta) = -S_W
    3. Identification of the most influential measurement parameter.

    Returns:
        Dict[str, Any] with ranked parameter swings and elasticity metrics.
    """
    w = float(width)
    l = float(length)
    p_pct = float(perturbation_pct)
    fraction = p_pct / 100.0

    base_ratio = min(w / l, 1.0)
    base_theta_rad = math.asin(base_ratio)
    base_theta_deg = math.degrees(base_theta_rad)

    parameters: List[Dict[str, Any]] = []

    # 1. Parameter: Stain Width W
    w_low = w * (1.0 - fraction)
    w_high = min(w * (1.0 + fraction), l)  # Clamp so W does not exceed L
    theta_w_low = math.degrees(math.asin(min(w_low / l, 1.0)))
    theta_w_high = math.degrees(math.asin(min(w_high / l, 1.0)))
    swing_w = abs(theta_w_high - theta_w_low)

    # Elasticity S_W
    diff_sq = max(l * l - w * w, 1e-9)
    sqrt_term = math.sqrt(diff_sq)
    d_theta_dw = 1.0 / sqrt_term
    d_theta_dl = -w / (l * sqrt_term)
    elasticity_w = (d_theta_dw * w) / base_theta_rad if base_theta_rad > 0 else 0.0

    parameters.append({
        "name": "Stain Width (W)",
        "parameter_key": "width",
        "baseline": round(w, 4),
        "unit": "mm",
        "low_val": round(w_low, 4),
        "high_val": round(w_high, 4),
        "theta_at_low": round(theta_w_low, 2),
        "theta_at_high": round(theta_w_high, 2),
        "delta_theta_low": round(theta_w_low - base_theta_deg, 2),
        "delta_theta_high": round(theta_w_high - base_theta_deg, 2),
        "swing_deg": round(swing_w, 2),
        "elasticity": round(elasticity_w, 4),
        "influence_direction": "Positive (Higher width increases impact angle)",
    })

    # 2. Parameter: Stain Length L
    l_low = max(l * (1.0 - fraction), w)  # Clamp so L does not become smaller than W
    l_high = l * (1.0 + fraction)
    theta_l_low = math.degrees(math.asin(min(w / l_low, 1.0)))
    theta_l_high = math.degrees(math.asin(min(w / l_high, 1.0)))
    swing_l = abs(theta_l_high - theta_l_low)
    elasticity_l = (d_theta_dl * l) / base_theta_rad if base_theta_rad > 0 else 0.0

    parameters.append({
        "name": "Stain Length (L)",
        "parameter_key": "length",
        "baseline": round(l, 4),
        "unit": "mm",
        "low_val": round(l_low, 4),
        "high_val": round(l_high, 4),
        "theta_at_low": round(theta_l_low, 2),
        "theta_at_high": round(theta_l_high, 2),
        "delta_theta_low": round(theta_l_low - base_theta_deg, 2),
        "delta_theta_high": round(theta_l_high - base_theta_deg, 2),
        "swing_deg": round(swing_l, 2),
        "elasticity": round(elasticity_l, 4),
        "influence_direction": "Negative (Higher length decreases impact angle)",
    })

    # 3. Parameter: Scale Calibration Factor (s)
    # Scales W and L identically; does not alter ratio W/L
    parameters.append({
        "name": "Calibration Scale Factor (s)",
        "parameter_key": "scale_factor",
        "baseline": round(scale_factor, 4),
        "unit": "mm/px",
        "low_val": round(scale_factor * (1.0 - fraction), 4),
        "high_val": round(scale_factor * (1.0 + fraction), 4),
        "theta_at_low": round(base_theta_deg, 2),
        "theta_at_high": round(base_theta_deg, 2),
        "delta_theta_low": 0.0,
        "delta_theta_high": 0.0,
        "swing_deg": 0.0,
        "elasticity": 0.0,
        "influence_direction": "Invariant (Cancels out of dimensionless ratio W/L)",
    })

    # 4. Parameter: Width Uncertainty (sigma_W) if provided
    if width_uncertainty is not None and width_uncertainty > 0:
        w_unc_low = width_uncertainty * (1.0 - fraction)
        w_unc_high = width_uncertainty * (1.0 + fraction)
        # Propagated sigma at low and high
        prop_low = propagate_impact_angle_uncertainty(w, l, w_unc_low, length_uncertainty)
        prop_high = propagate_impact_angle_uncertainty(w, l, w_unc_high, length_uncertainty)
        sig_low = prop_low["sigma_theta_deg"] or 0.0
        sig_high = prop_high["sigma_theta_deg"] or 0.0
        parameters.append({
            "name": "Width Uncertainty (σ_W)",
            "parameter_key": "width_uncertainty",
            "baseline": round(width_uncertainty, 4),
            "unit": "mm",
            "low_val": round(w_unc_low, 4),
            "high_val": round(w_unc_high, 4),
            "theta_at_low": round(base_theta_deg - sig_low, 2),
            "theta_at_high": round(base_theta_deg + sig_high, 2),
            "delta_theta_low": round(-sig_low, 2),
            "delta_theta_high": round(sig_high, 2),
            "swing_deg": round(sig_high + sig_low, 2),
            "elasticity": None,
            "influence_direction": "Expands uncertainty bounds symmetrically",
        })

    # 5. Parameter: Length Uncertainty (sigma_L) if provided
    if length_uncertainty is not None and length_uncertainty > 0:
        l_unc_low = length_uncertainty * (1.0 - fraction)
        l_unc_high = length_uncertainty * (1.0 + fraction)
        prop_low = propagate_impact_angle_uncertainty(w, l, width_uncertainty, l_unc_low)
        prop_high = propagate_impact_angle_uncertainty(w, l, width_uncertainty, l_unc_high)
        sig_low = prop_low["sigma_theta_deg"] or 0.0
        sig_high = prop_high["sigma_theta_deg"] or 0.0
        parameters.append({
            "name": "Length Uncertainty (σ_L)",
            "parameter_key": "length_uncertainty",
            "baseline": round(length_uncertainty, 4),
            "unit": "mm",
            "low_val": round(l_unc_low, 4),
            "high_val": round(l_unc_high, 4),
            "theta_at_low": round(base_theta_deg - sig_low, 2),
            "theta_at_high": round(base_theta_deg + sig_high, 2),
            "delta_theta_low": round(-sig_low, 2),
            "delta_theta_high": round(sig_high, 2),
            "swing_deg": round(sig_high + sig_low, 2),
            "elasticity": None,
            "influence_direction": "Expands uncertainty bounds symmetrically",
        })

    # Sort parameters by swing magnitude descending
    parameters.sort(key=lambda p: p["swing_deg"], reverse=True)

    most_influential = parameters[0]["name"] if parameters else "None"

    return {
        "status": "valid",
        "baseline_angle_deg": round(base_theta_deg, 2),
        "perturbation_percentage": p_pct,
        "parameters": parameters,
        "most_influential_parameter": most_influential,
        "elasticity_symmetry_note": (
            f"Normalized sensitivity indices: S_W = {elasticity_w:.4f}, S_L = {elasticity_l:.4f}. "
            "S_L is mathematically identical in magnitude and opposite in sign to S_W."
        ),
    }


def render_monte_carlo_histogram(mc_results: Dict[str, Any]) -> str:
    """
    Render a clean forensic histogram of the Monte Carlo angle distribution
    and return as a base64-encoded PNG data URI.
    """
    if mc_results.get("status") != "valid" or not mc_results.get("histogram"):
        return ""

    counts = mc_results["histogram"]["counts"]
    edges = mc_results["histogram"]["bin_edges"]
    mean_val = mc_results["mean_angle_deg"]
    p2_5 = mc_results["p2_5_deg"]
    p97_5 = mc_results["p97_5_deg"]
    std_val = mc_results["std_angle_deg"]
    n_valid = mc_results["valid_count"]

    fig, ax = plt.subplots(figsize=(7.5, 4.0), dpi=120)
    fig.patch.set_facecolor("#0b0f19")
    ax.set_facecolor("#111827")

    widths = [edges[i + 1] - edges[i] for i in range(len(counts))]
    centers = [edges[i] + widths[i] / 2.0 for i in range(len(counts))]

    # Forensic histogram bars
    ax.bar(centers, counts, width=widths, color="#38bdf8", alpha=0.72, edgecolor="#0284c7", linewidth=0.8)

    # Vertical markers for Mean, 2.5th, and 97.5th percentiles
    ax.axvline(mean_val, color="#f59e0b", linestyle="--", linewidth=1.8, label=f"Mean: {mean_val:.2f}° (±{std_val:.2f}°)")
    ax.axvline(p2_5, color="#ef4444", linestyle=":", linewidth=1.6, label=f"2.5% Bound: {p2_5:.2f}°")
    ax.axvline(p97_5, color="#10b981", linestyle=":", linewidth=1.6, label=f"97.5% Bound: {p97_5:.2f}°")

    ax.set_title(
        f"Monte Carlo Impact Angle Distribution (N = {n_valid:,} Valid Draws)",
        fontsize=11.5,
        fontweight="bold",
        color="#f8fafc",
        pad=10,
    )
    ax.set_xlabel("Simulated Impact Angle θ (°)", fontsize=10, color="#94a3b8", labelpad=6)
    ax.set_ylabel("Frequency / Count", fontsize=10, color="#94a3b8", labelpad=6)

    ax.tick_params(colors="#94a3b8", labelsize=8.5)
    ax.grid(True, linestyle="--", alpha=0.25, color="#334155")
    for spine in ax.spines.values():
        spine.set_color("#334155")

    leg = ax.legend(facecolor="#1e293b", edgecolor="#334155", fontsize=8.5, labelcolor="#f8fafc", loc="upper right")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    b64_str = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{b64_str}"


def render_tornado_chart(sensitivity_results: Dict[str, Any]) -> str:
    """
    Render a horizontal Tornado Chart comparing the impact angle swing for each parameter
    and return as a base64-encoded PNG data URI.
    """
    if sensitivity_results.get("status") != "valid" or not sensitivity_results.get("parameters"):
        return ""

    params = sensitivity_results["parameters"]
    base_theta = sensitivity_results["baseline_angle_deg"]
    p_pct = sensitivity_results["perturbation_percentage"]

    # Invert order for horizontal chart (most influential at top)
    names = [p["name"] for p in reversed(params)]
    swings = [p["swing_deg"] for p in reversed(params)]
    low_deltas = [p["delta_theta_low"] for p in reversed(params)]
    high_deltas = [p["delta_theta_high"] for p in reversed(params)]

    fig, ax = plt.subplots(figsize=(8.0, 4.2), dpi=120)
    fig.patch.set_facecolor("#0b0f19")
    ax.set_facecolor("#111827")

    y_pos = np.arange(len(names))

    # Low perturbation bars (left / negative deviation)
    bars_low = ax.barh(y_pos, low_deltas, align="center", color="#38bdf8", alpha=0.85, edgecolor="#0284c7", label=f"-{p_pct:.0f}% Perturbation")
    # High perturbation bars (right / positive deviation)
    bars_high = ax.barh(y_pos, high_deltas, align="center", color="#f59e0b", alpha=0.85, edgecolor="#d97706", label=f"+{p_pct:.0f}% Perturbation")

    ax.axvline(0.0, color="#ffffff", linestyle="-", linewidth=1.2, alpha=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9.5, color="#f8fafc")
    ax.set_xlabel(f"Impact Angle Deviation from Baseline {base_theta:.2f}° (Δθ in degrees)", fontsize=10, color="#94a3b8", labelpad=6)
    ax.set_title(
        f"Parameter Sensitivity Tornado Chart (±{p_pct:.0f}% Local Perturbation)",
        fontsize=11.5,
        fontweight="bold",
        color="#f8fafc",
        pad=10,
    )

    ax.tick_params(colors="#94a3b8", labelsize=8.5)
    ax.grid(True, linestyle="--", alpha=0.25, color="#334155", axis="x")
    for spine in ax.spines.values():
        spine.set_color("#334155")

    # Annotate total swing values
    for i, swing in enumerate(swings):
        max_delta = max(abs(low_deltas[i]), abs(high_deltas[i]))
        offset = 0.15 if max_delta > 0 else 0.05
        ax.text(
            max_delta + offset,
            y_pos[i],
            f"Swing: {swing:.2f}°",
            va="center",
            ha="left",
            fontsize=8.5,
            color="#34d399",
            fontweight="bold",
        )

    ax.legend(facecolor="#1e293b", edgecolor="#334155", fontsize=8.5, labelcolor="#f8fafc", loc="lower right")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    b64_str = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{b64_str}"


def execute_full_uncertainty_analysis(
    width: float,
    length: float,
    width_uncertainty: Optional[float] = None,
    length_uncertainty: Optional[float] = None,
    correlation: float = 0.0,
    scale_factor: float = 1.0,
    scale_uncertainty: float = 0.0,
    iterations: int = 10000,
    seed: Optional[int] = 42,
    distribution: str = "normal",
    perturbation_pct: float = 5.0,
    uncertainty_source: str = "Automated Estimation",
    is_manual: bool = False,
    stain_label: str = "Stain-01",
) -> Dict[str, Any]:
    """
    Execute complete end-to-end Phase 5.3 uncertainty and sensitivity analysis pipeline:
    1. Validation & Input Sanitation
    2. Analytical Error Propagation
    3. Monte Carlo Simulation
    4. Sensitivity Analysis & Tornado Metrics
    5. Figure Rendering (Histogram & Tornado Base64)

    Returns:
        Structured dictionary ready for database persistence, JSON serialization, and UI rendering.
    """
    # 1. Validation
    validated = validate_uncertainty_inputs(
        width=width,
        length=length,
        width_uncertainty=width_uncertainty,
        length_uncertainty=length_uncertainty,
        correlation=correlation,
        scale_factor=scale_factor,
        scale_uncertainty=scale_uncertainty,
    )

    w = validated["width"]
    l = validated["length"]
    w_unc = validated["width_uncertainty"]
    l_unc = validated["length_uncertainty"]
    rho = validated["correlation"]
    s = validated["scale_factor"]
    s_unc = validated["scale_uncertainty"]

    # 2. Analytical Error Propagation
    analytical = propagate_impact_angle_uncertainty(
        width=w,
        length=l,
        width_uncertainty=w_unc,
        length_uncertainty=l_unc,
        correlation=rho,
        scale_factor=s,
        scale_uncertainty=s_unc,
    )

    # 3. Monte Carlo Simulation (if uncertainties provided)
    if w_unc is not None and l_unc is not None:
        monte_carlo = run_monte_carlo_simulation(
            width=w,
            length=l,
            width_uncertainty=w_unc,
            length_uncertainty=l_unc,
            iterations=iterations,
            seed=seed,
            distribution=distribution,
            correlation=rho,
        )
        histogram_b64 = render_monte_carlo_histogram(monte_carlo)
    else:
        monte_carlo = {
            "status": "unavailable",
            "message": "Monte Carlo simulation requires non-null width and length uncertainties.",
            "iterations": iterations,
            "seed": seed,
            "distribution": distribution,
        }
        histogram_b64 = ""

    # 4. Sensitivity Analysis
    sensitivity = run_sensitivity_analysis(
        width=w,
        length=l,
        width_uncertainty=w_unc,
        length_uncertainty=l_unc,
        scale_factor=s,
        scale_uncertainty=s_unc,
        perturbation_pct=perturbation_pct,
    )
    tornado_b64 = render_tornado_chart(sensitivity)

    return {
        "status": "success",
        "stain_label": stain_label,
        "uncertainty_source": uncertainty_source,
        "is_manual": is_manual,
        "inputs": validated,
        "analytical": analytical,
        "monte_carlo": monte_carlo,
        "sensitivity": sensitivity,
        "figures": {
            "histogram_b64": histogram_b64,
            "tornado_b64": tornado_b64,
        },
        "scientific_disclaimer": (
            "Forensic Uncertainty Governance: Impact angle estimates assume an idealized ellipse. "
            "Error propagation applies first-order Taylor expansion; Monte Carlo accounts for "
            "non-linear trigonometric transformation. Neither simulation nor analytical error bounds "
            "represent an absolute probability that a specific incident deposition occurred."
        ),
    }
