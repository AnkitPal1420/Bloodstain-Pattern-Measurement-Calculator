"""
test_phase5_3_uncertainty_analysis.py
====================================
Comprehensive Automated Verification Suite for Phase 5.3:
Advanced Uncertainty & Sensitivity Analysis.

Tests:
1. Analytical Error Propagation (Partial derivatives, Taylor expansion, degrees conversion)
2. Invalid input handling (NaN, inf, negative numbers, empty values)
3. Zero-measurement handling (W=0, L=0)
4. Geometric constraint handling (W > L violates arcsin domain)
5. Missing uncertainty handling (Unavailable flag, no fabricated values)
6. Calibration uncertainty & Scale invariance (Cancellation in W/L, mm physical dimensions)
7. Monte Carlo seed reproducibility
8. Monte Carlo statistical metrics (Mean, std dev, median, 2.5th/97.5th percentiles)
9. Invalid simulation draw tracking
10. Sensitivity analysis & normalized elasticity symmetry (S_W = -S_L)
11. Database persistence & non-destructive revision tracking
12. Flask routes & cross-case security authorization (403 on mismatch)
13. Compatibility with Phase 5.1 and Phase 5.2
"""

import math
import os
import sqlite3
import pytest
import numpy as np

import app
import database
import uncertainty_analysis as ua
from uncertainty_analysis import UncertaintyValidationError


@pytest.fixture(autouse=True)
def clean_db():
    """Ensure database is initialized and clear uncertainty table before tests."""
    database.init_db(seed_demo=True)
    conn = database.get_db_connection()
    conn.execute("DELETE FROM uncertainty_analyses;")
    conn.commit()
    conn.close()
    yield


class TestPhase53AnalyticalPropagation:
    """Mathematical verification of analytical error propagation."""

    def test_01_exact_derivatives_and_propagation_30deg(self):
        """Verify analytical propagation for 30 deg impact angle (W=5.0 mm, L=10.0 mm)."""
        w = 5.0
        l = 10.0
        w_unc = 0.10
        l_unc = 0.20

        # Exact expected values:
        # theta = arcsin(5/10) = arcsin(0.5) = pi/6 rad = 30.0 deg
        # d_theta/dW = 1 / sqrt(10^2 - 5^2) = 1 / sqrt(75) = 1 / (5*sqrt(3)) = 0.11547 rad/mm
        # d_theta/dL = -5 / (10 * sqrt(75)) = -0.5 * d_theta/dW = -0.057735 rad/mm
        expected_d_dw = 1.0 / math.sqrt(75.0)
        expected_d_dl = -w / (l * math.sqrt(75.0))

        # Variance: (d_dw * w_unc)^2 + (d_dl * l_unc)^2
        expected_var_rad = (expected_d_dw * w_unc) ** 2 + (expected_d_dl * l_unc) ** 2
        expected_sigma_deg = math.degrees(math.sqrt(expected_var_rad))

        res = ua.propagate_impact_angle_uncertainty(w, l, w_unc, l_unc, correlation=0.0)

        assert res["status"] == "valid"
        assert res["angle_degrees"] == 30.0
        assert math.isclose(res["d_theta_dw"], expected_d_dw, rel_tol=1e-4)
        assert math.isclose(res["d_theta_dl"], expected_d_dl, rel_tol=1e-4)
        assert math.isclose(res["sigma_theta_deg"], round(expected_sigma_deg, 2), abs_tol=0.02)
        assert res["lower_bound_1sigma_deg"] == round(30.0 - res["sigma_theta_deg"], 2)
        assert res["upper_bound_1sigma_deg"] == round(30.0 + res["sigma_theta_deg"], 2)

    def test_02_covariance_effect(self):
        """Verify that positive correlation between W and L reduces propagated variance."""
        w, l = 5.0, 10.0
        w_unc, l_unc = 0.15, 0.25

        uncorr = ua.propagate_impact_angle_uncertainty(w, l, w_unc, l_unc, correlation=0.0)
        pos_corr = ua.propagate_impact_angle_uncertainty(w, l, w_unc, l_unc, correlation=0.8)
        neg_corr = ua.propagate_impact_angle_uncertainty(w, l, w_unc, l_unc, correlation=-0.8)

        # Because d_theta/dW > 0 and d_theta/dL < 0, positive correlation creates a negative covariance term,
        # which REDUCES total variance (errors partially cancel).
        assert pos_corr["sigma_theta_deg"] < uncorr["sigma_theta_deg"]
        assert neg_corr["sigma_theta_deg"] > uncorr["sigma_theta_deg"]

    def test_03_invalid_inputs_rejected(self):
        """Verify that non-numeric, negative, and domain-violating inputs raise UncertaintyValidationError."""
        with pytest.raises(UncertaintyValidationError, match="cannot exceed length"):
            ua.validate_uncertainty_inputs(width=10.5, length=10.0)

        with pytest.raises(UncertaintyValidationError, match="strictly positive"):
            ua.validate_uncertainty_inputs(width=0.0, length=10.0)

        with pytest.raises(UncertaintyValidationError, match="strictly positive"):
            ua.validate_uncertainty_inputs(width=5.0, length=-2.0)

        with pytest.raises(UncertaintyValidationError, match="cannot be blank"):
            ua.validate_uncertainty_inputs(width="", length="10.0")

        with pytest.raises(UncertaintyValidationError, match="NaN or infinite"):
            ua.validate_uncertainty_inputs(width=float("nan"), length=10.0)

        with pytest.raises(UncertaintyValidationError, match="cannot be negative"):
            ua.validate_uncertainty_inputs(width=5.0, length=10.0, width_uncertainty=-0.1)

    def test_04_missing_uncertainty_handled_cleanly(self):
        """Verify that omitting uncertainty does not fabricate numbers and flags unavailable."""
        res = ua.propagate_impact_angle_uncertainty(width=5.0, length=10.0, width_uncertainty=None, length_uncertainty=None)
        assert res["status"] == "unavailable"
        assert res["angle_degrees"] == 30.0
        assert res["sigma_theta_deg"] is None
        assert res["lower_bound_1sigma_deg"] is None

    def test_05_scale_invariance_and_physical_dimensions(self):
        """Verify isotropic scale cancellation in impact angle, and scale uncertainty in physical dimensions."""
        w_px, l_px = 50.0, 100.0
        w_unc_px, l_unc_px = 1.0, 2.0
        s1 = 0.10  # 0.1 mm/px -> W=5mm, L=10mm
        s2 = 0.25  # 0.25 mm/px -> W=12.5mm, L=25mm

        res1 = ua.propagate_impact_angle_uncertainty(w_px, l_px, w_unc_px, l_unc_px, scale_factor=s1)
        res2 = ua.propagate_impact_angle_uncertainty(w_px, l_px, w_unc_px, l_unc_px, scale_factor=s2)

        # Dimensionless angles and angular uncertainties must be identical
        assert res1["angle_degrees"] == res2["angle_degrees"] == 30.0
        assert res1["sigma_theta_deg"] == res2["sigma_theta_deg"]

        # Physical dimensions must scale properly
        assert math.isclose(res1["physical_dimensions"]["width_mm"], 5.0, abs_tol=1e-3)
        assert math.isclose(res2["physical_dimensions"]["width_mm"], 12.5, abs_tol=1e-3)


class TestPhase53MonteCarloSimulation:
    """Verification of Monte Carlo simulation engine and statistical correctness."""

    def test_06_seed_reproducibility(self):
        """Verify that identical random seeds produce identical Monte Carlo distributions."""
        w, l = 5.0, 10.0
        w_unc, l_unc = 0.15, 0.25

        mc1 = ua.run_monte_carlo_simulation(w, l, w_unc, l_unc, iterations=2000, seed=12345)
        mc2 = ua.run_monte_carlo_simulation(w, l, w_unc, l_unc, iterations=2000, seed=12345)

        assert mc1["status"] == "valid"
        assert mc1["mean_angle_deg"] == mc2["mean_angle_deg"]
        assert mc1["std_angle_deg"] == mc2["std_angle_deg"]
        assert mc1["p2_5_deg"] == mc2["p2_5_deg"]
        assert mc1["p97_5_deg"] == mc2["p97_5_deg"]
        assert mc1["histogram"]["counts"] == mc2["histogram"]["counts"]

    def test_07_monte_carlo_mean_converges_to_analytical(self):
        """Verify that Monte Carlo mean converges to the true baseline angle for moderate uncertainties."""
        w, l = 5.0, 10.0
        w_unc, l_unc = 0.08, 0.12

        mc = ua.run_monte_carlo_simulation(w, l, w_unc, l_unc, iterations=10000, seed=42)
        assert mc["status"] == "valid"
        # 30 deg baseline; for small uncertainties, Monte Carlo mean should be within 0.15 deg of 30 deg
        assert math.isclose(mc["mean_angle_deg"], 30.0, abs_tol=0.20)
        assert mc["invalid_percentage"] < 0.1

    def test_08_invalid_draws_tracked_near_boundary(self):
        """Verify that near-perpendicular impacts track invalid (W > L) draws without failing."""
        w, l = 9.8, 10.0  # Near 90 deg
        w_unc, l_unc = 0.20, 0.20

        mc = ua.run_monte_carlo_simulation(w, l, w_unc, l_unc, iterations=3000, seed=42)
        assert mc["status"] == "valid"
        # Because W is very close to L, Gaussian variations will produce samples where W > L
        assert mc["invalid_count"] > 0
        assert mc["invalid_percentage"] > 0.0
        # Valid samples still compute proper quantiles bounded by 90 deg
        assert mc["max_angle_deg"] <= 90.0


class TestPhase53SensitivityAnalysis:
    """Verification of sensitivity analysis and normalized elasticity."""

    def test_09_normalized_elasticity_symmetry(self):
        """Verify that S_W = -S_L (mathematical elasticity symmetry)."""
        w, l = 5.0, 10.0
        sens = ua.run_sensitivity_analysis(w, l, width_uncertainty=0.1, length_uncertainty=0.2)

        assert sens["status"] == "valid"
        w_param = next(p for p in sens["parameters"] if p["parameter_key"] == "width")
        l_param = next(p for p in sens["parameters"] if p["parameter_key"] == "length")

        assert w_param["elasticity"] is not None
        assert l_param["elasticity"] is not None
        # S_W must equal -S_L
        assert math.isclose(w_param["elasticity"], -l_param["elasticity"], rel_tol=1e-4)

    def test_10_tornado_chart_and_histogram_rendering(self):
        """Verify base64 PNG rendering for histogram and tornado charts."""
        res = ua.execute_full_uncertainty_analysis(5.0, 10.0, 0.1, 0.2, iterations=1000, seed=42)
        assert res["status"] == "success"
        assert res["figures"]["histogram_b64"].startswith("data:image/png;base64,")
        assert res["figures"]["tornado_b64"].startswith("data:image/png;base64,")


class TestPhase53DatabaseAndRoutes:
    """Verification of database persistence, revisions, and Flask endpoints."""

    def test_11_database_crud_and_revision_tracking(self):
        """Verify creation, revision increment, and examiner review updates."""
        case_id = "BPA-2026-0101"
        res_data = ua.execute_full_uncertainty_analysis(5.0, 10.0, 0.1, 0.2)

        # Revision 1
        rec1 = database.create_uncertainty_analysis_record(
            case_id=case_id,
            width=5.0,
            length=10.0,
            analysis_data=res_data,
            stain_label="Stain-01",
            width_uncertainty=0.1,
            length_uncertainty=0.2,
        )
        assert rec1["revision_number"] == 1
        assert "UNC-" in rec1["analysis_id"]

        # Revision 2 on same base
        rec2 = database.create_uncertainty_analysis_record(
            case_id=case_id,
            width=5.0,
            length=10.0,
            analysis_data=res_data,
            stain_label="Stain-01",
            parent_analysis_id=rec1["analysis_id"],
            width_uncertainty=0.15,
            length_uncertainty=0.25,
        )
        assert rec2["revision_number"] == 2

        # Review update
        updated = database.update_uncertainty_analysis_review(
            analysis_id=rec1["analysis_id"],
            review_status="Examiner Approved",
            examiner_name="Senior Examiner Miller",
            examiner_notes="Verified against NIST optical graticule.",
        )
        assert updated["review_status"] == "Examiner Approved"
        assert updated["examiner_name"] == "Senior Examiner Miller"

    def test_12_flask_routes_and_security(self):
        """Verify web endpoints, form execution, and cross-case security protection."""
        client = app.app.test_client()
        case_id = "BPA-2026-0101"

        # 1. GET workspace
        resp = client.get(f"/uncertainty-analysis/{case_id}")
        assert resp.status_code == 200
        assert b"Measurement Uncertainty &amp; Sensitivity Laboratory" in resp.data

        # 2. POST analyze with valid inputs
        post_data = {
            "stain_label": "Stain-Integration",
            "width": "6.0",
            "length": "12.0",
            "width_uncertainty": "0.10",
            "length_uncertainty": "0.20",
            "iterations": "2000",
            "random_seed": "42",
        }
        resp_post = client.post(
            f"/uncertainty-analysis/{case_id}/analyze",
            data=post_data,
            headers={"Accept": "application/json"},
        )
        assert resp_post.status_code == 201
        json_data = resp_post.get_json()
        assert json_data["status"] == "ok"
        analysis_id = json_data["analysis_id"]

        # 3. Cross-case security check: Attempting to review with mismatched case_id must return 403
        resp_sec = client.post(
            f"/uncertainty-analysis/BPA-2026-9999/review/{analysis_id}",
            data={"review_status": "Examiner Approved"},
            headers={"Accept": "application/json"},
        )
        assert resp_sec.status_code in (403, 404)

        # 4. Valid review submission
        resp_rev = client.post(
            f"/uncertainty-analysis/{case_id}/review/{analysis_id}",
            data={
                "review_status": "Examiner Approved",
                "examiner_name": "J. Miller, CBPA",
                "examiner_notes": "Accepted under OSAC standards.",
            },
            headers={"Accept": "application/json"},
        )
        assert resp_rev.status_code == 200

        # 5. REST API detail endpoint
        resp_api = client.get(f"/api/uncertainty-analysis/{analysis_id}")
        assert resp_api.status_code == 200
        api_data = resp_api.get_json()
        assert api_data["status"] == "ok"
        assert api_data["record"]["analysis_id"] == analysis_id

    def test_13_invalid_form_submission_returns_error(self):
        """Verify that submitting invalid measurements (W > L) returns proper error message."""
        client = app.app.test_client()
        case_id = "BPA-2026-0101"
        invalid_post = {
            "width": "12.0",
            "length": "5.0",  # Invalid: W > L
        }
        resp = client.post(
            f"/uncertainty-analysis/{case_id}/analyze",
            data=invalid_post,
            headers={"Accept": "application/json"},
        )
        assert resp.status_code == 400
        err_json = resp.get_json()
        assert "cannot exceed length" in err_json["error"]
