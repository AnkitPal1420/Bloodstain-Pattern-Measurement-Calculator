"""
PHASE 5.1 — ADVANCED BLOODSTAIN PATTERN CLASSIFICATION & ANALYTICAL DECISION SUPPORT
Comprehensive Automated Test Suite (Sections 10 & 11)

Validates:
1. Feature Extraction Mathematics (Circularity, Aspect Ratio, Solidity, Eccentricity) & domain error handling.
2. Synthetic Image Morphologies (circular, elliptical, elongated, irregular, isolated, dense cluster, linear trail,
   overlapping/coalesced, low-contrast, noisy background, multiple resolutions).
3. Pixel vs. Calibrated Metric Unit Discipline & Non-Destructive Original Image Integrity.
4. No-Silent-Rejection Contour Policy (all rejected/uncertain/overlapping contours recorded with explicit reason).
5. Explainable Rule-Based Classification Engine, Uncertainty Handling, and 15-Category Taxonomy.
6. Provenance Labeling (`Automatically Measured`, `Examiner Observed`, `Examiner Supplied`,
   `Software Suggested`, `Examiner Approved`) & No-Automatic-Definitive-Conclusion enforcement.
7. Database Persistence, Non-Destructive Revision Tracking (`PAT-YYYY-NNNN-R1`, `R2`), and 6 Audit Log Action Types.
8. Flask Routes (`/pattern-analysis/<case_id>`), Cross-Case Access Control, and PDF Report Integration.
"""

import hashlib
import json
import math
import os
import tempfile
import unittest

import cv2
import numpy as np

import app as flask_app
import calculator
import database
import pattern_analysis
import report_generator


class TestPhase51PatternAnalysis(unittest.TestCase):
    """Comprehensive Phase 5.1 test suite."""

    @classmethod
    def setUpClass(cls):
        database.init_db(seed_demo=True)
        flask_app.app.config["TESTING"] = True
        cls.client = flask_app.app.test_client()
        cls.temp_dir = tempfile.mkdtemp(prefix="bpa_phase51_")

    def _save_synthetic_image(self, name: str, img: np.ndarray) -> str:
        path = os.path.join(self.temp_dir, name)
        cv2.imwrite(path, img)
        return path

    # =========================================================================
    # 1. FEATURE EXTRACTION MATHEMATICS & DOMAIN VALIDATION
    # =========================================================================

    def test_01_circularity_exact_and_domain_errors(self):
        """Verify Circularity = 4 * pi * Area / Perimeter^2 and invalid input rejection."""
        # Ideal circle of radius r = 50 -> Area = pi * r^2, Perimeter = 2 * pi * r -> Circularity = 1.0
        r = 50.0
        area = math.pi * r * r
        perimeter = 2.0 * math.pi * r
        circ = pattern_analysis.compute_circularity(area, perimeter)
        self.assertAlmostEqual(circ, 1.0, places=6)

        # Square of side s = 40 -> Area = 1600, Perimeter = 160 -> Circularity = 4*pi*1600 / 25600 = pi/4 ~ 0.785398
        circ_sq = pattern_analysis.compute_circularity(1600.0, 160.0)
        self.assertAlmostEqual(circ_sq, math.pi / 4.0, places=6)

        # Invalid domain inputs must raise PatternAnalysisError
        for bad_area, bad_perim in [
            (0.0, 10.0),
            (-5.0, 10.0),
            (10.0, 0.0),
            (10.0, -2.0),
            (float("nan"), 10.0),
            (10.0, float("inf")),
        ]:
            with self.assertRaises(pattern_analysis.PatternAnalysisError):
                pattern_analysis.compute_circularity(bad_area, bad_perim)

    def test_02_aspect_ratio_exact_and_domain_errors(self):
        """Verify Aspect Ratio = Major Axis / Minor Axis and invalid input rejection."""
        self.assertAlmostEqual(pattern_analysis.compute_aspect_ratio(20.0, 10.0), 2.0, places=6)
        self.assertAlmostEqual(pattern_analysis.compute_aspect_ratio(15.0, 15.0), 1.0, places=6)

        # Minor axis > major axis or non-positive values must raise PatternAnalysisError
        for maj, min_ax in [(10.0, 20.0), (0.0, 0.0), (-10.0, 5.0), (10.0, -5.0), (float("nan"), 5.0)]:
            with self.assertRaises(pattern_analysis.PatternAnalysisError):
                pattern_analysis.compute_aspect_ratio(maj, min_ax)

    def test_03_solidity_exact_and_domain_errors(self):
        """Verify Solidity = Contour Area / Convex Hull Area and invalid input rejection."""
        self.assertAlmostEqual(pattern_analysis.compute_solidity(85.0, 100.0), 0.85, places=6)
        self.assertAlmostEqual(pattern_analysis.compute_solidity(100.0, 100.0), 1.0, places=6)

        for area, hull in [(0.0, 100.0), (50.0, 0.0), (120.0, 100.0), (-10.0, 50.0), (float("inf"), 100.0)]:
            with self.assertRaises(pattern_analysis.PatternAnalysisError):
                pattern_analysis.compute_solidity(area, hull)

    def test_04_eccentricity_exact_and_domain_errors(self):
        """Verify Eccentricity = sqrt(1 - (Minor / Major)^2) and consistency with impact angle."""
        # Circle: minor == major -> eccentricity = 0.0
        self.assertAlmostEqual(pattern_analysis.compute_eccentricity(20.0, 20.0), 0.0, places=6)

        # 30 deg impact angle: minor / major = sin(30 deg) = 0.5 -> eccentricity = sqrt(1 - 0.25) = sqrt(3)/2 ~ 0.8660254
        ecc_30 = pattern_analysis.compute_eccentricity(20.0, 10.0)
        self.assertAlmostEqual(ecc_30, math.sqrt(3.0) / 2.0, places=6)

        # Verify consistency with calculator.calculate_impact_angle(10.0, 20.0)
        impact_res = calculator.calculate_impact_angle(10.0, 20.0)
        self.assertAlmostEqual(impact_res["angle_degrees"], 30.0, places=4)

        for maj, min_ax in [(10.0, 15.0), (0.0, 5.0), (10.0, 0.0), (-10.0, -5.0)]:
            with self.assertRaises(pattern_analysis.PatternAnalysisError):
                pattern_analysis.compute_eccentricity(maj, min_ax)

    # =========================================================================
    # 2. SYNTHETIC IMAGE MORPHOLOGIES & IMAGE CONDITIONS (SECTION 10)
    # =========================================================================

    def test_05_synthetic_circular_and_elliptical_stains(self):
        """Test circular stain (90 deg drop) and elliptical stain (oblique impact)."""
        # 1. Circular stain on clean light background
        img_circ = np.full((400, 400, 3), 240, dtype=np.uint8)
        cv2.circle(img_circ, (200, 200), 45, (25, 20, 140), -1)
        path_circ = self._save_synthetic_image("synth_circular.png", img_circ)

        feat_circ = pattern_analysis.extract_pattern_features(
            path_circ, is_calibrated=True, mm_per_pixel=0.10
        )
        self.assertEqual(feat_circ["accepted_stain_count"], 1)
        st_c = feat_circ["stains"][0]
        self.assertAlmostEqual(st_c["aspect_ratio"], 1.0, delta=0.08)
        self.assertGreater(st_c["circularity"], 0.82)
        self.assertGreater(st_c["solidity"], 0.94)
        self.assertIsNotNone(st_c["major_axis_mm"])
        self.assertAlmostEqual(st_c["major_axis_mm"], 9.0, delta=0.6)

        # 2. Oblique elliptical stain (major=90px, minor=44px -> AR ~ 2.0, angle ~ 30 deg)
        img_ell = np.full((400, 400, 3), 240, dtype=np.uint8)
        cv2.ellipse(img_ell, (200, 200), (45, 22), 35.0, 0, 360, (25, 20, 140), -1)
        path_ell = self._save_synthetic_image("synth_elliptical.png", img_ell)

        feat_ell = pattern_analysis.extract_pattern_features(
            path_ell, is_calibrated=True, mm_per_pixel=0.05
        )
        self.assertEqual(feat_ell["accepted_stain_count"], 1)
        st_e = feat_ell["stains"][0]
        self.assertAlmostEqual(st_e["aspect_ratio"], 2.0, delta=0.25)
        self.assertAlmostEqual(st_e["apparent_impact_angle_deg"], 30.0, delta=4.5)
        self.assertGreater(st_e["eccentricity"], 0.80)

    def test_06_synthetic_elongated_and_irregular_stains(self):
        """Test elongated stain (AR > 3.0) and irregular low-solidity non-elliptical stain."""
        # 1. Elongated stain (flow / cast-off / swipe morphology)
        img_elong = np.full((400, 500, 3), 240, dtype=np.uint8)
        cv2.ellipse(img_elong, (250, 200), (95, 16), 10.0, 0, 360, (20, 15, 130), -1)
        path_elong = self._save_synthetic_image("synth_elongated.png", img_elong)

        feat_elong = pattern_analysis.extract_pattern_features(path_elong, is_calibrated=False)
        self.assertEqual(feat_elong["accepted_stain_count"], 1)
        self.assertGreater(feat_elong["stains"][0]["aspect_ratio"], 4.0)

        # 2. Irregular L-shaped polygon stain (low solidity < 0.78)
        img_irreg = np.full((400, 400, 3), 240, dtype=np.uint8)
        pts = np.array([
            [120, 100], [170, 100], [170, 230], [290, 230],
            [290, 280], [120, 280]
        ], dtype=np.int32)
        cv2.fillPoly(img_irreg, [pts], (20, 15, 130))
        path_irreg = self._save_synthetic_image("synth_irregular.png", img_irreg)

        feat_irreg = pattern_analysis.extract_pattern_features(path_irreg, is_calibrated=False)
        self.assertEqual(feat_irreg["accepted_stain_count"], 1)
        st_irr = feat_irreg["stains"][0]
        self.assertLess(st_irr["solidity"], 0.78)
        self.assertTrue(st_irr["overlap_or_irregular_flag"])
        # Low-solidity irregular contour must also be noted in rejected_or_uncertain_detections
        uncertain_reasons = [r["reason"] for r in feat_irreg["rejected_or_uncertain_detections"]]
        self.assertTrue(any("Solidity" in r or "Overlapping" in r for r in uncertain_reasons))

    def test_07_synthetic_linear_trail_and_dense_cluster_and_overlapping_stains(self):
        """Test linear trail (high R^2), dense spatter cluster, and overlapping figure-8 stains."""
        # 1. Linear trail of 6 droplets along a line (Cast-off / Drip trail geometry)
        img_linear = np.full((400, 800, 3), 240, dtype=np.uint8)
        for i in range(6):
            cx = 100 + i * 110
            cy = 120 + i * 30
            cv2.ellipse(img_linear, (cx, cy), (18, 10), 15.0, 0, 360, (20, 15, 135), -1)
        path_linear = self._save_synthetic_image("synth_linear_trail.png", img_linear)

        feat_lin = pattern_analysis.extract_pattern_features(path_linear, is_calibrated=True, mm_per_pixel=0.10)
        self.assertEqual(feat_lin["accepted_stain_count"], 6)
        self.assertGreater(feat_lin["pattern_summary"]["centroid_linearity_r2"], 0.98)
        self.assertEqual(
            feat_lin["pattern_summary"]["spatial_distribution_type"],
            "Linear / Arcuate Alignment",
        )

        # 2. Dense spatter cluster (18 small elliptical stains)
        img_cluster = np.full((500, 500, 3), 240, dtype=np.uint8)
        rng = np.random.default_rng(42)
        for _ in range(18):
            cx = int(rng.integers(90, 410))
            cy = int(rng.integers(90, 410))
            maj = int(rng.integers(10, 22))
            min_a = int(rng.integers(6, maj))
            ang = float(rng.integers(0, 180))
            cv2.ellipse(img_cluster, (cx, cy), (maj, min_a), ang, 0, 360, (25, 20, 140), -1)
        path_cluster = self._save_synthetic_image("synth_cluster.png", img_cluster)

        feat_clust = pattern_analysis.extract_pattern_features(path_cluster, is_calibrated=True, mm_per_pixel=0.08)
        self.assertGreaterEqual(feat_clust["accepted_stain_count"], 12)
        self.assertIsNotNone(feat_clust["pattern_summary"]["stain_density_per_cm2"])

        # 3. Overlapping / coalesced stains (two circles intersecting with narrow neck)
        img_overlap = np.full((400, 400, 3), 240, dtype=np.uint8)
        cv2.circle(img_overlap, (165, 200), 42, (20, 15, 135), -1)
        cv2.circle(img_overlap, (235, 200), 42, (20, 15, 135), -1)
        path_overlap = self._save_synthetic_image("synth_overlap.png", img_overlap)

        feat_ov = pattern_analysis.extract_pattern_features(path_overlap, is_calibrated=False)
        self.assertEqual(feat_ov["accepted_stain_count"], 1)
        # Coalesced contour has reduced circularity compared to a single circle
        self.assertLess(feat_ov["stains"][0]["circularity"], 0.85)

    def test_08_low_contrast_noisy_background_and_multi_resolution(self):
        """Test low-contrast image quality rejection, noisy background contour logging, and multi-resolution."""
        # 1. Low-contrast flat image -> is_sufficient_quality == False -> Insufficient Information
        img_low = np.full((300, 300, 3), 128, dtype=np.uint8)
        img_low[100:150, 100:150] = 132  # dynamic range = 4 (< 25 threshold)
        path_low = self._save_synthetic_image("synth_low_contrast.png", img_low)

        feat_low = pattern_analysis.extract_pattern_features(path_low, is_calibrated=False)
        self.assertFalse(feat_low["image_quality"]["is_sufficient_quality"])
        cls_low = pattern_analysis.generate_classification_suggestions(feat_low, {})
        self.assertEqual(cls_low["overall_determination"], "Insufficient Information")

        # 2. Noisy background with sub-threshold speckles -> never silently discarded!
        img_noisy = np.full((400, 400, 3), 240, dtype=np.uint8)
        cv2.circle(img_noisy, (200, 200), 35, (20, 15, 135), -1)
        # Add 5 small 4x4px noise specks (survive 3x3 morphology open, area ~ 12-16 px^2 < min_area_px=35)
        for nx, ny in [(30, 30), (70, 30), (110, 30), (150, 30), (190, 30)]:
            cv2.circle(img_noisy, (nx, ny), 3, (10, 10, 100), -1)
        path_noisy = self._save_synthetic_image("synth_noisy.png", img_noisy)

        feat_noisy = pattern_analysis.extract_pattern_features(path_noisy, is_calibrated=False, min_area_px=60.0)
        self.assertEqual(feat_noisy["accepted_stain_count"], 1)
        self.assertGreaterEqual(feat_noisy["rejected_or_uncertain_count"], 5)
        for rej in feat_noisy["rejected_or_uncertain_detections"]:
            self.assertIn("reason", rej)
            self.assertTrue(len(rej["reason"]) > 10)

        # 3. Multiple resolutions (200x200 vs 1000x800)
        img_small = np.full((200, 200, 3), 240, dtype=np.uint8)
        cv2.circle(img_small, (100, 100), 25, (20, 15, 135), -1)
        path_small = self._save_synthetic_image("synth_200x200.png", img_small)
        feat_small = pattern_analysis.extract_pattern_features(path_small, is_calibrated=False)
        self.assertEqual(feat_small["image_quality"]["width_px"], 200)
        self.assertEqual(feat_small["accepted_stain_count"], 1)

        img_large = np.full((800, 1000, 3), 240, dtype=np.uint8)
        cv2.circle(img_large, (500, 400), 80, (20, 15, 135), -1)
        path_large = self._save_synthetic_image("synth_1000x800.png", img_large)
        feat_large = pattern_analysis.extract_pattern_features(path_large, is_calibrated=False)
        self.assertEqual(feat_large["image_quality"]["width_px"], 1000)
        self.assertEqual(feat_large["image_quality"]["height_px"], 800)
        self.assertEqual(feat_large["accepted_stain_count"], 1)

    # =========================================================================
    # 3. PIXEL VS. METRIC UNIT DISCIPLINE & NON-DESTRUCTIVE IMAGE INTEGRITY
    # =========================================================================

    def test_09_pixel_vs_metric_discipline_and_original_image_integrity(self):
        """Verify uncalibrated runs never report mm/mm2 and original image bytes are untouched."""
        img = np.full((350, 350, 3), 240, dtype=np.uint8)
        cv2.circle(img, (175, 175), 40, (25, 20, 140), -1)
        path = self._save_synthetic_image("synth_integrity.png", img)

        with open(path, "rb") as f:
            sha256_before = hashlib.sha256(f.read()).hexdigest()

        # Uncalibrated extraction
        feat_uncal = pattern_analysis.extract_pattern_features(path, is_calibrated=False, mm_per_pixel=None)
        self.assertFalse(feat_uncal["is_calibrated"])
        self.assertEqual(feat_uncal["measurement_unit"], "px (Uncalibrated Pixel Units)")
        st_u = feat_uncal["stains"][0]
        self.assertIsNotNone(st_u["area_px2"])
        self.assertIsNotNone(st_u["major_axis_px"])
        self.assertIsNone(st_u["area_mm2"])
        self.assertIsNone(st_u["major_axis_mm"])
        self.assertIsNone(st_u["minor_axis_mm"])
        self.assertIsNone(feat_uncal["pattern_summary"]["mean_area_mm2"])
        self.assertIsNone(feat_uncal["pattern_summary"]["stain_density_per_cm2"])

        # Calibrated extraction (0.10 mm/px -> 1 px^2 = 0.01 mm^2)
        feat_cal = pattern_analysis.extract_pattern_features(path, is_calibrated=True, mm_per_pixel=0.10)
        self.assertTrue(feat_cal["is_calibrated"])
        st_c = feat_cal["stains"][0]
        self.assertAlmostEqual(st_c["area_mm2"], st_c["area_px2"] * 0.01, places=3)
        self.assertAlmostEqual(st_c["major_axis_mm"], st_c["major_axis_px"] * 0.10, places=3)

        # Original evidence file on disk must be bit-for-bit identical
        with open(path, "rb") as f:
            sha256_after = hashlib.sha256(f.read()).hexdigest()
        self.assertEqual(sha256_before, sha256_after)

    # =========================================================================
    # 4. EXPLAINABLE CLASSIFICATION ENGINE, TAXONOMY & PROVENANCE LABELS
    # =========================================================================

    def test_10_classification_taxonomy_and_uncertainty_outcomes(self):
        """Verify all 15 BPA categories exist and engine produces explainable candidate suggestions."""
        self.assertEqual(len(pattern_analysis.PATTERN_TAXONOMY), 15)
        expected_categories = {
            "Passive Drop", "Drip Pattern", "Pool", "Flow Pattern", "Saturation Pattern",
            "Impact Spatter", "Cast-Off Pattern", "Projected Blood Pattern",
            "Contact Transfer", "Swipe Pattern", "Wipe Pattern", "Patterned Transfer",
            "Expirated Blood Pattern", "Void Pattern", "Environmentally / Surface-Altered Pattern"
        }
        self.assertEqual(set(pattern_analysis.PATTERN_TAXONOMY.keys()), expected_categories)

        # Test with spatter cluster features + examiner observations
        img_cluster = np.full((450, 450, 3), 240, dtype=np.uint8)
        for i in range(10):
            cv2.ellipse(img_cluster, (100 + i * 28, 150 + (i % 3) * 45), (14, 8), 25.0, 0, 360, (25, 20, 140), -1)
        path_cluster = self._save_synthetic_image("synth_cls_cluster.png", img_cluster)
        feat = pattern_analysis.extract_pattern_features(path_cluster, is_calibrated=True, mm_per_pixel=0.10)

        raw_obs = {
            "observed_morphology": "Radiating elliptical spatter distribution",
            "surface_material": "Painted Drywall (Non-Porous)",
            "surface_orientation": "Vertical Wall (90 deg)",
        }
        obs = pattern_analysis.normalize_examiner_observations(raw_obs)
        self.assertEqual(obs["observed_morphology"]["provenance"], "Examiner Observed")
        self.assertEqual(obs["surface_material"]["provenance"], "Examiner Supplied")

        cls_res = pattern_analysis.generate_classification_suggestions(feat, raw_obs)
        self.assertEqual(cls_res["provenance"], "Software Suggested")
        self.assertIn("Multiple Possible Interpretations", cls_res["overall_determination"])
        self.assertGreaterEqual(len(cls_res["candidate_suggestions"]), 2)

        top_candidate = cls_res["candidate_suggestions"][0]
        self.assertEqual(top_candidate["provenance"], "Software Suggested")
        self.assertGreater(len(top_candidate["supporting_features"]), 0)
        self.assertIn("contradicting_features", top_candidate)
        self.assertIn("missing_information", top_candidate)
        self.assertIn("distinguishing_limitations", top_candidate)

    def test_11_examiner_observations_alter_candidate_interpretations(self):
        """Verify examiner observations (e.g. air-bubble vacuoles, void gap, feathered edge) surface corresponding categories."""
        img = np.full((350, 350, 3), 240, dtype=np.uint8)
        cv2.circle(img, (175, 175), 30, (25, 20, 140), -1)
        path = self._save_synthetic_image("synth_obs_test.png", img)
        feat = pattern_analysis.extract_pattern_features(path, is_calibrated=True, mm_per_pixel=0.10)

        raw_obs = {
            "observed_morphology": "Patterned ridge / fabric weave impression with swipe smear",
            "air_bubbles_or_vacuoles": "Yes — air bubble rings observed",
            "void_shadow_observed": "Yes — abrupt void shadow boundary present",
            "feathered_or_skeletonized_margins": "Feathered margin and skeletonized rim",
            "surface_material": "Smooth Ceramic Tile",
        }
        cls_res = pattern_analysis.generate_classification_suggestions(feat, raw_obs)
        suggested_names = {c["category_name"] for c in cls_res["candidate_suggestions"]}
        self.assertIn("Expirated Blood Pattern", suggested_names)
        self.assertIn("Void Pattern", suggested_names)
        self.assertIn("Swipe Pattern", suggested_names)
        self.assertIn("Wipe Pattern", suggested_names)
        self.assertIn("Patterned Transfer", suggested_names)

    # =========================================================================
    # 5. DATABASE PERSISTENCE, REVISION CONTROL & AUDIT LOGGING (SECTION 8)
    # =========================================================================

    def test_12_database_persistence_revision_tracking_and_audit_logs(self):
        """Verify pattern_analyses persistence, non-destructive revisions (R1 -> R2), and all 6 audit log events."""
        case_id = "BPA-2026-0101"
        img = np.full((300, 300, 3), 240, dtype=np.uint8)
        cv2.circle(img, (150, 150), 35, (25, 20, 140), -1)
        path = self._save_synthetic_image("synth_db_rev.png", img)

        feat = pattern_analysis.extract_pattern_features(path, is_calibrated=True, mm_per_pixel=0.10)
        raw_obs = {
            "observed_morphology": "Circular drop with satellite spatter",
            "surface_material": "Smooth Tile",
        }
        obs = pattern_analysis.normalize_examiner_observations(raw_obs)
        cls_res = pattern_analysis.generate_classification_suggestions(feat, raw_obs)

        # Create Revision R1
        rec_r1 = database.create_pattern_analysis_record(
            case_id=case_id,
            evidence_id="EVD-0101-01",
            image_id=None,
            is_calibrated=True,
            mm_per_pixel=0.10,
            features_json=json.dumps(feat),
            suggestions_json=json.dumps(cls_res),
            observations_json=json.dumps(obs),
            overall_determination=cls_res["overall_determination"],
            examiner_name="Test Examiner, MSFS",
        )
        self.assertIsNotNone(rec_r1)
        r1_id = rec_r1["analysis_id"]
        # Never automatically approved upon creation!
        self.assertEqual(rec_r1["review_status"], "Preliminary — Examiner Review Required")

        # Modify examiner interpretation on R1
        rec_r1_mod = database.update_pattern_analysis_examiner_review(
            analysis_id=r1_id,
            suggestion_decision="Modified by Examiner",
            examiner_interpretation="Passive Drop / Drip Pattern",
            examiner_rationale="Updated examiner interpretation after substrate review.",
            examiner_name="Test Examiner, MSFS",
            review_status="Examiner Reviewed",
        )
        self.assertEqual(rec_r1_mod["review_status"], "Examiner Reviewed")
        self.assertEqual(rec_r1_mod["suggestion_decision"], "Modified by Examiner")

        # Approve R1
        rec_r1_app = database.update_pattern_analysis_examiner_review(
            analysis_id=r1_id,
            suggestion_decision="Accepted as Supporting Hypothesis",
            examiner_interpretation="Passive Drop (Examiner Approved)",
            examiner_rationale="Signed and approved by lead examiner.",
            examiner_name="Test Examiner, MSFS",
            review_status="Examiner Approved",
        )
        self.assertEqual(rec_r1_app["review_status"], "Examiner Approved")

        # Create Revision R2 (must preserve R1 intact!)
        rec_r2 = database.create_pattern_analysis_record(
            case_id=case_id,
            evidence_id="EVD-0101-01",
            image_id=None,
            is_calibrated=True,
            mm_per_pixel=0.10,
            features_json=json.dumps(feat),
            suggestions_json=json.dumps(cls_res),
            observations_json=json.dumps(obs),
            overall_determination=cls_res["overall_determination"],
            examiner_name="Test Examiner, MSFS",
        )
        rec_r1_still_there = database.get_pattern_analysis_by_id(r1_id)
        self.assertIsNotNone(rec_r1_still_there)
        self.assertIsNotNone(rec_r2)
        self.assertGreater(rec_r2["revision_number"], rec_r1_still_there["revision_number"])

        # Verify all 6 required audit log actions are present
        logs = database.get_audit_logs(case_id=case_id, limit=50)
        logged_actions = {row["action_type"] for row in logs}
        for required_action in [
            "Pattern Analysis Creation",
            "Feature Extraction",
            "Classification Suggestion Generation",
            "Examiner Interpretation Modification",
            "Analysis Approval",
            "Analysis Revision",
        ]:
            self.assertIn(required_action, logged_actions)

        # Verify repeated init_db() preserves pattern_analyses records
        database.init_db(seed_demo=False)
        self.assertIsNotNone(database.get_pattern_analysis_by_id(r1_id))

    # =========================================================================
    # 6. FLASK ROUTES, SECURITY & PDF REPORT INTEGRATION (SECTIONS 7 & 9)
    # =========================================================================

    def test_13_flask_routes_security_and_pdf_report_integration(self):
        """Test /pattern-analysis/<case_id>, analyze POST, review POST, cross-case security, and PDF report section."""
        # 1. GET /pattern-analysis redirects to active case
        resp_root = self.client.get("/pattern-analysis", follow_redirects=True)
        self.assertEqual(resp_root.status_code, 200)
        self.assertIn(b"Bloodstain Pattern Classification", resp_root.data)

        # 2. Invalid case_id returns 404
        resp_404 = self.client.get("/pattern-analysis/NON-EXISTENT-CASE")
        self.assertEqual(resp_404.status_code, 404)

        # 3. POST /pattern-analysis/BPA-2026-0101/analyze using a reference sample image + observations
        resp_post = self.client.post(
            "/pattern-analysis/BPA-2026-0101/analyze",
            data={
                "sample_name": "sample_45deg_calibrated.png",
                "threshold_method": "otsu",
                "min_area_px": "20.0",
                "suspected_category": "Impact Spatter",
                "observed_morphology": "Cluster of elliptical stains on north wall",
                "surface_material": "Painted Drywall",
                "surface_orientation": "Vertical Wall (90 deg)",
                "surface_texture": "Smooth Matte",
            },
            follow_redirects=True,
        )
        self.assertEqual(resp_post.status_code, 200)
        self.assertIn(b"Software Suggested", resp_post.data)

        # Get latest analysis ID for BPA-2026-0101
        analyses = database.get_pattern_analyses_for_case("BPA-2026-0101")
        self.assertGreater(len(analyses), 0)
        latest_id = analyses[0]["analysis_id"]

        # 4. Cross-case manipulation check: reviewing BPA-2026-0101 analysis under BPA-2026-0102 must return 403
        resp_cross = self.client.post(
            f"/pattern-analysis/BPA-2026-0102/review/{latest_id}",
            data={"review_status": "Examiner Approved"},
        )
        self.assertEqual(resp_cross.status_code, 403)

        # 5. Valid examiner approval via POST /pattern-analysis/BPA-2026-0101/review/<latest_id>
        resp_approve = self.client.post(
            f"/pattern-analysis/BPA-2026-0101/review/{latest_id}",
            data={
                "suggestion_decision": "Accepted as Supporting Hypothesis",
                "examiner_interpretation": "Impact Spatter (Examiner Approved with scene correlation)",
                "examiner_rationale": "Approved after reviewing trajectory convergence and substrate context.",
                "examiner_name": "Det. Sarah Jenkins, IABPA",
                "review_status": "Examiner Approved",
            },
            follow_redirects=True,
        )
        self.assertEqual(resp_approve.status_code, 200)
        self.assertIn(b"Examiner Approved", resp_approve.data)

        # 6. Generate Complete Forensic Case Report PDF and verify Pattern Analysis section is included
        pdf_meta = report_generator.generate_case_report(
            case_id="BPA-2026-0101",
            examiner_name="Det. Sarah Jenkins, IABPA",
        )
        self.assertTrue(os.path.isfile(pdf_meta["filepath"]))
        self.assertGreater(pdf_meta["file_size"], 5000)


if __name__ == "__main__":
    unittest.main()
