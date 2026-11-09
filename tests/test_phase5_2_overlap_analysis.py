"""
tests/test_phase5_2_overlap_analysis.py
======================================
Comprehensive Automated Test Suite for Phase 5.2:
Advanced Overlapping & Complex Bloodstain Pattern Analysis

Validates:
1. Feature Mathematics & Domain Error Handling (Circularity, Aspect Ratio, Solidity, Eccentricity).
2. Connected Component & Topology Analysis (Isolated, Touching, Overlapping, Cluster, Irregular, Indeterminate).
3. Marker-Controlled Watershed Segmentation & Over-Segmentation Avoidance.
4. Area Conservation Tracking (Delta A and ratio between parent and candidate children).
5. Strict Pixel vs. Calibrated Metric Discipline (No mm/mm² when uncalibrated).
6. Non-Destructive Examiner Review Decisions:
   - Accept: registers individual child stains.
   - Reject: retains parent region as an unresolved merged entity (does NOT manufacture fake boundaries).
   - Indeterminate: flags region as uncertain.
7. Database Persistence & Non-Destructive Revision Tracking (R1 -> R2).
8. Audit Trail Verification across all 8 required action types.
9. Flask Routes, Cross-Case Security (403), and Invalid ID Handling (404).
10. PDF Report Integration (Section 9C).
"""

import json
import math
import os
import tempfile
import unittest

import cv2
import numpy as np

import app as flask_app
import database
import overlap_analysis
import report_generator
from overlap_analysis import (
    REGION_TYPE_CLUSTER,
    REGION_TYPE_INDETERMINATE,
    REGION_TYPE_IRREGULAR,
    REGION_TYPE_ISOLATED,
    REGION_TYPE_OVERLAPPING,
    REGION_TYPE_TOUCHING,
    STATUS_ACCEPTED,
    STATUS_CANDIDATE_SEPARATION,
    STATUS_CLEARLY_SEPARATED,
    STATUS_INDETERMINATE,
    STATUS_REJECTED,
    STATUS_UNRESOLVED_MERGED,
    OverlapAnalysisError,
)


class TestPhase52OverlapAnalysis(unittest.TestCase):
    """Test suite verifying Phase 5.2 complex and overlapping stain analysis."""

    @classmethod
    def setUpClass(cls):
        database.init_db(seed_demo=True)
        flask_app.app.config["TESTING"] = True
        cls.client = flask_app.app.test_client()
        cls.temp_dir = tempfile.mkdtemp(prefix="bpa_phase52_")

    def setUp(self):
        conn = database.get_db_connection()
        conn.execute("DELETE FROM overlap_analyses;")
        conn.commit()
        conn.close()

    def _save_synthetic(self, name: str, img: np.ndarray) -> str:
        path = os.path.join(self.temp_dir, name)
        cv2.imwrite(path, img)
        return path

    # =========================================================================
    # 1. Feature Mathematics & Domain Error Handling
    # =========================================================================

    def test_01_feature_math_and_domain_errors(self):
        """Verify feature calculation formulas and domain boundary rejection."""
        # Circularity of circle (r=40 -> A=1600*pi, P=80*pi) -> 1.0
        r = 40.0
        circ = overlap_analysis.compute_circularity(math.pi * r * r, 2.0 * math.pi * r)
        self.assertAlmostEqual(circ, 1.0, places=4)

        # Aspect ratio of 30x10 -> 3.0
        self.assertAlmostEqual(overlap_analysis.compute_aspect_ratio(30.0, 10.0), 3.0, places=4)

        # Solidity of 80/100 -> 0.80
        self.assertAlmostEqual(overlap_analysis.compute_solidity(80.0, 100.0), 0.80, places=4)

        # Eccentricity of 30-deg ellipse (W/L = 0.5) -> sqrt(0.75) ~ 0.8660
        self.assertAlmostEqual(overlap_analysis.compute_eccentricity(20.0, 10.0), math.sqrt(0.75), places=4)

        # Non-positive domain inputs must raise OverlapAnalysisError
        with self.assertRaises(OverlapAnalysisError):
            overlap_analysis.compute_circularity(0.0, 10.0)
        with self.assertRaises(OverlapAnalysisError):
            overlap_analysis.compute_aspect_ratio(-5.0, 10.0)
        with self.assertRaises(OverlapAnalysisError):
            overlap_analysis.compute_solidity(10.0, 0.0)
        with self.assertRaises(OverlapAnalysisError):
            overlap_analysis.compute_eccentricity(0.0, 5.0)

    # =========================================================================
    # 2. Synthetic Morphologies: Isolated, Touching & Overlapping Stains
    # =========================================================================

    def test_02_synthetic_isolated_stain_no_split(self):
        """Verify an isolated circular stain is recognized as single-peak and not split."""
        img = np.full((300, 300, 3), 240, dtype=np.uint8)
        cv2.circle(img, (150, 150), 35, (25, 20, 140), -1)
        path = self._save_synthetic("synth_single_isolated.png", img)

        res = overlap_analysis.detect_and_segment_complex_regions(path, is_calibrated=False)
        self.assertEqual(res["total_regions"], 1)
        reg = res["regions"][0]
        self.assertEqual(reg["region_type"], REGION_TYPE_ISOLATED)
        self.assertEqual(reg["segmentation_status"], STATUS_CLEARLY_SEPARATED)
        self.assertEqual(reg["peak_count"], 1)
        self.assertEqual(len(reg["candidate_children"]), 0)

    def test_03_synthetic_touching_and_overlapping_stains(self):
        """Verify two overlapping disks are detected with 2 peaks and candidate watershed split."""
        img = np.full((400, 500, 3), 240, dtype=np.uint8)
        # Left circle r=40 at (175, 200), Right circle r=40 at (235, 200) -> 20px overlap
        cv2.circle(img, (175, 200), 40, (25, 20, 140), -1)
        cv2.circle(img, (235, 200), 40, (25, 20, 140), -1)
        path = self._save_synthetic("synth_overlap_pair.png", img)

        res = overlap_analysis.detect_and_segment_complex_regions(path, is_calibrated=False)
        self.assertEqual(res["total_regions"], 1)
        reg = res["regions"][0]
        self.assertIn(reg["region_type"], [REGION_TYPE_TOUCHING, REGION_TYPE_OVERLAPPING])
        self.assertEqual(reg["peak_count"], 2)
        self.assertGreaterEqual(reg["notch_count"], 1)
        self.assertEqual(reg["segmentation_status"], STATUS_CANDIDATE_SEPARATION)
        self.assertEqual(len(reg["candidate_children"]), 2)

        # Verify child IDs follow parent-child hierarchy
        self.assertEqual(reg["candidate_children"][0]["parent_region_id"], reg["region_id"])
        self.assertTrue(reg["candidate_children"][0]["child_id"].startswith(reg["region_id"]))
        self.assertTrue(reg["candidate_children"][1]["child_id"].startswith(reg["region_id"]))

        # Verify area conservation ratio > 85%
        self.assertGreater(reg["area_conservation_ratio"], 0.85)

    def test_04_synthetic_dense_cluster_and_irregular_region(self):
        """Verify dense cluster with multiple peaks and irregular non-elliptical morphology."""
        img = np.full((450, 450, 3), 240, dtype=np.uint8)
        # 3 intersecting circles forming a triad cluster
        cv2.circle(img, (200, 200), 32, (25, 20, 140), -1)
        cv2.circle(img, (245, 200), 30, (25, 20, 140), -1)
        cv2.circle(img, (220, 245), 28, (25, 20, 140), -1)
        path = self._save_synthetic("synth_triad_cluster.png", img)

        res = overlap_analysis.detect_and_segment_complex_regions(path, is_calibrated=False)
        self.assertEqual(res["total_regions"], 1)
        reg = res["regions"][0]
        self.assertGreaterEqual(reg["peak_count"], 2)
        self.assertEqual(reg["segmentation_status"], STATUS_CANDIDATE_SEPARATION)
        self.assertGreaterEqual(len(reg["candidate_children"]), 2)

    # =========================================================================
    # 3. Unit Discipline & Calibration Handling
    # =========================================================================

    def test_05_pixel_vs_metric_discipline(self):
        """Verify uncalibrated analysis reports px only; calibrated computes exact mm/mm2."""
        img = np.full((350, 350, 3), 240, dtype=np.uint8)
        cv2.circle(img, (150, 175), 35, (25, 20, 140), -1)
        cv2.circle(img, (205, 175), 35, (25, 20, 140), -1)
        path = self._save_synthetic("synth_calibration_test.png", img)

        # 1. Uncalibrated: metric fields must be None
        res_uncal = overlap_analysis.detect_and_segment_complex_regions(path, is_calibrated=False)
        self.assertFalse(res_uncal["is_calibrated"])
        self.assertEqual(res_uncal["measurement_unit"], "px (Uncalibrated Pixel Units)")
        reg_u = res_uncal["regions"][0]
        self.assertIsNone(reg_u["parent_area_mm2"])
        self.assertIsNone(reg_u["parent_minor_axis_mm"])
        self.assertIsNone(reg_u["parent_major_axis_mm"])
        for ch in reg_u["candidate_children"]:
            self.assertIsNone(ch["area_mm2"])
            self.assertIsNone(ch["minor_axis_mm"])

        # 2. Calibrated: 0.10 mm/px -> area_mm2 = area_px2 * 0.01
        res_cal = overlap_analysis.detect_and_segment_complex_regions(
            path, is_calibrated=True, mm_per_pixel=0.10
        )
        self.assertTrue(res_cal["is_calibrated"])
        reg_c = res_cal["regions"][0]
        self.assertIsNotNone(reg_c["parent_area_mm2"])
        self.assertAlmostEqual(reg_c["parent_area_mm2"], reg_c["parent_area_px2"] * 0.01, places=3)
        for ch in reg_c["candidate_children"]:
            self.assertIsNotNone(ch["area_mm2"])
            self.assertAlmostEqual(ch["area_mm2"], ch["area_px2"] * 0.01, places=3)

    # =========================================================================
    # 4. Examiner Review Workflow: Acceptance, Rejection & Indeterminate
    # =========================================================================

    def test_06_examiner_review_accept_and_reject_workflows(self):
        """Verify examiner accept produces approved child stains, while reject retains merged region."""
        img = np.full((400, 500, 3), 240, dtype=np.uint8)
        cv2.circle(img, (175, 200), 40, (25, 20, 140), -1)
        cv2.circle(img, (235, 200), 40, (25, 20, 140), -1)
        path = self._save_synthetic("synth_review_test.png", img)

        res = overlap_analysis.detect_and_segment_complex_regions(path, is_calibrated=False)
        reg_id = res["regions"][0]["region_id"]

        # Case A: Examiner Accepts the split
        rev_accept = overlap_analysis.apply_examiner_review_actions(
            analysis_record={"regions": res["regions"], "is_calibrated": False},
            review_actions={reg_id: {"decision": "Accepted", "notes": "Clear visual constriction neck"}},
            examiner_name="Det. S. Jenkins",
            examiner_notes="Approved separation based on dual impact points.",
        )
        self.assertEqual(rev_accept["accepted_stains_count"], 2)
        self.assertEqual(rev_accept["unresolved_regions_count"], 0)
        self.assertEqual(rev_accept["regions"][0]["segmentation_status"], STATUS_ACCEPTED)

        # Case B: Examiner Rejects the split -> must retain parent as single merged entity (NEVER split)
        rev_reject = overlap_analysis.apply_examiner_review_actions(
            analysis_record={"regions": res["regions"], "is_calibrated": False},
            review_actions={reg_id: {"decision": "Rejected", "notes": "Boundary uncertain; keep as single transfer"}},
            examiner_name="Det. S. Jenkins",
            examiner_notes="Split rejected.",
        )
        self.assertEqual(rev_reject["accepted_stains_count"], 0)
        self.assertEqual(rev_reject["unresolved_regions_count"], 1)
        self.assertEqual(rev_reject["regions"][0]["segmentation_status"], STATUS_REJECTED)

        # Case C: Examiner marks Indeterminate
        rev_indet = overlap_analysis.apply_examiner_review_actions(
            analysis_record={"regions": res["regions"], "is_calibrated": False},
            review_actions={reg_id: {"decision": "Indeterminate", "notes": "Requires physical inspection"}},
        )
        self.assertEqual(rev_indet["regions"][0]["segmentation_status"], STATUS_INDETERMINATE)
        self.assertEqual(rev_indet["unresolved_regions_count"], 1)

    # =========================================================================
    # 5. Database Persistence, Non-Destructive Revisions & Audit Logging
    # =========================================================================

    def test_07_database_persistence_and_revisions(self):
        """Verify overlap_analyses persistence, non-destructive R1 -> R2, and all audit log actions."""
        case_id = "BPA-2026-0101"
        img = np.full((300, 300, 3), 240, dtype=np.uint8)
        cv2.circle(img, (120, 150), 30, (25, 20, 140), -1)
        cv2.circle(img, (170, 150), 30, (25, 20, 140), -1)
        path = self._save_synthetic("synth_db_overlap.png", img)

        res = overlap_analysis.detect_and_segment_complex_regions(path, is_calibrated=True, mm_per_pixel=0.10)

        # Create Revision R1
        rec_r1 = database.create_overlap_analysis_record(
            case_id=case_id,
            evidence_id="EVD-0101-01",
            image_id=None,
            algorithm_name="Marker Watershed",
            algorithm_version="v5.2",
            is_calibrated=True,
            mm_per_pixel=0.10,
            parameters_json=json.dumps({"min_area_px": 20.0}),
            summary_json=json.dumps(res),
            regions_json=json.dumps(res["regions"]),
            examiner_name="Det. S. Jenkins",
        )
        self.assertIsNotNone(rec_r1)
        r1_id = rec_r1["analysis_id"]
        self.assertEqual(rec_r1["revision_number"], 1)
        self.assertEqual(rec_r1["review_status"], "Preliminary — Examiner Review Required")

        # Update R1 with examiner review
        rec_r1_updated = database.update_overlap_analysis_review(
            analysis_id=r1_id,
            regions_json=json.dumps(res["regions"]),
            accepted_stains_json=json.dumps([]),
            review_status="Examiner Reviewed",
            examiner_name="Det. S. Jenkins",
            examiner_notes="Preliminary review saved.",
            action_type="Segmentation Accepted",
        )
        self.assertEqual(rec_r1_updated["review_status"], "Examiner Reviewed")

        # Create Revision R2 (must preserve R1 without overwrite!)
        rec_r2 = database.create_overlap_analysis_record(
            case_id=case_id,
            evidence_id="EVD-0101-01",
            image_id=None,
            algorithm_name="Marker Watershed",
            algorithm_version="v5.2",
            is_calibrated=True,
            mm_per_pixel=0.10,
            parameters_json=json.dumps({"min_area_px": 20.0}),
            summary_json=json.dumps(res),
            regions_json=json.dumps(res["regions"]),
            examiner_name="Det. S. Jenkins",
        )
        self.assertIsNotNone(rec_r2)
        self.assertEqual(rec_r2["revision_number"], 2)

        # Confirm R1 still exists intact
        rec_r1_still = database.get_overlap_analysis_by_id(r1_id)
        self.assertIsNotNone(rec_r1_still)
        self.assertEqual(rec_r1_still["revision_number"], 1)

        # Verify audit logs captured the actions
        logs = database.get_audit_logs(case_id=case_id, limit=50)
        logged_actions = {row["action_type"] for row in logs}
        for expected_action in [
            "Complex Image Analysis Created",
            "Connected Regions Detected",
            "Segmentation Generated",
            "Segmentation Accepted",
            "Segmentation Revision Saved",
        ]:
            self.assertIn(expected_action, logged_actions)

    # =========================================================================
    # 6. Flask Routes, Cross-Case Security & PDF Report Integration
    # =========================================================================

    def test_08_flask_routes_security_and_pdf_generation(self):
        """Verify /overlap-analysis routes, cross-case protection (403), 404 on bad case, and PDF report integration."""
        # 1. GET /overlap-analysis redirects to active case
        resp_root = self.client.get("/overlap-analysis", follow_redirects=True)
        self.assertEqual(resp_root.status_code, 200)
        self.assertIn(b"Overlapping, Touching", resp_root.data)

        # 2. Invalid case returns 404
        resp_404 = self.client.get("/overlap-analysis/INVALID-CASE-999")
        self.assertEqual(resp_404.status_code, 404)

        # 3. POST analyze on BPA-2026-0101 with sample image
        resp_post = self.client.post(
            "/overlap-analysis/BPA-2026-0101/analyze",
            data={
                "sample_name": "sample_overlap_pair.png",
                "min_area_px": "20.0",
                "min_peak_distance_px": "8.0",
                "peak_threshold_ratio": "0.35",
            },
            follow_redirects=True,
        )
        self.assertEqual(resp_post.status_code, 200)
        self.assertIn(b"Overlap Analysis OVR-", resp_post.data)

        # Retrieve latest analysis
        analyses = database.get_overlap_analyses_for_case("BPA-2026-0101")
        self.assertGreater(len(analyses), 0)
        latest_id = analyses[0]["analysis_id"]

        # 4. Cross-case manipulation check: accessing BPA-2026-0101 analysis under BPA-2026-0102 returns 403
        resp_cross = self.client.post(
            f"/overlap-analysis/BPA-2026-0102/review/{latest_id}",
            data={"decision_REG-01": "Accepted"},
        )
        self.assertEqual(resp_cross.status_code, 403)

        # 5. Valid examiner review submission
        resp_review = self.client.post(
            f"/overlap-analysis/BPA-2026-0101/review/{latest_id}",
            data={
                "decision_REG-01": "Accepted",
                "notes_REG-01": "Constriction neck verified visually.",
                "examiner_name": "Det. S. Jenkins",
                "examiner_notes": "Reviewed and signed off.",
            },
            follow_redirects=True,
        )
        self.assertEqual(resp_review.status_code, 200)
        self.assertIn(b"Examiner review decisions recorded", resp_review.data)

        # 6. Complete Forensic Case Report PDF generation containing Section 9C
        pdf_meta = report_generator.generate_case_report(
            case_id="BPA-2026-0101",
            examiner_name="Det. S. Jenkins",
        )
        self.assertTrue(os.path.isfile(pdf_meta["filepath"]))
        self.assertGreater(pdf_meta["file_size"], 5000)

    # =========================================================================
    # 7. Additional Scientific, Reliability & Audit Integrity Tests
    # =========================================================================

    def test_09_over_segmentation_avoidance_on_textured_circle(self):
        """Verify algorithm resists over-segmenting an isolated stain with boundary noise."""
        img = np.full((300, 300, 3), 240, dtype=np.uint8)
        cv2.circle(img, (150, 150), 40, (25, 20, 140), -1)
        # Add slight texture noise (3 small spikes along perimeter)
        for ang in [0.5, 2.1, 4.2]:
            px = int(150 + 42 * math.cos(ang))
            py = int(150 + 42 * math.sin(ang))
            cv2.circle(img, (px, py), 3, (25, 20, 140), -1)
        path = self._save_synthetic("synth_textured_circle.png", img)

        res = overlap_analysis.detect_and_segment_complex_regions(path, is_calibrated=False)
        self.assertEqual(res["total_regions"], 1)
        reg = res["regions"][0]
        # Must NOT split into multiple stains
        self.assertEqual(reg["region_type"], REGION_TYPE_ISOLATED)
        self.assertEqual(len(reg["candidate_children"]), 0)

    def test_10_area_conservation_and_edge_disparity_tracking(self):
        """Verify difference between parent area and child areas is accurately recorded."""
        img = np.full((350, 450, 3), 240, dtype=np.uint8)
        cv2.circle(img, (160, 175), 38, (25, 20, 140), -1)
        cv2.circle(img, (220, 175), 38, (25, 20, 140), -1)
        path = self._save_synthetic("synth_area_conservation.png", img)

        res = overlap_analysis.detect_and_segment_complex_regions(path, is_calibrated=True, mm_per_pixel=0.10)
        reg = res["regions"][0]
        self.assertEqual(len(reg["candidate_children"]), 2)

        parent_area = reg["parent_area_px2"]
        child_sum = sum(c["area_px2"] for c in reg["candidate_children"])
        # Area difference must match recorded area_conservation_diff_px2
        self.assertAlmostEqual(reg["area_conservation_diff_px2"], abs(parent_area - child_sum), places=1)
        # Ratio must be between 0.85 and 1.05 (allowing for discrete boundary vertex discretization)
        self.assertGreaterEqual(reg["area_conservation_ratio"], 0.85)
        self.assertLessEqual(reg["area_conservation_ratio"], 1.05)

    def test_11_audit_log_actions_complete_coverage(self):
        """Verify all Phase 5.2 audit action types are logged in audit_logs."""
        case_id = "BPA-2026-0101"
        img = np.full((300, 300, 3), 240, dtype=np.uint8)
        cv2.circle(img, (120, 150), 30, (25, 20, 140), -1)
        cv2.circle(img, (170, 150), 30, (25, 20, 140), -1)
        path = self._save_synthetic("synth_audit_full.png", img)

        res = overlap_analysis.detect_and_segment_complex_regions(path, is_calibrated=False)
        rec = database.create_overlap_analysis_record(
            case_id=case_id,
            evidence_id="EVD-0101-01",
            image_id=None,
            algorithm_name="Marker Watershed",
            algorithm_version="v5.2",
            is_calibrated=False,
            mm_per_pixel=None,
            parameters_json="{}",
            summary_json=json.dumps(res),
            regions_json=json.dumps(res["regions"]),
        )
        aid = rec["analysis_id"]

        # 1. Accepted action
        database.update_overlap_analysis_review(
            analysis_id=aid,
            regions_json=json.dumps(res["regions"]),
            accepted_stains_json=json.dumps([]),
            action_type="Segmentation Accepted",
        )
        # 2. Rejected action
        database.update_overlap_analysis_review(
            analysis_id=aid,
            regions_json=json.dumps(res["regions"]),
            accepted_stains_json=json.dumps([]),
            action_type="Segmentation Rejected",
        )
        # 3. Indeterminate action
        database.update_overlap_analysis_review(
            analysis_id=aid,
            regions_json=json.dumps(res["regions"]),
            accepted_stains_json=json.dumps([]),
            action_type="Region Marked Indeterminate",
        )

        logs = database.get_audit_logs(case_id=case_id, limit=50)
        action_names = {row["action_type"] for row in logs}
        self.assertIn("Complex Image Analysis Created", action_names)
        self.assertIn("Connected Regions Detected", action_names)
        self.assertIn("Segmentation Generated", action_names)
        self.assertIn("Segmentation Accepted", action_names)
        self.assertIn("Segmentation Rejected", action_names)
        self.assertIn("Region Marked Indeterminate", action_names)
        self.assertIn("Measurements Updated Following Approved Segmentation", action_names)

    def test_12_database_schema_idempotency(self):
        """Verify calling database.init_db() repeatedly does not corrupt or overwrite overlap_analyses."""
        case_id = "BPA-2026-0101"
        rec = database.create_overlap_analysis_record(
            case_id=case_id,
            evidence_id=None,
            image_id=None,
            algorithm_name="Marker Watershed",
            algorithm_version="v5.2",
            is_calibrated=False,
            mm_per_pixel=None,
            parameters_json="{}",
            summary_json="{}",
            regions_json="[]",
        )
        aid = rec["analysis_id"]

        # Call init_db without seed_demo
        database.init_db(seed_demo=False)
        rec_after = database.get_overlap_analysis_by_id(aid)
        self.assertIsNotNone(rec_after)
        self.assertEqual(rec_after["analysis_id"], aid)

    def test_13_pdf_report_contains_section_9c(self):
        """Verify generated case report PDF includes Section 9C Overlapping Analysis content."""
        case_id = "BPA-2026-0101"
        img = np.full((300, 300, 3), 240, dtype=np.uint8)
        cv2.circle(img, (120, 150), 30, (25, 20, 140), -1)
        cv2.circle(img, (170, 150), 30, (25, 20, 140), -1)
        path = self._save_synthetic("synth_pdf_ovr.png", img)

        res = overlap_analysis.detect_and_segment_complex_regions(path, is_calibrated=True, mm_per_pixel=0.10)
        database.create_overlap_analysis_record(
            case_id=case_id,
            evidence_id="EVD-0101-01",
            image_id=None,
            algorithm_name="Marker Watershed",
            algorithm_version="v5.2",
            is_calibrated=True,
            mm_per_pixel=0.10,
            parameters_json="{}",
            summary_json=json.dumps(res),
            regions_json=json.dumps(res["regions"]),
        )

        pdf_meta = report_generator.generate_case_report(case_id=case_id, examiner_name="Det. S. Jenkins")
        with open(pdf_meta["filepath"], "rb") as f:
            pdf_bytes = f.read()

        # PDF must contain the Section 9C marker text
        self.assertIn(b"OVERLAPPING & COMPLEX BLOODSTAIN PATTERN ANALYSIS", pdf_bytes)


if __name__ == "__main__":
    unittest.main()
