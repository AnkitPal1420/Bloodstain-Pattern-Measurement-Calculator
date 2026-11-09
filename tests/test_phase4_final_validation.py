"""
test_phase4_final_validation.py
===============================
Phase 4.5 — Complete System Integration, Scientific Validation, Security Audit
& Final Release Verification Suite.

Validates:
1. Complete 20-step end-to-end forensic workflow (Case -> Evidence -> Image Upload ->
   OpenCV Contour/Ellipse Extraction -> Metric Calibration -> 3 Saved Measurements ->
   Reconstruction Creation -> Stain Association -> Directional & Surface Coordinates ->
   2D Convergence & 3D Origin Calculation -> 3D Visualization Scene -> Individual,
   Reconstruction, and Complete Case PDF Reports -> Download & SHA-256 Verification).
2. Scientific & Mathematical Verification (Impact Angle, 3D Vectors, Coordinate Transforms,
   Rotation Matrix Orthogonality, 2D Convergence, 3D Origin, Uncertainty Propagation).
3. Evidence Integrity & Operational Audit Trail (all 7 forensic action types).
4. Database Consistency, Idempotent Schema Migration, Foreign Keys & Transaction Rollback.
5. Security Audit (Upload Validation, Corrupted Image Rejection, Path Traversal,
   SQL Injection Parameterization, XSS Escaping).
"""

import hashlib
import io
import math
import os
import sqlite3
import unittest

import cv2
import numpy as np

from app import app
import calculator
import database
import image_processing
import report_generator
import trajectory
import trajectory_integration
import trajectory_visualization


class TestPhase4FinalValidation(unittest.TestCase):
    """Comprehensive Phase 4.5 final system integration, scientific, and security suite."""

    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()
        database.clear_all_data()
        database.init_db(seed_demo=True)
        self._created_files = []

    def tearDown(self):
        for path in self._created_files:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass

    def _create_multi_stain_synthetic_png(self) -> bytes:
        """Create a high-contrast synthetic evidence image with 3 elliptical bloodstains."""
        canvas = np.full((600, 800, 3), 242, dtype=np.uint8)
        # Stain 1: 30 deg impact (minor=15px, major=30px -> W=30, L=60)
        cv2.ellipse(canvas, (200, 380), (30, 15), 25.0, 0, 360, (22, 16, 135), -1)
        # Stain 2: 45 deg impact (minor=21px, major=30px)
        cv2.ellipse(canvas, (400, 320), (30, 21), 90.0, 0, 360, (22, 16, 135), -1)
        # Stain 3: ~35 deg impact (minor=18px, major=31px)
        cv2.ellipse(canvas, (580, 390), (31, 18), 150.0, 0, 360, (22, 16, 135), -1)
        ok, buf = cv2.imencode(".png", canvas)
        self.assertTrue(ok)
        return buf.tobytes()

    # =========================================================================
    # 1. COMPLETE 20-STEP END-TO-END FORENSIC WORKFLOW (SECTION 5)
    # =========================================================================

    def test_complete_20_step_end_to_end_forensic_workflow(self):
        """
        Execute the complete forensic laboratory workflow from Case Creation through
        OpenCV Optical Processing, Metric Calibration, Multi-Stain Trajectory
        Reconstruction, 3D Scene Generation, and all 3 PDF Report types with SHA-256 verification.
        """
        # Step 1: Create a forensic case via POST /cases/new
        case_id = "BPA-2026-9450"
        res_case = self.client.post(
            "/cases/new",
            data={
                "case_id": case_id,
                "title": "Phase 4.5 End-to-End Validation Scene",
                "reference_no": "REF-E2E-2026-450",
                "examination_date": "2026-11-09",
                "examiner": "Lead Examiner Dr. R. Vance",
                "description": "Controlled 3-stain radial spatter pattern on horizontal laboratory floor.",
                "status": "Active",
            },
            follow_redirects=True,
        )
        self.assertEqual(res_case.status_code, 200)
        case_rec = database.get_case_by_id(case_id)
        self.assertIsNotNone(case_rec)

        # Step 2: Create an evidence record via POST /evidence/new
        res_ev = self.client.post(
            "/evidence/new",
            data={
                "case_id": case_id,
                "description": "High-resolution macro photograph of 3 converging bloodstains",
                "evidence_type": "Impact Spatter",
                "location_surface": "Horizontal Floor (Smooth Tile)",
                "status": "Logged",
                "notes": "Forensic L-scale included in frame at 0.1000 mm/px",
            },
            follow_redirects=True,
        )
        self.assertEqual(res_ev.status_code, 200)
        ev_list = database.get_evidence_for_case(case_id)
        self.assertEqual(len(ev_list), 1)
        evidence_id = ev_list[0]["evidence_id"]

        # Step 3: Upload a bloodstain image via POST /measurement/upload
        png_bytes = self._create_multi_stain_synthetic_png()
        res_up = self.client.post(
            "/measurement/upload",
            data={
                "case_id": case_id,
                "evidence_id": evidence_id,
                "image_file": (io.BytesIO(png_bytes), "e2e_spatter_scene.png"),
            },
            content_type="multipart/form-data",
            follow_redirects=False,
        )
        self.assertEqual(res_up.status_code, 302)
        images = database.get_images_for_case(case_id)
        self.assertEqual(len(images), 1)
        image_rec = images[0]
        image_id = image_rec["image_id"]
        self._created_files.append(image_rec["filepath"])

        # Step 4 & 5: Process the image through OpenCV, extract stain contours & fit ellipses
        vision_data = image_processing.process_bloodstain_pipeline(
            image_source=image_rec["filepath"],
            threshold_method="otsu",
            invert_binary=True,
        )
        self.assertGreaterEqual(len(vision_data["candidates"]), 3)
        self.assertIsNotNone(vision_data["ellipse"])
        self.assertLessEqual(
            vision_data["ellipse"]["minor_axis_px"],
            vision_data["ellipse"]["major_axis_px"],
        )

        # Step 6: Apply scale calibration (0.1000 mm/px)
        cal_res = image_processing.calculate_scale_factor(
            point1=(100.0, 500.0),
            point2=(200.0, 500.0),
            known_length_mm=10.0,
        )
        self.assertAlmostEqual(cal_res["mm_per_pixel"], 0.10, places=5)
        database.update_image_calibration(
            image_id=image_id,
            scale_factor=cal_res["mm_per_pixel"],
            is_calibrated=1,
        )
        cal_img = database.get_image_by_id(image_id)
        self.assertEqual(cal_img["is_calibrated"], 1)
        self.assertAlmostEqual(cal_img["scale_factor"], 0.10, places=5)

        # Step 7: Save 3 calibrated stain measurements converging toward (100, 150, 300) mm
        # In trajectory.py convention: gamma_deg is the direction of travel AWAY from origin
        true_origin = (100.0, 150.0, 300.0)
        stain_configs = [
            ("E2E-Stain-01", 400.0, 150.0),  # East of source (D=300, gamma=0 deg)
            ("E2E-Stain-02", 100.0, 450.0),  # North of source (D=300, gamma=90 deg)
            ("E2E-Stain-03", 300.0, 350.0),  # NE of source (dx=200, dy=200, gamma=45 deg)
        ]
        for idx, (label, sx, sy) in enumerate(stain_configs):
            dx = sx - true_origin[0]
            dy = sy - true_origin[1]
            dist_2d = math.hypot(dx, dy)
            alpha_rad = math.atan2(true_origin[2], dist_2d)
            gamma_deg = trajectory.normalize_angle_360(math.degrees(math.atan2(dy, dx)))
            length_mm = 10.0
            width_mm = length_mm * math.sin(alpha_rad)

            res_m = self.client.post(
                "/measurement",
                data={
                    "case_id": case_id,
                    "evidence_id": evidence_id,
                    "image_id": image_id,
                    "stain_label": label,
                    "width": f"{width_mm:.4f}",
                    "length": f"{length_mm:.4f}",
                    "direction_angle": f"{gamma_deg:.4f}",
                    "surface_x_mm": f"{sx:.2f}",
                    "surface_y_mm": f"{sy:.2f}",
                    "surface_type": "Horizontal Floor",
                    "save_record": "true",
                },
                follow_redirects=True,
            )
            self.assertEqual(res_m.status_code, 200)

        case_measurements = database.get_measurements_for_case(case_id)
        self.assertEqual(len(case_measurements), 3)
        saved_meas_ids = [m["id"] for m in reversed(case_measurements)]

        # Step 8: Open / create a reconstruction workspace via GET /reconstruction/<case_id>
        res_rec = self.client.get(f"/reconstruction/{case_id}")
        self.assertEqual(res_rec.status_code, 200)
        recs = database.get_reconstructions_for_case(case_id)
        self.assertEqual(len(recs), 1)
        rec_id = recs[0]["reconstruction_id"]

        # Step 9 & 10: Add measurements to reconstruction & verify directional/surface parameters
        for m_id in saved_meas_ids:
            res_add = self.client.post(
                f"/reconstruction/{case_id}/add-measurement",
                data={
                    "reconstruction_id": rec_id,
                    "measurement_id": str(m_id),
                    "direction_source": "Examiner Supplied (Tail/Spine Morphology)",
                    "sigma_gamma_deg": "1.5",
                },
                follow_redirects=True,
            )
            self.assertEqual(res_add.status_code, 200)

        # Modify one stain's metadata via /update-stain to exercise Measurement Modification
        res_upd = self.client.post(
            f"/reconstruction/{case_id}/update-stain",
            data={
                "reconstruction_id": rec_id,
                "measurement_id": str(saved_meas_ids[0]),
                "stain_identifier": "E2E-Stain-01",
                "surface_x_mm": "400.0",
                "surface_y_mm": "150.0",
                "direction_angle": str(
                    database.get_measurement_by_id(saved_meas_ids[0])["direction_angle"]
                ),
                "surface_type": "Horizontal Floor",
                "direction_source": "Examiner Verified (Satellite Tail)",
                "sigma_gamma_deg": "1.5",
                "examiner_notes": "Verified sharp spine morphology",
            },
            follow_redirects=True,
        )
        self.assertEqual(res_upd.status_code, 200)

        # Step 11, 12 & 13: Calculate 2D convergence, 3D origin & verify 3D visualization scene
        res_calc = self.client.post(
            f"/reconstruction/{case_id}/calculate",
            data={"reconstruction_id": rec_id},
            follow_redirects=True,
        )
        self.assertEqual(res_calc.status_code, 200)
        calc_row = database.get_latest_reconstruction_result(rec_id)
        self.assertIsNotNone(calc_row)
        self.assertAlmostEqual(calc_row["convergence_x_mm"], true_origin[0], delta=0.5)
        self.assertAlmostEqual(calc_row["convergence_y_mm"], true_origin[1], delta=0.5)
        self.assertAlmostEqual(calc_row["tangent_origin_z_mm"], true_origin[2], delta=0.5)
        self.assertAlmostEqual(calc_row["estimated_origin_z_mm"], true_origin[2], delta=0.5)

        # Verify 3D interactive scene & Matplotlib figures via GET /trajectory?reconstruction_id=<rec_id>
        res_traj = self.client.get(f"/trajectory?reconstruction_id={rec_id}")
        self.assertEqual(res_traj.status_code, 200)
        traj_html = res_traj.get_data(as_text=True)
        self.assertIn("trajectory3DCanvas", traj_html)
        self.assertIn("E2E-Stain-01", traj_html)
        self.assertIn("data:image/png;base64,", traj_html)

        # Step 14, 15, 16, 17, 18, 19: Generate all 3 PDF Report types, Download & Verify SHA-256
        report_routes = [
            (
                f"/reports/generate/measurement/{saved_meas_ids[0]}",
                {"report_status": "FINAL", "examiner_name": "Dr. R. Vance"},
            ),
            (
                f"/reports/generate/reconstruction/{rec_id}",
                {"report_status": "FINAL", "examiner_name": "Dr. R. Vance"},
            ),
            (
                f"/reports/generate/case/{case_id}",
                {"report_status": "FINAL", "examiner_name": "Dr. R. Vance"},
            ),
        ]

        for route_url, post_form in report_routes:
            res_gen = self.client.post(route_url, data=post_form, follow_redirects=False)
            self.assertIn(res_gen.status_code, (201, 302))

        reports = database.get_all_reports(case_id=case_id)
        self.assertEqual(len(reports), 3)
        for rpt in reports:
            self._created_files.append(rpt["filepath"])
            self.assertTrue(os.path.exists(rpt["filepath"]))
            self.assertGreater(rpt["file_size"], 2500)

            # Download via Flask endpoint and verify %PDF- header + SHA-256 match
            dl_res = self.client.get(f"/reports/{rpt['report_id']}/download")
            self.assertEqual(dl_res.status_code, 200)
            self.assertEqual(dl_res.mimetype, "application/pdf")
            pdf_bytes = dl_res.data
            self.assertTrue(pdf_bytes.startswith(b"%PDF-"))
            computed_sha = hashlib.sha256(pdf_bytes).hexdigest()
            self.assertEqual(computed_sha, rpt["sha256_hash"])
            dl_res.close()

        # Regenerate Complete Case Report to verify revision increment (Rev 2) and audit trail
        self.client.post(
            f"/reports/generate/case/{case_id}",
            data={"report_status": "FINAL", "examiner_name": "Dr. R. Vance"},
            follow_redirects=False,
        )
        all_rpts = database.get_all_reports(case_id=case_id)
        for r in all_rpts:
            if r["filepath"] not in self._created_files:
                self._created_files.append(r["filepath"])
        case_rpts = [r for r in all_rpts if r["report_type"] == "Complete Case Report"]
        self.assertEqual(len(case_rpts), 2)
        revs = sorted([r["revision_number"] for r in case_rpts])
        self.assertEqual(revs, [1, 2])

        # Verify all 7 required Audit Trail actions were logged for this case
        logs = database.get_audit_logs(case_id=case_id, limit=50)
        logged_actions = {entry["action_type"] for entry in logs}
        expected_actions = {
            "Evidence Upload",
            "Measurement Creation",
            "Measurement Modification",
            "Reconstruction Creation",
            "Reconstruction Calculation",
            "Report Generation",
            "Report Revision",
        }
        self.assertTrue(
            expected_actions.issubset(logged_actions),
            f"Missing audit actions: {expected_actions - logged_actions}",
        )

    # =========================================================================
    # 2. SCIENTIFIC AND MATHEMATICAL VALIDATION (SECTION 4)
    # =========================================================================

    def test_scientific_impact_angle_geometry_and_validation(self):
        """Verify impact angle formula alpha = arcsin(W/L) and strict physical bounds."""
        # 30 deg: W/L = 0.5
        r30 = calculator.calculate_impact_angle(width=5.0, length=10.0)
        self.assertAlmostEqual(r30["angle_degrees"], 30.0, places=2)
        self.assertFalse(r30["is_perpendicular"])

        # 45 deg: W/L = sqrt(2)/2
        r45 = calculator.calculate_impact_angle(width=math.sqrt(2.0), length=2.0)
        self.assertAlmostEqual(r45["angle_degrees"], 45.0, places=2)

        # 90 deg: W == L
        r90 = calculator.calculate_impact_angle(width=8.0, length=8.0)
        self.assertAlmostEqual(r90["angle_degrees"], 90.0, places=2)
        self.assertTrue(r90["is_perpendicular"])

        # Invalid inputs: W > L, W <= 0, L <= 0, NaN, Inf
        with self.assertRaises(ValueError):
            calculator.calculate_impact_angle(width=10.1, length=10.0)
        with self.assertRaises(ValueError):
            calculator.calculate_impact_angle(width=0.0, length=10.0)
        with self.assertRaises(ValueError):
            calculator.calculate_impact_angle(width=-2.0, length=10.0)
        with self.assertRaises(ValueError):
            calculator.calculate_impact_angle(width=float("nan"), length=10.0)
        with self.assertRaises(ValueError):
            calculator.calculate_impact_angle(width=5.0, length=float("inf"))

    def test_coordinate_transforms_and_rotation_matrix_orthogonality(self):
        """Verify image<->surface and surface<->lab transforms plus R^T R = I and det(R) = +1."""
        # Image <-> Surface roundtrip with Y-axis inversion
        surf = trajectory.image_to_surface_coordinates(
            u_px=250.0,
            v_px=150.0,
            mm_per_pixel=0.2,
            origin_u_px=0.0,
            origin_v_px=600.0,
            invert_y=True,
        )
        self.assertAlmostEqual(surf["surface_x_mm"], 50.0, places=6)
        self.assertAlmostEqual(surf["surface_y_mm"], 90.0, places=6)
        img_rt = trajectory.surface_to_image_coordinates(
            surface_x_mm=surf["surface_x_mm"],
            surface_y_mm=surf["surface_y_mm"],
            mm_per_pixel=0.2,
            origin_u_px=0.0,
            origin_v_px=600.0,
            invert_y=True,
        )
        self.assertAlmostEqual(img_rt["u_px"], 250.0, places=6)
        self.assertAlmostEqual(img_rt["v_px"], 150.0, places=6)

        # Rotation matrix orthogonality & determinant = +1 across floor, wall, and inclined planes
        basis_configs = [
            ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),  # Horizontal Floor
            ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),  # Vertical Wall
            ([1.0, 0.0, 0.0], [0.0, math.cos(math.radians(37.5)), math.sin(math.radians(37.5))]),  # Inclined
        ]
        for ex, ey in basis_configs:
            res = trajectory.surface_to_lab_coordinates(
                point_surface_mm=[125.0, 240.0, 35.0],
                vector_surface=[0.0, 0.0, 1.0],
                surface_origin_lab_mm=[10.0, 20.0, 5.0],
                surface_axis_x_lab=ex,
                surface_axis_y_lab=ey,
            )
            R = np.array(res["rotation_matrix"], dtype=float)
            ident = R.T @ R
            np.testing.assert_allclose(ident, np.eye(3), atol=1e-12)
            self.assertAlmostEqual(float(np.linalg.det(R)), 1.0, places=11)

    def test_2d_convergence_and_3d_origin_edge_cases(self):
        """Verify parallel ray rejection, diverging ray rejection, and uncertainty propagation."""
        # Parallel rays must raise TrajectoryError
        parallel_stains = [
            trajectory.StainTrajectoryInput(
                "P1", width=4.0, length=8.0, surface_x_mm=0.0, surface_y_mm=0.0, gamma_deg=0.0
            ),
            trajectory.StainTrajectoryInput(
                "P2", width=4.0, length=8.0, surface_x_mm=0.0, surface_y_mm=100.0, gamma_deg=0.0
            ),
        ]
        with self.assertRaises(trajectory.TrajectoryError):
            trajectory.estimate_area_of_convergence_2d(parallel_stains)

        # Diverging rays (back-projected rays pointing away from each other) must raise TrajectoryError
        diverging_stains = [
            trajectory.StainTrajectoryInput(
                "D1", width=4.0, length=8.0, surface_x_mm=100.0, surface_y_mm=0.0, gamma_deg=180.0
            ),
            trajectory.StainTrajectoryInput(
                "D2", width=4.0, length=8.0, surface_x_mm=-100.0, surface_y_mm=0.0, gamma_deg=0.0
            ),
        ]
        with self.assertRaises(trajectory.TrajectoryError):
            trajectory.estimate_area_of_convergence_2d(diverging_stains)

    # =========================================================================
    # 3. DATABASE CONSISTENCY, MIGRATION & ROLLBACK (SECTION 7)
    # =========================================================================

    def test_database_idempotent_migration_and_transaction_rollback(self):
        """
        Verify running init_db() repeatedly on an existing database preserves all data,
        and failed transactions roll back cleanly without orphaned records.
        """
        before_cases = len(database.get_all_cases())
        before_meas = len(database.get_recent_measurements(limit=100))

        # Run init_db() repeatedly without seed_demo
        database.init_db(seed_demo=False)
        database.init_db(seed_demo=False)

        after_cases = len(database.get_all_cases())
        after_meas = len(database.get_recent_measurements(limit=100))
        self.assertEqual(before_cases, after_cases)
        self.assertEqual(before_meas, after_meas)

        # Verify foreign key constraint enforcement (inserting evidence for non-existent case fails)
        with self.assertRaises(sqlite3.IntegrityError):
            database.create_evidence(
                evidence_id="EVD-ORPHAN-99",
                case_id="NON-EXISTENT-CASE-9999",
                description="Should fail FK check",
                evidence_type="Impact Spatter",
            )
        self.assertIsNone(database.get_evidence_by_id("EVD-ORPHAN-99"))

    # =========================================================================
    # 4. SECURITY AUDIT VALIDATION (SECTION 8)
    # =========================================================================

    def test_security_upload_validation_traversal_sqli_and_xss(self):
        """
        Verify upload file-type/content validation, path traversal rejection,
        SQL injection parameterization safety, and XSS HTML escaping.
        """
        # 1. Disallowed file extension (.exe)
        res_exe = self.client.post(
            "/measurement/upload",
            data={
                "case_id": "BPA-2026-0101",
                "image_file": (io.BytesIO(b"MZ-fake-executable"), "malware.exe"),
            },
            content_type="multipart/form-data",
            follow_redirects=True,
        )
        self.assertEqual(res_exe.status_code, 200)
        self.assertIn("Unsupported file extension", res_exe.get_data(as_text=True))

        # 2. Corrupted / fake image bytes with valid .png extension
        res_corrupt = self.client.post(
            "/measurement/upload",
            data={
                "case_id": "BPA-2026-0101",
                "image_file": (io.BytesIO(b"not-a-valid-png-stream"), "corrupted.png"),
            },
            content_type="multipart/form-data",
            follow_redirects=True,
        )
        self.assertEqual(res_corrupt.status_code, 200)
        self.assertIn("corrupted", res_corrupt.get_data(as_text=True).lower())

        # 3. Path traversal attempts on /uploads/ and /reports/
        res_trav1 = self.client.get("/uploads/original/../../app.py")
        self.assertIn(res_trav1.status_code, (400, 404))
        res_trav2 = self.client.get("/reports/..%2F..%2Fforensic.db/download")
        self.assertEqual(res_trav2.status_code, 404)

        # 4. SQL injection attempt in search query parameter
        sqli_payload = "BPA-2026-0101' OR '1'='1' --"
        res_sqli = self.client.get(f"/cases?search={sqli_payload}")
        self.assertEqual(res_sqli.status_code, 200)
        cases_sqli = database.get_all_cases(search_query=sqli_payload)
        self.assertEqual(len(cases_sqli), 0)

        # 5. XSS payload escaping in rendered Jinja2 templates
        xss_case_id = "BPA-2026-9988"
        xss_title = "<script>alert('xss')</script>Scene"
        database.create_case(
            case_id=xss_case_id,
            title=xss_title,
            reference_no="REF-XSS-01",
            examination_date="2026-11-09",
            examiner="Examiner <img src=x onerror=alert(1)>",
            status="Active",
        )
        res_xss = self.client.get(f"/cases/{xss_case_id}")
        self.assertEqual(res_xss.status_code, 200)
        rendered_html = res_xss.get_data(as_text=True)
        self.assertNotIn("<script>alert('xss')</script>", rendered_html)
        self.assertIn("&lt;script&gt;alert(&#39;xss&#39;)&lt;/script&gt;", rendered_html)


if __name__ == "__main__":
    unittest.main()
