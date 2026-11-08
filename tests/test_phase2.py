"""
test_phase2.py
==============
Comprehensive automated tests for Phase 2:
- Dashboard routes & metrics
- Case Management (creation, ID generation, retrieval, validation, status updates)
- Evidence Workspace (listing, linking to cases, creation)
- Measurement Workspace (placeholders, calculation, persistence)
- Results Dashboard (measurement history, trajectory placeholders)
"""

import unittest
from app import app
import database


class TestPhase2Suite(unittest.TestCase):
    """Test suite covering all Phase 2 user interface, routes, and database models."""

    def setUp(self):
        """Set up Flask test client and ensure clean database state."""
        app.config["TESTING"] = True
        app.config["WTF_CSRF_ENABLED"] = False
        self.client = app.test_client()
        database.init_db(seed_demo=True)

    def test_case_id_format_and_uniqueness(self):
        """Verify unique case ID format (BPA-YYYY-NNNN) and uniqueness across calls."""
        id1 = database.generate_case_id()
        id2 = database.generate_case_id()
        self.assertTrue(id1.startswith("BPA-"))
        self.assertTrue(id2.startswith("BPA-"))
        self.assertNotEqual(id1, id2)
        parts = id1.split("-")
        self.assertEqual(len(parts), 3)
        self.assertEqual(len(parts[1]), 4)  # YYYY
        self.assertEqual(len(parts[2]), 4)  # NNNN

    def test_dashboard_metrics_and_routes(self):
        """GET / should display live database metrics, branding, and quick actions."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("Bloodstain Pattern Measurement Calculator", html)
        self.assertIn("Total Cases", html)
        self.assertIn("Active Investigations", html)
        self.assertIn("Evidence Items Logged", html)
        self.assertIn("Recent Forensic Case Activity", html)
        self.assertIn("Create New Case", html)

    def test_cases_list_and_filtering(self):
        """GET /cases should display case table and support search/filter."""
        # Baseline check
        response = self.client.get("/cases")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("Forensic Case Management Registry", html)
        self.assertIn("BPA-2026-0101", html)

        # Search check
        search_res = self.client.get("/cases?search=Elm")
        self.assertEqual(search_res.status_code, 200)
        search_html = search_res.get_data(as_text=True)
        self.assertIn("142 Elm Street", search_html)

        # Status filter check
        filter_res = self.client.get("/cases?status=Active")
        self.assertEqual(filter_res.status_code, 200)

    def test_create_case_flow(self):
        """POST /cases/new should create a new case and persist to database."""
        unique_cid = database.generate_case_id()
        unique_ref = f"REF-{unique_cid}"
        form_data = {
            "case_id": unique_cid,
            "title": "Automated Unit Test Case Scene",
            "reference_no": unique_ref,
            "examination_date": "2026-11-07",
            "examiner": "Test Agent",
            "description": "Test case description for automated test suite.",
            "status": "Active",
        }
        response = self.client.post("/cases/new", data=form_data, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn(unique_cid, html)
        self.assertIn("Automated Unit Test Case Scene", html)

        # Verify database retrieval
        case = database.get_case_by_id(unique_cid)
        self.assertIsNotNone(case)
        self.assertEqual(case["title"], "Automated Unit Test Case Scene")
        self.assertEqual(case["reference_no"], unique_ref)

    def test_create_case_validation(self):
        """POST /cases/new with missing mandatory fields should trigger validation error."""
        form_data = {
            "case_id": "",
            "title": "",  # Missing title
            "reference_no": "",  # Missing ref
            "examination_date": "",
            "examiner": "",
        }
        response = self.client.post("/cases/new", data=form_data, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("Please fill in all mandatory case fields", html)

    def test_case_detail_and_status_update(self):
        """Case detail view and status update should function correctly."""
        # Create case first with unique ID
        cid = database.generate_case_id()
        database.create_case(
            case_id=cid,
            title="Status Test Case",
            reference_no=f"REF-{cid}",
            examination_date="2026-11-07",
            examiner="Examiner Status",
            status="Active",
        )

        # GET detail
        response = self.client.get(f"/cases/{cid}")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Status Test Case", response.get_data(as_text=True))

        # POST status update
        up_res = self.client.post(f"/cases/{cid}/status", data={"status": "Completed"}, follow_redirects=True)
        self.assertEqual(up_res.status_code, 200)

        # Verify DB updated
        c_after = database.get_case_by_id(cid)
        self.assertEqual(c_after["status"], "Completed")

    def test_evidence_workspace_listing_and_creation(self):
        """Evidence registry listing and logging flow."""
        # GET list
        response = self.client.get("/evidence")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("Physical Bloodstain Evidence Registry", html)

        # GET new form
        form_page = self.client.get("/evidence/new")
        self.assertEqual(form_page.status_code, 200)

        # POST new evidence
        post_data = {
            "case_id": "BPA-2026-0101",
            "description": "Cluster of 15 high-velocity droplets on south baseboard",
            "evidence_type": "Impact Spatter",
            "location_surface": "Pine wood baseboard",
            "status": "Logged",
            "notes": "Photographed under 5000K daylight LED",
        }
        res_post = self.client.post("/evidence/new", data=post_data, follow_redirects=True)
        self.assertEqual(res_post.status_code, 200)

        # Verify evidence persisted in database
        evs = database.get_evidence_for_case("BPA-2026-0101")
        matching = [e for e in evs if "south baseboard" in e["description"]]
        self.assertTrue(len(matching) >= 1)

    def test_measurement_workspace_layout_and_placeholders(self):
        """Measurement lab should render Phase 3 optical preview placeholder and controls."""
        response = self.client.get("/measurement")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("Optical Inspection &amp; Ellipse Detection", html)
        self.assertIn("Computer Vision Optical Station", html)
        self.assertIn("Pending Spatial Calibration", html)
        self.assertIn("Stain Measurement Controls", html)
        self.assertIn('name="width"', html)
        self.assertIn('name="length"', html)

    def test_measurement_workspace_calculation_and_persistence(self):
        """Measurement workspace calculation should compute angle and save record when requested."""
        post_data = {
            "case_id": "BPA-2026-0101",
            "stain_label": "TestStain-99",
            "width": "5.0",
            "length": "10.0",
            "save_record": "true",
        }
        response = self.client.post("/measurement", data=post_data, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("30.00°", html)
        self.assertIn("0.5000", html)
        self.assertIn("saved to history", html)
        self.assertIn("TestStain-99", html)

    def test_results_dashboard_placeholders(self):
        """Results dashboard should display history and trajectory placeholders."""
        response = self.client.get("/results")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("Forensic Results &amp; Trajectory Analysis Dashboard", html)
        self.assertIn("Verified Bloodstain Measurement History", html)
        self.assertIn("2D Area of Convergence", html)
        self.assertIn("3D Spatial Origin Reconstruction", html)
        self.assertIn("Annotated Computer-Vision Overlay", html)


if __name__ == "__main__":
    unittest.main()
