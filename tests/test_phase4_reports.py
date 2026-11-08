"""
tests/test_phase4_reports.py
============================
Automated Unit and Integration Test Suite for Phase 4.4:
Professional Forensic PDF Report Generation, Cryptographic Traceability & Revision Control.

Covers all 18 required test scenarios:
1. Individual measurement report generation (Type A).
2. Reconstruction report generation (Type B).
3. Complete case report generation (Type C) & strict isolation from unrelated cases.
4. PDF file structural validity (%PDF- header, %%EOF trailer, page objects, uncompressed text).
5. Correct report metadata (report_id, case_id, examiner, laboratory, observations).
6. Correct source measurement references (source_measurement_ids).
7. Original image inclusion & SHA-256 preservation.
8. Annotated computer-vision overlay image inclusion.
9. Graceful missing image file handling.
10. Invalid case ID handling.
11. Invalid reconstruction ID handling.
12. PDF SHA-256 hash verification against disk file.
13. Non-destructive report revision handling (Rev R1 -> R2 -> R3, never overwriting FINAL).
14. Draft and Final status enforcement and invalid status rejection.
15. Path traversal protection on /reports/<report_id> and /reports/<report_id>/download.
16. SQLite database persistence and filtering.
17. Flask generation, inspection, and download route integration.
18. Module independence of report_generator.py from Flask.
"""

import ast
import hashlib
import json
import os
import shutil
import pytest

import app as flask_app
import database
import image_processing
import report_generator
import trajectory_integration
from calculator import calculate_impact_angle
from report_generator import ReportGenerationError


@pytest.fixture
def report_test_env(tmp_path):
    """
    Create a deterministic synthetic forensic case environment with:
    - Primary Case (CASE-2026-RPT1) with 3 calibrated stains, 1 optical image, and a calculated reconstruction.
    - Unrelated Secondary Case (CASE-2026-UNRELATED) to verify strict case isolation in reports.
    - Dedicated temporary directory for generated PDF reports.
    """
    database.init_db()
    database.clear_all_data()

    reports_dir = str(tmp_path / "reports_out")
    os.makedirs(reports_dir, exist_ok=True)

    # 1. Primary Case
    case_id = "CASE-2026-RPT1"
    database.create_case(
        case_id=case_id,
        title="Forensic PDF Report Verification Case",
        reference_no="REF-RPT-2026-01",
        examination_date="2026-11-08",
        examiner="Examiner Jane Doe",
        description="Controlled synthetic case for Phase 4.4 PDF report verification.",
        status="Active",
    )

    ev_id = "EVD-RPT1-01"
    database.create_evidence(
        evidence_id=ev_id,
        case_id=case_id,
        description="Synthetic bloodstain spatter panel A",
        evidence_type="Impact Spatter",
        location_surface="Horizontal Floor Panel",
        status="Analyzed",
        notes="Controlled synthetic drop pattern",
    )

    # Copy sample image into tmp_path so original image hashing and overlay generation work cleanly
    sample_src = os.path.join(report_generator.BASE_DIR, "sample_images", "sample_45deg_calibrated.png")
    orig_copy = str(tmp_path / "IMG-2026-RPT01_sample_45deg.png")
    shutil.copyfile(sample_src, orig_copy)

    img_id = "IMG-2026-RPT01"
    database.create_evidence_image(
        image_id=img_id,
        case_id=case_id,
        evidence_id=ev_id,
        filename=os.path.basename(orig_copy),
        filepath=orig_copy,
        original_filename="sample_45deg_calibrated.png",
        file_size=os.path.getsize(orig_copy),
        mime_type="image/png",
        width_px=500,
        height_px=500,
        scale_factor=0.25,
        is_calibrated=1,
    )

    # Create 3 calibrated stains converging at (100, 150, 300) mm
    c1 = calculate_impact_angle(7.0710678, 10.0)
    m1_id = database.create_measurement(
        case_id=case_id,
        evidence_id=ev_id,
        stain_label="Stain-RPT-01",
        width=c1["width"],
        length=c1["length"],
        ratio=c1["ratio"],
        angle_degrees=c1["angle_degrees"],
        angle_radians=c1["angle_radians"],
        is_perpendicular=c1["is_perpendicular"],
        interpretation=c1["interpretation"],
        image_id=img_id,
        width_px=70.71,
        length_px=100.0,
        scale_factor=0.25,
        ellipse_center_x=250.0,
        ellipse_center_y=200.0,
        ellipse_angle=159.9,
        calibration_status="Calibrated (4.00 px/mm)",
        direction_angle=0.0,
        surface_x_mm=400.0,
        surface_y_mm=150.0,
        surface_type="Horizontal Floor",
    )

    c2 = calculate_impact_angle(7.0710678, 10.0)
    m2_id = database.create_measurement(
        case_id=case_id,
        evidence_id=ev_id,
        stain_label="Stain-RPT-02",
        width=c2["width"],
        length=c2["length"],
        ratio=c2["ratio"],
        angle_degrees=c2["angle_degrees"],
        angle_radians=c2["angle_radians"],
        is_perpendicular=c2["is_perpendicular"],
        interpretation=c2["interpretation"],
        image_id=img_id,
        width_px=70.71,
        length_px=100.0,
        scale_factor=0.25,
        ellipse_center_x=100.0,
        ellipse_center_y=150.0,
        ellipse_angle=90.0,
        calibration_status="Calibrated (4.00 px/mm)",
        direction_angle=90.0,
        surface_x_mm=100.0,
        surface_y_mm=450.0,
        surface_type="Horizontal Floor",
    )

    c3 = calculate_impact_angle(7.2760688, 10.0)
    m3_id = database.create_measurement(
        case_id=case_id,
        evidence_id=ev_id,
        stain_label="Stain-RPT-03",
        width=c3["width"],
        length=c3["length"],
        ratio=c3["ratio"],
        angle_degrees=c3["angle_degrees"],
        angle_radians=c3["angle_radians"],
        is_perpendicular=c3["is_perpendicular"],
        interpretation=c3["interpretation"],
        image_id=img_id,
        width_px=72.76,
        length_px=100.0,
        scale_factor=0.25,
        ellipse_center_x=300.0,
        ellipse_center_y=250.0,
        ellipse_angle=45.0,
        calibration_status="Calibrated (4.00 px/mm)",
        direction_angle=45.0,
        surface_x_mm=300.0,
        surface_y_mm=350.0,
        surface_type="Horizontal Floor",
    )

    rec = trajectory_integration.get_or_create_case_reconstruction(case_id)
    rec_id = rec["reconstruction_id"]
    for mid in (m1_id, m2_id, m3_id):
        trajectory_integration.add_measurement_to_case_reconstruction(
            case_id=case_id, measurement_id=mid, reconstruction_id=rec_id
        )
    trajectory_integration.execute_case_reconstruction(case_id=case_id, reconstruction_id=rec_id)

    # 2. Unrelated Secondary Case (must never appear in Primary Case reports)
    unrelated_case_id = "CASE-2026-UNRELATED"
    database.create_case(
        case_id=unrelated_case_id,
        title="Unrelated Secret Case",
        reference_no="REF-SECRET-999",
        examination_date="2026-11-08",
        examiner="Other Examiner",
        description="Should never leak into CASE-2026-RPT1 report.",
        status="Active",
    )
    database.create_measurement(
        case_id=unrelated_case_id,
        evidence_id=None,
        stain_label="UNRELATED-STAIN-999",
        width=5.0,
        length=10.0,
        ratio=0.5,
        angle_degrees=30.0,
        angle_radians=0.5236,
        is_perpendicular=False,
        interpretation="Unrelated",
    )

    yield {
        "reports_dir": reports_dir,
        "case_id": case_id,
        "ev_id": ev_id,
        "img_id": img_id,
        "orig_copy": orig_copy,
        "m1_id": m1_id,
        "m2_id": m2_id,
        "m3_id": m3_id,
        "rec_id": rec_id,
        "unrelated_case_id": unrelated_case_id,
    }

    database.clear_all_data()


def test_report_generator_independent_of_flask():
    """18. Verify report_generator.py does not import Flask."""
    src_path = os.path.join(report_generator.BASE_DIR, "report_generator.py")
    with open(src_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())

    imported_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module.split(".")[0])

    assert "flask" not in imported_modules


def test_individual_measurement_report_generation_and_images(report_test_env):
    """1, 4, 5, 6, 7, 8, 12. Verify Individual Bloodstain Examination Report (Type A) generation, images, and SHA-256."""
    m1_id = report_test_env["m1_id"]
    reports_dir = report_test_env["reports_dir"]
    orig_copy = report_test_env["orig_copy"]
    orig_sha_before = report_generator.compute_file_sha256(orig_copy)

    record = report_generator.generate_measurement_report(
        measurement_id=m1_id,
        output_dir=reports_dir,
        report_status="DRAFT",
        examiner_name="Analyst K. Sharma",
        laboratory_name="Central BPA Laboratory",
        examiner_observations="Well-defined elliptical contour with clear directional spine.",
    )

    assert record is not None
    assert record["report_type"] == "Individual Measurement"
    assert record["report_status"] == "DRAFT"
    assert record["revision_number"] == 1
    assert record["case_id"] == report_test_env["case_id"]
    assert record["measurement_id"] == m1_id
    assert json.loads(record["source_measurement_ids"]) == [m1_id]
    assert json.loads(record["source_image_ids"]) == [report_test_env["img_id"]]

    # Verify original image was preserved untouched and its SHA-256 was recorded
    orig_sha_after = report_generator.compute_file_sha256(orig_copy)
    assert orig_sha_before == orig_sha_after
    img_hashes = json.loads(record["image_hashes_json"])
    assert img_hashes[report_test_env["img_id"]] == orig_sha_before

    # Verify PDF validity, embedded images (/Subtype /Image), and SHA-256 match
    pdf_path = record["filepath"]
    assert os.path.isfile(pdf_path)
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    assert pdf_bytes.startswith(b"%PDF-1.")
    assert b"%%EOF" in pdf_bytes[-1024:]
    assert hashlib.sha256(pdf_bytes).hexdigest().lower() == record["sha256_hash"]

    # Check uncompressed PDF stream contains required text and embedded XObject images
    assert b"Stain-RPT-01" in pdf_bytes
    assert b"Analyst K. Sharma" in pdf_bytes
    assert b"Central BPA Laboratory" in pdf_bytes
    assert b"Unmodified Original Evidence Photograph" in pdf_bytes
    assert b"Processed Computer-Vision Annotated Overlay" in pdf_bytes
    assert b"/Subtype /Image" in pdf_bytes


def test_reconstruction_report_generation(report_test_env):
    """2, 4, 5, 6. Verify Trajectory Reconstruction Report (Type B) generation and 2D/3D convergence content."""
    rec_id = report_test_env["rec_id"]
    reports_dir = report_test_env["reports_dir"]

    record = report_generator.generate_reconstruction_report(
        reconstruction_id=rec_id,
        output_dir=reports_dir,
        report_status="FINAL",
        examiner_name="Senior Examiner R. Verma",
        laboratory_name="Regional Forensic Science Lab",
        examiner_observations="Three-stain radial convergence verified on horizontal floor.",
    )

    assert record["report_type"] == "Trajectory Reconstruction"
    assert record["report_status"] == "FINAL"
    assert record["reconstruction_id"] == rec_id
    assert len(json.loads(record["source_measurement_ids"])) == 3

    with open(record["filepath"], "rb") as f:
        pdf_bytes = f.read()

    assert pdf_bytes.startswith(b"%PDF-1.")
    assert b"SECTION 7" in pdf_bytes
    assert b"SECTION 8" in pdf_bytes
    assert b"SECTION 9" in pdf_bytes
    assert b"100.00 mm" in pdf_bytes
    assert b"150.00 mm" in pdf_bytes
    assert b"300.00" in pdf_bytes
    assert b"UNRELATED-STAIN-999" not in pdf_bytes


def test_complete_case_report_and_unrelated_case_isolation(report_test_env):
    """3, 6. Verify Complete Forensic Case Report (Type C) includes all case records and excludes unrelated cases."""
    case_id = report_test_env["case_id"]
    reports_dir = report_test_env["reports_dir"]

    record = report_generator.generate_case_report(
        case_id=case_id,
        output_dir=reports_dir,
        report_status="FINAL",
    )

    assert record["report_type"] == "Complete Case Report"
    assert record["case_id"] == case_id
    assert record["reconstruction_id"] == report_test_env["rec_id"]

    with open(record["filepath"], "rb") as f:
        pdf_bytes = f.read()

    assert b"CASE-2026-RPT1" in pdf_bytes
    assert b"EVD-RPT1-01" in pdf_bytes
    assert b"Stain-RPT-01" in pdf_bytes
    assert b"Stain-RPT-02" in pdf_bytes
    assert b"Stain-RPT-03" in pdf_bytes
    # Strictly verify unrelated case is excluded
    assert b"CASE-2026-UNRELATED" not in pdf_bytes
    assert b"UNRELATED-STAIN-999" not in pdf_bytes
    assert b"REF-SECRET-999" not in pdf_bytes


def test_missing_image_file_handled_gracefully(report_test_env):
    """9. Verify missing image file on disk is handled gracefully without crashing report generation."""
    img_ghost_id = "IMG-2026-GHOST"
    database.create_evidence_image(
        image_id=img_ghost_id,
        case_id=report_test_env["case_id"],
        evidence_id=report_test_env["ev_id"],
        filename="nonexistent_file.png",
        filepath="/nonexistent/path/nonexistent_file.png",
        original_filename="nonexistent_file.png",
        file_size=1024,
        mime_type="image/png",
        width_px=400,
        height_px=400,
        scale_factor=0.2,
        is_calibrated=1,
    )
    m_ghost_id = database.create_measurement(
        case_id=report_test_env["case_id"],
        evidence_id=report_test_env["ev_id"],
        stain_label="Stain-Ghost-Img",
        width=5.0,
        length=10.0,
        ratio=0.5,
        angle_degrees=30.0,
        angle_radians=0.5236,
        is_perpendicular=False,
        interpretation="30 deg",
        image_id=img_ghost_id,
        calibration_status="Calibrated (5.00 px/mm)",
    )

    rec = report_generator.generate_measurement_report(
        measurement_id=m_ghost_id,
        output_dir=report_test_env["reports_dir"],
        report_status="DRAFT",
    )
    assert os.path.isfile(rec["filepath"])
    img_hashes = json.loads(rec["image_hashes_json"])
    assert img_hashes[img_ghost_id] == "MISSING_ON_DISK"


def test_invalid_ids_and_status_validation(report_test_env):
    """10, 11, 14. Verify invalid case_id, reconstruction_id, measurement_id, and report_status are rejected."""
    reports_dir = report_test_env["reports_dir"]

    with pytest.raises(ReportGenerationError, match="not found"):
        report_generator.generate_case_report("NON-EXISTENT-CASE", output_dir=reports_dir)

    with pytest.raises(ReportGenerationError, match="not found"):
        report_generator.generate_reconstruction_report("REC-NON-EXISTENT", output_dir=reports_dir)

    with pytest.raises(ReportGenerationError, match="not found"):
        report_generator.generate_measurement_report(999999, output_dir=reports_dir)

    with pytest.raises(ReportGenerationError, match="Invalid report status"):
        report_generator.generate_case_report(
            report_test_env["case_id"],
            output_dir=reports_dir,
            report_status="INVALID_STATUS",
        )


def test_report_revision_control_and_non_overwrite(report_test_env):
    """13, 14, 16. Verify regenerating reports increments revision numbers (R1 -> R2 -> R3) and preserves prior FINAL files."""
    case_id = report_test_env["case_id"]
    reports_dir = report_test_env["reports_dir"]

    r1 = report_generator.generate_case_report(case_id, output_dir=reports_dir, report_status="DRAFT")
    r2 = report_generator.generate_case_report(case_id, output_dir=reports_dir, report_status="FINAL")
    r3 = report_generator.generate_case_report(case_id, output_dir=reports_dir, report_status="FINAL")

    assert r1["revision_number"] == 1
    assert r2["revision_number"] == 2
    assert r3["revision_number"] == 3

    assert r1["report_id"].endswith("-R1")
    assert r2["report_id"].endswith("-R2")
    assert r3["report_id"].endswith("-R3")

    # All three files exist independently on disk and in SQLite
    assert os.path.isfile(r1["filepath"])
    assert os.path.isfile(r2["filepath"])
    assert os.path.isfile(r3["filepath"])
    assert len({r1["filepath"], r2["filepath"], r3["filepath"]}) == 3

    all_case_reports = database.get_all_reports(case_id=case_id)
    assert len(all_case_reports) == 3


def test_flask_report_routes_download_and_path_traversal(report_test_env, monkeypatch):
    """15, 17. Verify Flask report generation, metadata view, PDF download, and path traversal protection."""
    monkeypatch.setattr(flask_app, "UPLOAD_REPORTS_DIR", report_test_env["reports_dir"])
    client = flask_app.app.test_client()

    # 1. POST /reports/generate/measurement/<id>
    resp_m = client.post(
        f"/reports/generate/measurement/{report_test_env['m1_id']}",
        data={"report_status": "DRAFT", "examiner_name": "Examiner Route Test"},
        headers={"Accept": "application/json"},
    )
    assert resp_m.status_code == 201
    rpt_m = resp_m.get_json()["report"]
    assert rpt_m["report_type"] == "Individual Measurement"

    # 2. POST /reports/generate/reconstruction/<id>
    resp_r = client.post(
        f"/reports/generate/reconstruction/{report_test_env['rec_id']}",
        data={"report_status": "FINAL"},
        headers={"Accept": "application/json"},
    )
    assert resp_r.status_code == 201
    rpt_r = resp_r.get_json()["report"]
    assert rpt_r["report_type"] == "Trajectory Reconstruction"

    # 3. POST /reports/generate/case/<id>
    resp_c = client.post(
        f"/reports/generate/case/{report_test_env['case_id']}",
        data={"report_status": "FINAL"},
        headers={"Accept": "application/json"},
    )
    assert resp_c.status_code == 201
    rpt_c = resp_c.get_json()["report"]
    assert rpt_c["report_type"] == "Complete Case Report"

    # 4. GET /reports and GET /reports/<report_id>
    resp_list = client.get("/reports")
    assert resp_list.status_code == 200
    assert rpt_c["report_id"].encode() in resp_list.data

    resp_detail = client.get(f"/reports/{rpt_c['report_id']}", headers={"Accept": "application/json"})
    assert resp_detail.status_code == 200
    assert resp_detail.get_json()["report"]["sha256_hash"] == rpt_c["sha256_hash"]

    # 5. GET /reports/<report_id>/download
    resp_dl = client.get(f"/reports/{rpt_c['report_id']}/download")
    assert resp_dl.status_code == 200
    assert resp_dl.mimetype == "application/pdf"
    assert resp_dl.data.startswith(b"%PDF-1.")
    assert hashlib.sha256(resp_dl.data).hexdigest().lower() == rpt_c["sha256_hash"]

    # 6. Path traversal protection
    for malicious_id in ("../app.py", "..\\app.py", "../../etc/passwd", "RPT-2026-0001/../secret"):
        resp_trav = client.get(f"/reports/{malicious_id}/download")
        assert resp_trav.status_code == 404

