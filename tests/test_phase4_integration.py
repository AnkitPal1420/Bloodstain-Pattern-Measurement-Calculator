"""
tests/test_phase4_integration.py
================================
Automated Unit and Integration Test Suite for Phase 4.3:
Measurement Integration & Automated Trajectory Reconstruction.

Verifies:
1. Creating a case trajectory reconstruction.
2. Adding existing saved measurements to a reconstruction workspace.
3. Preventing duplicate measurement associations within the same reconstruction.
4. Removing measurements from a reconstruction workspace.
5. Updating directional information (gamma in [0, 360)) and direction source.
6. Updating surface coordinates (surface_x_mm, surface_y_mm) and receiving-surface orientation.
7. Rejecting uncalibrated measurements (pixel units cannot be silently treated as mm).
8. Rejecting missing directionality or perpendicular (90 deg) stains without silent discard.
9. Multiple calibrated stain reconstruction (2D Area of Convergence & 3D Area of Origin).
10. Full SQLite persistence across reconstructions, reconstruction_stains, and reconstruction_results.
11. Reconstruction result retrieval via Python API and JSON/HTML endpoints.
12. Handling invalid case IDs, invalid measurement IDs, and cross-case measurement isolation.
13. Flask route integration for all 6 /reconstruction/<case_id> endpoints and /trajectory integration.
"""

import json
import math
import pytest

import app as flask_app
import database
import trajectory_integration
from calculator import calculate_impact_angle
from trajectory_integration import ReconstructionValidationError


@pytest.fixture
def clean_case_env():
    """Provide a clean database state with a test case and calibrated/uncalibrated measurements."""
    database.init_db()
    database.clear_all_data()

    case_id = "CASE-2026-P43"
    database.create_case(
        case_id=case_id,
        title="Phase 4.3 Integration Verification Case",
        reference_no="REF-P43-001",
        examination_date="2026-11-08",
        examiner="Dr. Forensic Integration",
        description="Synthetic benchmark for end-to-end measurement-to-trajectory integration.",
        status="Active",
    )

    # Create a calibrated evidence image record
    img_cal_id = "IMG-2026-CAL01"
    database.create_evidence_image(
        image_id=img_cal_id,
        case_id=case_id,
        evidence_id=None,
        filename="cal_sample.png",
        filepath="sample_images/sample_45deg_calibrated.png",
        original_filename="cal_sample.png",
        file_size=4096,
        mime_type="image/png",
        width_px=800,
        height_px=600,
        scale_factor=0.1,
        is_calibrated=1,
    )

    # Create an uncalibrated evidence image record
    img_uncal_id = "IMG-2026-UNCAL01"
    database.create_evidence_image(
        image_id=img_uncal_id,
        case_id=case_id,
        evidence_id=None,
        filename="uncal_sample.png",
        filepath="sample_images/sample_30deg_oblique.png",
        original_filename="uncal_sample.png",
        file_size=4096,
        mime_type="image/png",
        width_px=800,
        height_px=600,
        scale_factor=1.0,
        is_calibrated=0,
    )

    # Known true origin O = (100, 150, 300) mm on Horizontal Floor
    # Stain 1 at (400, 150): d=300, Z=300 -> alpha=45 deg (w=7.0710678, l=10.0), gamma=0.0
    # Stain 2 at (100, 450): d=300, Z=300 -> alpha=45 deg (w=7.0710678, l=10.0), gamma=90.0
    # Stain 3 at (300, 350): d=282.8427, Z=300 -> alpha=46.6861 deg (w=7.2760688, l=10.0), gamma=45.0
    c1 = calculate_impact_angle(7.0710678, 10.0)
    m1_id = database.create_measurement(
        case_id=case_id,
        evidence_id=None,
        stain_label="Stain-A",
        width=c1["width"],
        length=c1["length"],
        ratio=c1["ratio"],
        angle_degrees=c1["angle_degrees"],
        angle_radians=c1["angle_radians"],
        is_perpendicular=c1["is_perpendicular"],
        interpretation=c1["interpretation"],
        image_id=img_cal_id,
        width_px=70.71,
        length_px=100.0,
        scale_factor=0.1,
        ellipse_center_x=400.0,
        ellipse_center_y=300.0,
        ellipse_angle=0.0,
        calibration_status="Calibrated (10.00 px/mm)",
        direction_angle=0.0,
        surface_x_mm=400.0,
        surface_y_mm=150.0,
        surface_type="Horizontal Floor",
    )

    c2 = calculate_impact_angle(7.0710678, 10.0)
    m2_id = database.create_measurement(
        case_id=case_id,
        evidence_id=None,
        stain_label="Stain-B",
        width=c2["width"],
        length=c2["length"],
        ratio=c2["ratio"],
        angle_degrees=c2["angle_degrees"],
        angle_radians=c2["angle_radians"],
        is_perpendicular=c2["is_perpendicular"],
        interpretation=c2["interpretation"],
        image_id=img_cal_id,
        width_px=70.71,
        length_px=100.0,
        scale_factor=0.1,
        ellipse_center_x=100.0,
        ellipse_center_y=150.0,
        ellipse_angle=90.0,
        calibration_status="Calibrated (10.00 px/mm)",
        direction_angle=90.0,
        surface_x_mm=100.0,
        surface_y_mm=450.0,
        surface_type="Horizontal Floor",
    )

    c3 = calculate_impact_angle(7.2760688, 10.0)
    m3_id = database.create_measurement(
        case_id=case_id,
        evidence_id=None,
        stain_label="Stain-C",
        width=c3["width"],
        length=c3["length"],
        ratio=c3["ratio"],
        angle_degrees=c3["angle_degrees"],
        angle_radians=c3["angle_radians"],
        is_perpendicular=c3["is_perpendicular"],
        interpretation=c3["interpretation"],
        image_id=img_cal_id,
        width_px=72.76,
        length_px=100.0,
        scale_factor=0.1,
        ellipse_center_x=300.0,
        ellipse_center_y=250.0,
        ellipse_angle=45.0,
        calibration_status="Calibrated (10.00 px/mm)",
        direction_angle=45.0,
        surface_x_mm=300.0,
        surface_y_mm=350.0,
        surface_type="Horizontal Floor",
    )

    # Uncalibrated optical measurement
    m_uncal_id = database.create_measurement(
        case_id=case_id,
        evidence_id=None,
        stain_label="Stain-Uncal",
        width=70.0,
        length=100.0,
        ratio=0.7,
        angle_degrees=44.43,
        angle_radians=math.radians(44.43),
        is_perpendicular=False,
        interpretation="Moderate angle",
        image_id=img_uncal_id,
        width_px=70.0,
        length_px=100.0,
        scale_factor=1.0,
        ellipse_center_x=250.0,
        ellipse_center_y=250.0,
        ellipse_angle=30.0,
        calibration_status="Uncalibrated Optical (Pixel Units)",
        direction_angle=30.0,
        surface_x_mm=250.0,
        surface_y_mm=250.0,
        surface_type="Horizontal Floor",
    )

    # Calibrated measurement without direction_angle yet
    m_nodir_id = database.create_measurement(
        case_id=case_id,
        evidence_id=None,
        stain_label="Stain-NoDir",
        width=5.0,
        length=10.0,
        ratio=0.5,
        angle_degrees=30.0,
        angle_radians=math.radians(30.0),
        is_perpendicular=False,
        interpretation="30 deg",
        image_id=img_cal_id,
        width_px=50.0,
        length_px=100.0,
        scale_factor=0.1,
        calibration_status="Calibrated (10.00 px/mm)",
        direction_angle=None,
        surface_x_mm=350.0,
        surface_y_mm=250.0,
        surface_type="Horizontal Floor",
    )

    yield {
        "case_id": case_id,
        "m1_id": m1_id,
        "m2_id": m2_id,
        "m3_id": m3_id,
        "m_uncal_id": m_uncal_id,
        "m_nodir_id": m_nodir_id,
    }

    database.clear_all_data()


def test_create_and_retrieve_case_reconstruction(clean_case_env):
    """1. Verify creating and retrieving a case trajectory reconstruction."""
    case_id = clean_case_env["case_id"]
    rec = trajectory_integration.get_or_create_case_reconstruction(case_id)
    assert rec is not None
    assert rec["case_id"] == case_id
    assert rec["reconstruction_id"].startswith("REC-")
    assert rec["status"] == "Draft"

    # Calling again returns the existing reconstruction
    rec_again = trajectory_integration.get_or_create_case_reconstruction(case_id)
    assert rec_again["reconstruction_id"] == rec["reconstruction_id"]


def test_add_prevent_duplicate_and_remove_measurements(clean_case_env):
    """2, 3, 4. Verify adding measurements, preventing duplicate associations, and removing measurements."""
    case_id = clean_case_env["case_id"]
    m1_id = clean_case_env["m1_id"]
    m2_id = clean_case_env["m2_id"]

    rec = trajectory_integration.get_or_create_case_reconstruction(case_id)
    rec_id = rec["reconstruction_id"]

    s1 = trajectory_integration.add_measurement_to_case_reconstruction(
        case_id=case_id,
        measurement_id=m1_id,
        reconstruction_id=rec_id,
    )
    assert s1["measurement_id"] == m1_id
    assert s1["stain_identifier"] == "Stain-A"

    # Duplicate association must raise ReconstructionValidationError
    with pytest.raises(ReconstructionValidationError, match="already associated"):
        trajectory_integration.add_measurement_to_case_reconstruction(
            case_id=case_id,
            measurement_id=m1_id,
            reconstruction_id=rec_id,
        )

    # Add second measurement
    s2 = trajectory_integration.add_measurement_to_case_reconstruction(
        case_id=case_id,
        measurement_id=m2_id,
        reconstruction_id=rec_id,
    )
    assert s2["measurement_id"] == m2_id

    stains = database.get_stains_for_reconstruction(rec_id)
    assert len(stains) == 2

    # Remove m2
    removed = trajectory_integration.remove_measurement_from_case_reconstruction(
        case_id=case_id,
        measurement_id=m2_id,
        reconstruction_id=rec_id,
    )
    assert removed is True
    stains_after = database.get_stains_for_reconstruction(rec_id)
    assert len(stains_after) == 1
    assert stains_after[0]["measurement_id"] == m1_id


def test_update_directional_and_surface_coordinates(clean_case_env):
    """5 & 6. Verify updating directional information and surface coordinates with range validation."""
    case_id = clean_case_env["case_id"]
    m_nodir_id = clean_case_env["m_nodir_id"]
    rec = trajectory_integration.get_or_create_case_reconstruction(case_id)
    rec_id = rec["reconstruction_id"]

    added = trajectory_integration.add_measurement_to_case_reconstruction(
        case_id=case_id,
        measurement_id=m_nodir_id,
        reconstruction_id=rec_id,
    )
    assert added["direction_angle"] is None
    assert trajectory_integration.evaluate_stain_eligibility(added)["is_eligible"] is False

    # Invalid direction angle (< 0 or >= 360) must be rejected
    with pytest.raises(ReconstructionValidationError, match="0° <= gamma < 360°"):
        trajectory_integration.update_stain_in_case_reconstruction(
            case_id=case_id,
            measurement_id=m_nodir_id,
            reconstruction_id=rec_id,
            direction_angle=360.0,
        )

    with pytest.raises(ReconstructionValidationError, match="0° <= gamma < 360°"):
        trajectory_integration.update_stain_in_case_reconstruction(
            case_id=case_id,
            measurement_id=m_nodir_id,
            reconstruction_id=rec_id,
            direction_angle=-15.0,
        )

    # Valid update makes stain eligible
    updated = trajectory_integration.update_stain_in_case_reconstruction(
        case_id=case_id,
        measurement_id=m_nodir_id,
        reconstruction_id=rec_id,
        stain_identifier="Stain-Updated",
        surface_x_mm=380.5,
        surface_y_mm=210.25,
        direction_angle=62.5,
        direction_source="Tail / Scalloping Alignment",
    )
    assert updated["stain_identifier"] == "Stain-Updated"
    assert updated["surface_x_mm"] == pytest.approx(380.5)
    assert updated["surface_y_mm"] == pytest.approx(210.25)
    assert updated["direction_angle"] == pytest.approx(62.5)
    assert updated["direction_source"] == "Tail / Scalloping Alignment"
    assert trajectory_integration.evaluate_stain_eligibility(updated)["is_eligible"] is True


def test_reject_uncalibrated_and_missing_directionality(clean_case_env):
    """7 & 8. Verify uncalibrated measurements and missing directionality block calculation without silent discard."""
    case_id = clean_case_env["case_id"]
    rec = trajectory_integration.get_or_create_case_reconstruction(case_id)
    rec_id = rec["reconstruction_id"]

    # Add two valid stains + one uncalibrated stain
    trajectory_integration.add_measurement_to_case_reconstruction(
        case_id=case_id, measurement_id=clean_case_env["m1_id"], reconstruction_id=rec_id
    )
    trajectory_integration.add_measurement_to_case_reconstruction(
        case_id=case_id, measurement_id=clean_case_env["m2_id"], reconstruction_id=rec_id
    )
    trajectory_integration.add_measurement_to_case_reconstruction(
        case_id=case_id, measurement_id=clean_case_env["m_uncal_id"], reconstruction_id=rec_id
    )

    with pytest.raises(ReconstructionValidationError, match="Uncalibrated"):
        trajectory_integration.execute_case_reconstruction(case_id=case_id, reconstruction_id=rec_id)

    # Remove uncalibrated stain, add stain missing direction_angle
    trajectory_integration.remove_measurement_from_case_reconstruction(
        case_id=case_id, measurement_id=clean_case_env["m_uncal_id"], reconstruction_id=rec_id
    )
    trajectory_integration.add_measurement_to_case_reconstruction(
        case_id=case_id, measurement_id=clean_case_env["m_nodir_id"], reconstruction_id=rec_id
    )

    with pytest.raises(ReconstructionValidationError, match="direction of travel"):
        trajectory_integration.execute_case_reconstruction(case_id=case_id, reconstruction_id=rec_id)


def test_multiple_stain_reconstruction_and_persistence(clean_case_env):
    """9, 10, 11. Verify full 3-stain trajectory reconstruction, SQLite persistence, and result retrieval."""
    case_id = clean_case_env["case_id"]
    rec = trajectory_integration.get_or_create_case_reconstruction(case_id)
    rec_id = rec["reconstruction_id"]

    trajectory_integration.add_measurement_to_case_reconstruction(
        case_id=case_id, measurement_id=clean_case_env["m1_id"], reconstruction_id=rec_id
    )
    trajectory_integration.add_measurement_to_case_reconstruction(
        case_id=case_id, measurement_id=clean_case_env["m2_id"], reconstruction_id=rec_id
    )
    trajectory_integration.add_measurement_to_case_reconstruction(
        case_id=case_id, measurement_id=clean_case_env["m3_id"], reconstruction_id=rec_id
    )

    output = trajectory_integration.execute_case_reconstruction(
        case_id=case_id,
        reconstruction_id=rec_id,
        surface_type="Horizontal Floor",
    )

    assert output["reconstruction"]["status"] == "Calculated"
    res = output["result"]
    assert res["stain_count"] == 3
    assert res["convergence_x_mm"] == pytest.approx(100.0, abs=0.5)
    assert res["convergence_y_mm"] == pytest.approx(150.0, abs=0.5)
    assert res["estimated_origin_x_mm"] == pytest.approx(100.0, abs=0.5)
    assert res["estimated_origin_y_mm"] == pytest.approx(150.0, abs=0.5)
    assert res["estimated_origin_z_mm"] == pytest.approx(300.0, abs=0.5)

    # Retrieve persisted result from SQLite
    persisted = database.get_latest_reconstruction_result(rec_id)
    assert persisted is not None
    assert persisted["estimated_origin_z_mm"] == pytest.approx(300.0, abs=0.5)
    parsed_scene = json.loads(persisted["result_json"])
    assert "stains" in parsed_scene
    assert len(parsed_scene["stains"]) == 3


def test_invalid_case_and_measurement_ids(clean_case_env):
    """12. Verify invalid case IDs, invalid measurement IDs, and cross-case measurement isolation."""
    case_id = clean_case_env["case_id"]
    rec = trajectory_integration.get_or_create_case_reconstruction(case_id)
    rec_id = rec["reconstruction_id"]

    with pytest.raises(ReconstructionValidationError, match="not found"):
        trajectory_integration.get_or_create_case_reconstruction("NON-EXISTENT-CASE")

    with pytest.raises(ReconstructionValidationError, match="does not exist"):
        trajectory_integration.add_measurement_to_case_reconstruction(
            case_id=case_id, measurement_id=999999, reconstruction_id=rec_id
        )

    # Create another case and verify cross-case measurement rejection
    other_case = "CASE-2026-OTHER"
    database.create_case(
        case_id=other_case,
        title="Other Case",
        reference_no="REF-OTHER",
        examination_date="2026-11-08",
        examiner="Examiner B",
        description="Secondary test case",
    )
    other_rec = trajectory_integration.get_or_create_case_reconstruction(other_case)
    with pytest.raises(ReconstructionValidationError, match="belongs to Case"):
        trajectory_integration.add_measurement_to_case_reconstruction(
            case_id=other_case,
            measurement_id=clean_case_env["m1_id"],
            reconstruction_id=other_rec["reconstruction_id"],
        )


def test_flask_reconstruction_routes_integration(clean_case_env):
    """13. Verify all 6 Flask /reconstruction/<case_id> endpoints and /trajectory integration."""
    case_id = clean_case_env["case_id"]
    m1_id = clean_case_env["m1_id"]
    m2_id = clean_case_env["m2_id"]
    m3_id = clean_case_env["m3_id"]

    client = flask_app.app.test_client()

    # 1. GET /reconstruction/<case_id>
    resp_get = client.get(f"/reconstruction/{case_id}")
    assert resp_get.status_code == 200
    assert b"Automated Multi-Stain Trajectory Reconstruction" in resp_get.data

    # 2. POST /reconstruction/<case_id>/add-measurement (JSON)
    for mid in (m1_id, m2_id, m3_id):
        resp_add = client.post(
            f"/reconstruction/{case_id}/add-measurement",
            data={"measurement_id": str(mid)},
            headers={"Accept": "application/json"},
        )
        assert resp_add.status_code == 201
        payload = resp_add.get_json()
        assert payload["status"] == "ok"

    # 3. POST /reconstruction/<case_id>/update-stain (JSON)
    resp_upd = client.post(
        f"/reconstruction/{case_id}/update-stain",
        data={
            "measurement_id": str(m1_id),
            "stain_identifier": "Stain-A-Verified",
            "surface_x_mm": "400.0",
            "surface_y_mm": "150.0",
            "direction_angle": "0.0",
            "direction_source": "Examiner Visual Inspection",
        },
        headers={"Accept": "application/json"},
    )
    assert resp_upd.status_code == 200
    assert resp_upd.get_json()["stain"]["stain_identifier"] == "Stain-A-Verified"

    # 4. POST /reconstruction/<case_id>/calculate (JSON)
    resp_calc = client.post(
        f"/reconstruction/{case_id}/calculate",
        data={"surface_type": "Horizontal Floor", "notes": "Flask route integration test"},
        headers={"Accept": "application/json"},
    )
    assert resp_calc.status_code == 200
    calc_json = resp_calc.get_json()
    assert calc_json["status"] == "ok"
    assert calc_json["result"]["estimated_origin_z_mm"] == pytest.approx(300.0, abs=0.5)
    rec_id = calc_json["reconstruction"]["reconstruction_id"]

    # 5. GET /reconstruction/<case_id>/results (JSON)
    resp_res = client.get(
        f"/reconstruction/{case_id}/results?reconstruction_id={rec_id}",
        headers={"Accept": "application/json"},
    )
    assert resp_res.status_code == 200
    res_json = resp_res.get_json()
    assert res_json["latest_result"]["stain_count"] == 3

    # 6. GET /trajectory?reconstruction_id=<rec_id> loads the workspace stains into 3D viewport
    resp_traj = client.get(f"/trajectory?reconstruction_id={rec_id}")
    assert resp_traj.status_code == 200
    assert b"Stain-A-Verified" in resp_traj.data

    # 7. POST /reconstruction/<case_id>/remove-measurement (JSON)
    resp_rem = client.post(
        f"/reconstruction/{case_id}/remove-measurement",
        data={"measurement_id": str(m3_id)},
        headers={"Accept": "application/json"},
    )
    assert resp_rem.status_code == 200
    assert resp_rem.get_json()["removed_measurement_id"] == m3_id
