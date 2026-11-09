"""
app.py
======
Forensic Laboratory Web Application for Bloodstain Pattern Analysis (BPA).

Provides interactive case management, physical evidence cataloging,
and droplet impact angle calculations using elliptical geometry.
"""

import json
import os
import shutil
from datetime import datetime

from flask import (
    Flask,
    abort,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from werkzeug.utils import secure_filename

import database
import image_processing
import report_generator
import trajectory
import trajectory_integration
import trajectory_visualization
from calculator import InvalidMeasurementError, calculate_impact_angle
from image_processing import ImageProcessingError
from report_generator import ReportGenerationError
from trajectory import StainTrajectoryInput, TrajectoryError
from trajectory_integration import ReconstructionValidationError

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "bpa-forensic-lab-secret-2026")
app.config["MAX_CONTENT_LENGTH"] = image_processing.MAX_FILE_SIZE_BYTES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_ORIGINAL_DIR = os.path.join(BASE_DIR, "uploads", "original")
UPLOAD_PROCESSED_DIR = os.path.join(BASE_DIR, "uploads", "processed")
UPLOAD_REPORTS_DIR = os.path.join(BASE_DIR, "uploads", "reports")
SAMPLE_IMAGES_DIR = os.path.join(BASE_DIR, "sample_images")

os.makedirs(UPLOAD_ORIGINAL_DIR, exist_ok=True)
os.makedirs(UPLOAD_PROCESSED_DIR, exist_ok=True)
os.makedirs(UPLOAD_REPORTS_DIR, exist_ok=True)

# Initialize SQLite database schema and baseline data on load
database.init_db()


@app.errorhandler(413)
def handle_file_too_large(err):
    """Handle uploads exceeding MAX_CONTENT_LENGTH (16 MB)."""
    msg = "Uploaded image exceeds the maximum permitted limit of 16 MB."
    if request.headers.get("Accept") == "application/json":
        return jsonify({"status": "error", "error": msg}), 413
    flash(msg, "danger")
    return redirect(url_for("measurement_workspace"))


@app.context_processor
def inject_navigation_data():
    """Inject active case/evidence counts for the sidebar navigation badges."""
    try:
        stats = database.get_case_statistics()
    except Exception:
        stats = {"total_cases": 0, "total_evidence": 0}
    return {"nav_stats": stats}


# ===========================================================================
# Main Dashboard (Subphase 2.2)
# ===========================================================================


@app.route("/", methods=["GET", "POST"])
def index():
    """
    Main forensic laboratory dashboard.

    Displays actual database case metrics, recent investigative activity,
    navigation shortcuts, and the quick trigonometric measurement engine.
    """
    result = None
    error = None
    width = 5.0
    length = 10.0

    if request.method == "POST":
        raw_width = request.form.get("width", "").strip()
        raw_length = request.form.get("length", "").strip()

        try:
            width = float(raw_width) if raw_width else None
            length = float(raw_length) if raw_length else None
            result = calculate_impact_angle(width, length)
        except InvalidMeasurementError as err:
            error = str(err)
        except (ValueError, TypeError) as err:
            error = f"Invalid numeric input provided: {err}"
        except Exception as exc:
            error = f"An unexpected calculation error occurred: {exc}"

    stats = database.get_case_statistics()
    recent_cases = database.get_recent_cases(limit=5)

    return render_template(
        "index.html",
        active_page="dashboard",
        stats=stats,
        recent_cases=recent_cases,
        result=result,
        error=error,
        width=width,
        length=length,
    )


# ===========================================================================
# Case Management (Subphase 2.3)
# ===========================================================================


@app.route("/cases", methods=["GET"])
def cases_list():
    """List forensic cases with search and status filtering."""
    search_query = request.args.get("search", "").strip()
    status_filter = request.args.get("status", "All").strip()

    cases = database.get_all_cases(
        search_query=search_query if search_query else None,
        status_filter=status_filter if status_filter else None,
    )

    return render_template(
        "cases.html",
        active_page="cases",
        cases=cases,
        search_query=search_query,
        status_filter=status_filter,
    )


@app.route("/cases/new", methods=["GET", "POST"])
def new_case():
    """Create a new forensic case with an auto-generated unique Case ID."""
    if request.method == "POST":
        case_id = request.form.get("case_id", "").strip()
        title = request.form.get("title", "").strip()
        reference_no = request.form.get("reference_no", "").strip()
        examination_date = request.form.get("examination_date", "").strip()
        examiner = request.form.get("examiner", "").strip()
        description = request.form.get("description", "").strip()
        status = request.form.get("status", "Active").strip()

        # Validation
        if not title or not reference_no or not examination_date or not examiner:
            flash("Please fill in all mandatory case fields.", "danger")
            return redirect(url_for("new_case"))

        # Fallback if case_id was not provided
        if not case_id:
            case_id = database.generate_case_id()

        try:
            database.create_case(
                case_id=case_id,
                title=title,
                reference_no=reference_no,
                examination_date=examination_date,
                examiner=examiner,
                description=description,
                status=status,
            )
            flash(f"Forensic Case {case_id} successfully created and registered.", "success")
            return redirect(url_for("case_detail", case_id=case_id))
        except Exception as exc:
            flash(f"Failed to create case: {exc}", "danger")
            return redirect(url_for("new_case"))

    generated_id = database.generate_case_id()
    current_date = datetime.now().strftime("%Y-%m-%d")

    return render_template(
        "case_form.html",
        active_page="cases",
        generated_id=generated_id,
        current_date=current_date,
    )


@app.route("/cases/<case_id>", methods=["GET"])
def case_detail(case_id: str):
    """View details and evidence associated with a specific case."""
    case = database.get_case_by_id(case_id)
    if not case:
        flash(f"Case '{case_id}' was not found in the laboratory registry.", "danger")
        return redirect(url_for("cases_list"))

    evidence_items = database.get_evidence_for_case(case_id)

    return render_template(
        "case_detail.html",
        active_page="cases",
        case=case,
        evidence_items=evidence_items,
    )


@app.route("/cases/<case_id>/status", methods=["POST"])
def update_case_status_route(case_id: str):
    """Update case status."""
    new_status = request.form.get("status", "").strip()
    if new_status:
        database.update_case_status(case_id, new_status)
        flash(f"Status for case {case_id} updated to '{new_status}'.", "success")
    return redirect(url_for("case_detail", case_id=case_id))


# ===========================================================================
# Evidence Workspace Interface (Subphase 2.4)
# ===========================================================================


@app.route("/evidence", methods=["GET"])
def evidence_list():
    """List physical bloodstain evidence records with optional filtering."""
    case_filter = request.args.get("case_id", "").strip()
    status_filter = request.args.get("status", "All").strip()
    search_query = request.args.get("search", "").strip()

    evidence_items = database.get_all_evidence(
        case_id=case_filter if case_filter else None,
        status_filter=status_filter if status_filter else None,
        search_query=search_query if search_query else None,
    )
    all_cases = database.get_all_cases()

    return render_template(
        "evidence.html",
        active_page="evidence",
        evidence_items=evidence_items,
        all_cases=all_cases,
        selected_case=case_filter,
        status_filter=status_filter,
        search_query=search_query,
    )


@app.route("/evidence/new", methods=["GET", "POST"])
def new_evidence():
    """Log a new physical evidence item attached to a case."""
    all_cases = database.get_all_cases()
    if not all_cases:
        flash("You must create at least one investigation case before logging evidence.", "warning")
        return redirect(url_for("new_case"))

    if request.method == "POST":
        case_id = request.form.get("case_id", "").strip()
        description = request.form.get("description", "").strip()
        evidence_type = request.form.get("evidence_type", "").strip()
        location_surface = request.form.get("location_surface", "").strip()
        status = request.form.get("status", "Logged").strip()
        notes = request.form.get("notes", "").strip()

        if not case_id or not description or not evidence_type:
            flash("Please specify the case, description, and evidence pattern type.", "danger")
            return redirect(url_for("new_evidence", case_id=case_id))

        evidence_id = database.generate_evidence_id(case_id)

        try:
            database.create_evidence(
                evidence_id=evidence_id,
                case_id=case_id,
                description=description,
                evidence_type=evidence_type,
                location_surface=location_surface,
                status=status,
                notes=notes,
            )
            flash(f"Evidence item {evidence_id} successfully logged under Case {case_id}.", "success")
            return redirect(url_for("case_detail", case_id=case_id))
        except Exception as exc:
            flash(f"Failed to log evidence: {exc}", "danger")
            return redirect(url_for("new_evidence"))

    selected_case = request.args.get("case_id", "")
    return render_template(
        "evidence_form.html",
        active_page="evidence",
        cases=all_cases,
        selected_case=selected_case,
    )


# ===========================================================================
# Phase 3: Secure Image Upload & Serving Routes (Subphase 3.1)
# ===========================================================================


@app.route("/uploads/original/<path:filename>", methods=["GET"])
def serve_original_upload(filename: str):
    """Serve original uploaded evidence image securely without path traversal."""
    safe_name = secure_filename(filename)
    if not safe_name or safe_name != filename:
        abort(404)
    return send_from_directory(UPLOAD_ORIGINAL_DIR, safe_name)


@app.route("/uploads/processed/<path:filename>", methods=["GET"])
def serve_processed_upload(filename: str):
    """Serve processed computer vision stage images securely."""
    safe_name = secure_filename(filename)
    if not safe_name or safe_name != filename:
        abort(404)
    return send_from_directory(UPLOAD_PROCESSED_DIR, safe_name)


@app.route("/measurement/upload", methods=["POST"])
@app.route("/evidence/upload", methods=["POST"])
def upload_evidence_image():
    """
    Acquire and validate a photographic bloodstain image (or load a reference sample),
    preserve the original inside uploads/original/, and register metadata in SQLite.
    """
    case_id = request.form.get("case_id", "").strip()
    evidence_id = request.form.get("evidence_id", "").strip()
    sample_name = request.form.get("sample_name", "").strip()

    try:
        if sample_name:
            safe_sample = secure_filename(sample_name)
            sample_path = os.path.join(SAMPLE_IMAGES_DIR, safe_sample)
            if not os.path.exists(sample_path) or not image_processing.is_allowed_file(safe_sample):
                raise ImageProcessingError(f"Requested laboratory sample '{sample_name}' is invalid or missing.")

            with open(sample_path, "rb") as f:
                raw_bytes = f.read()
            orig_filename = safe_sample
            mime_type = "image/png"
        else:
            if "image_file" not in request.files:
                raise ImageProcessingError("No image file was provided in the upload request.")

            file_obj = request.files["image_file"]
            if not file_obj or not file_obj.filename:
                raise ImageProcessingError("Please select a valid image file (.jpg, .jpeg, .png) before uploading.")

            orig_filename = secure_filename(file_obj.filename)
            if not orig_filename or not image_processing.is_allowed_file(orig_filename):
                raise ImageProcessingError(
                    "Unsupported file extension. Only .JPG, .JPEG, and .PNG forensic evidence formats are permitted."
                )

            raw_bytes = file_obj.read()
            mime_type = file_obj.mimetype or "image/png"

        # Validate pixel content & dimensions using OpenCV
        decoded_img = image_processing.load_and_validate_image(raw_bytes)
        height_px, width_px = decoded_img.shape[:2]

        # Generate unique image identifier and store original file
        image_id = database.generate_image_id()
        stored_filename = f"{image_id}_{orig_filename}"
        stored_filepath = os.path.join(UPLOAD_ORIGINAL_DIR, stored_filename)

        with open(stored_filepath, "wb") as out_f:
            out_f.write(raw_bytes)

        # Store metadata in SQLite
        record = database.create_evidence_image(
            image_id=image_id,
            case_id=case_id if case_id else None,
            evidence_id=evidence_id if evidence_id else None,
            filename=stored_filename,
            filepath=stored_filepath,
            original_filename=orig_filename,
            file_size=len(raw_bytes),
            mime_type=mime_type,
            width_px=width_px,
            height_px=height_px,
        )

        if request.headers.get("Accept") == "application/json":
            return jsonify({"status": "ok", "image_id": image_id, "image": record}), 201

        flash(
            f"Evidence photograph '{orig_filename}' ({width_px}x{height_px} px) registered as {image_id}.",
            "success",
        )
        return redirect(
            url_for(
                "measurement_workspace",
                image_id=image_id,
                case_id=case_id,
                evidence_id=evidence_id,
            )
        )

    except ImageProcessingError as img_err:
        if request.headers.get("Accept") == "application/json":
            return jsonify({"status": "error", "error": str(img_err)}), 400
        flash(f"Image Validation Failed: {img_err}", "danger")
        return redirect(url_for("measurement_workspace", case_id=case_id, evidence_id=evidence_id))
    except Exception as exc:
        if request.headers.get("Accept") == "application/json":
            return jsonify({"status": "error", "error": str(exc)}), 500
        flash(f"Unexpected upload error: {exc}", "danger")
        return redirect(url_for("measurement_workspace", case_id=case_id, evidence_id=evidence_id))


# ===========================================================================
# Measurement Workspace Interface (Subphase 2.5 & Phase 3 Computer Vision)
# ===========================================================================


@app.route("/measurement", methods=["GET", "POST"])
def measurement_workspace():
    """
    Measurement workspace for bloodstain optical staging, OpenCV ellipse fitting,
    spatial calibration, and impact angle calculations.
    """
    all_cases = database.get_all_cases()
    all_evidence = database.get_all_evidence()
    recent_images = database.get_all_images(limit=20)

    selected_case = request.values.get("case_id", "").strip()
    selected_evidence = request.values.get("evidence_id", "").strip()
    image_id = request.values.get("image_id", "").strip()

    # Computer Vision Pipeline Parameters
    threshold_method = request.values.get("threshold_method", "otsu").strip().lower()
    invert_binary = request.values.get("invert_binary", "true").lower() != "false"
    try:
        blur_kernel = int(request.values.get("blur_kernel", 5))
    except (ValueError, TypeError):
        blur_kernel = 5
    try:
        min_area = float(request.values.get("min_area", 30.0))
    except (ValueError, TypeError):
        min_area = 30.0
    try:
        selected_candidate_id = int(request.values.get("selected_candidate_id", 1))
    except (ValueError, TypeError):
        selected_candidate_id = 1

    # Spatial Calibration Parameters
    cal_x1 = request.values.get("cal_x1", "").strip()
    cal_y1 = request.values.get("cal_y1", "").strip()
    cal_x2 = request.values.get("cal_x2", "").strip()
    cal_y2 = request.values.get("cal_y2", "").strip()
    known_length_mm_raw = request.values.get("known_length_mm", "").strip()

    # Phase 4.3 Directional & Surface Position Metadata
    direction_angle_raw = request.values.get("direction_angle", "").strip()
    surface_x_raw = request.values.get("surface_x_mm", "").strip()
    surface_y_raw = request.values.get("surface_y_mm", "").strip()
    surface_type = request.values.get("surface_type", "Horizontal Floor").strip() or "Horizontal Floor"

    direction_angle_val = None
    if direction_angle_raw:
        try:
            direction_angle_val = trajectory_integration.validate_direction_angle_input(direction_angle_raw)
        except ReconstructionValidationError as dir_err:
            vision_error = str(dir_err)

    surface_x_val = None
    if surface_x_raw:
        try:
            surface_x_val = float(surface_x_raw)
        except ValueError:
            vision_error = f"Invalid Surface X coordinate: '{surface_x_raw}'"

    surface_y_val = None
    if surface_y_raw:
        try:
            surface_y_val = float(surface_y_raw)
        except ValueError:
            vision_error = f"Invalid Surface Y coordinate: '{surface_y_raw}'"

    result = None
    error = None
    vision_error = vision_error if "vision_error" in locals() else None
    width = 5.0
    length = 10.0
    stain_label = request.values.get("stain_label", "Stain-01").strip() or "Stain-01"

    active_image = None
    vision_data = None
    processed_views = None

    # 1. If an image_id is loaded, run the OpenCV computer vision pipeline
    if image_id:
        active_image = database.get_image_by_id(image_id)
        if not active_image:
            vision_error = f"Evidence image '{image_id}' was not found in the registry."
        else:
            if not selected_case and active_image.get("case_id"):
                selected_case = active_image["case_id"]
            if not selected_evidence and active_image.get("evidence_id"):
                selected_evidence = active_image["evidence_id"]

            scale_bar_points = None
            known_length_mm = None
            if cal_x1 and cal_y1 and cal_x2 and cal_y2 and known_length_mm_raw:
                try:
                    p1 = (float(cal_x1), float(cal_y1))
                    p2 = (float(cal_x2), float(cal_y2))
                    known_length_mm = float(known_length_mm_raw)
                    scale_bar_points = (p1, p2)
                except ValueError as cal_err:
                    vision_error = f"Invalid calibration coordinates or length: {cal_err}"

            try:
                vision_data = image_processing.process_bloodstain_pipeline(
                    image_source=active_image["filepath"],
                    threshold_method=threshold_method,
                    invert_binary=invert_binary,
                    blur_kernel=blur_kernel,
                    min_area=min_area,
                    selected_candidate_id=selected_candidate_id,
                    scale_bar_points=scale_bar_points,
                    known_length_mm=known_length_mm,
                    stain_label=stain_label,
                    direction_angle_deg=direction_angle_val,
                )
                processed_views = image_processing.save_processed_views(
                    pipeline_result=vision_data,
                    output_dir=UPLOAD_PROCESSED_DIR,
                    image_id=image_id,
                )

                # Update database calibration if newly calibrated
                if vision_data["calibration"] and vision_data["calibration"].get("is_calibrated"):
                    database.update_image_calibration(
                        image_id=image_id,
                        scale_factor=vision_data["calibration"]["mm_per_pixel"],
                        is_calibrated=1,
                    )
                    active_image = database.get_image_by_id(image_id)

                # Pre-populate Width and Length from the fitted ellipse
                ellipse = vision_data["ellipse"]
                if ellipse.get("is_calibrated") and ellipse.get("width_mm") and ellipse.get("length_mm"):
                    width = ellipse["width_mm"]
                    length = ellipse["length_mm"]
                    if surface_x_val is None and surface_y_val is None and vision_data.get("calibration"):
                        surf_coords = trajectory.image_to_surface_coordinates(
                            u_px=ellipse["center_x"],
                            v_px=ellipse["center_y"],
                            mm_per_pixel=vision_data["calibration"]["mm_per_pixel"],
                            origin_u_px=0.0,
                            origin_v_px=float(active_image["height_px"]),
                            invert_y=True,
                        )
                        surface_x_val = round(surf_coords["surface_x_mm"], 2)
                        surface_y_val = round(surf_coords["surface_y_mm"], 2)
                else:
                    width = ellipse["minor_axis_px"]
                    length = ellipse["major_axis_px"]

                # Automatically compute trigonometric impact angle from the CV-extracted axes
                result = calculate_impact_angle(width, length)

            except ImageProcessingError as cv_err:
                vision_error = str(cv_err)
            except Exception as exc:
                vision_error = f"Computer vision processing error: {exc}"

    # 2. Handle POST form submission (Manual Calculation or Saving CV/Manual Record)
    if request.method == "POST":
        form_action = request.form.get("form_action", "calculate").strip()
        save_record = request.form.get("save_record") == "true"

        # If manual inputs were explicitly submitted in calculate mode, use them
        if form_action == "calculate" or not vision_data:
            raw_width = request.form.get("width", "").strip()
            raw_length = request.form.get("length", "").strip()
            try:
                width = float(raw_width) if raw_width else None
                length = float(raw_length) if raw_length else None
                result = calculate_impact_angle(width, length)
            except InvalidMeasurementError as err:
                error = str(err)
                result = None
            except (ValueError, TypeError) as err:
                error = f"Invalid numeric input provided: {err}"
                result = None
            except Exception as exc:
                error = f"An unexpected calculation error occurred: {exc}"
                result = None

        # Persist to database if requested
        if save_record and result and not error:
            try:
                width_px = None
                length_px = None
                scale_factor = 1.0
                ellipse_cx = None
                ellipse_cy = None
                ellipse_ang = None
                cal_status = "Uncalibrated (Manual Pixels/mm)"

                if vision_data and vision_data.get("ellipse"):
                    el = vision_data["ellipse"]
                    width_px = el["minor_axis_px"]
                    length_px = el["major_axis_px"]
                    ellipse_cx = el["center_x"]
                    ellipse_cy = el["center_y"]
                    ellipse_ang = el["orientation_deg"]
                    if vision_data.get("calibration") and vision_data["calibration"].get("is_calibrated"):
                        scale_factor = vision_data["calibration"]["mm_per_pixel"]
                        cal_status = f"Calibrated ({vision_data['calibration']['pixels_per_mm']:.2f} px/mm)"
                    elif active_image and active_image.get("is_calibrated") and active_image.get("scale_factor"):
                        scale_factor = float(active_image["scale_factor"])
                        px_per_mm = (1.0 / scale_factor) if scale_factor > 0 else 1.0
                        cal_status = f"Calibrated ({px_per_mm:.2f} px/mm)"
                    else:
                        cal_status = "Uncalibrated Optical (Pixel Units)"

                database.create_measurement(
                    case_id=selected_case if selected_case else None,
                    evidence_id=selected_evidence if selected_evidence else None,
                    stain_label=stain_label,
                    width=result["width"],
                    length=result["length"],
                    ratio=result["ratio"],
                    angle_degrees=result["angle_degrees"],
                    angle_radians=result["angle_radians"],
                    is_perpendicular=result["is_perpendicular"],
                    interpretation=result["interpretation"],
                    image_id=image_id if image_id else None,
                    width_px=width_px,
                    length_px=length_px,
                    scale_factor=scale_factor,
                    ellipse_center_x=ellipse_cx,
                    ellipse_center_y=ellipse_cy,
                    ellipse_angle=ellipse_ang,
                    calibration_status=cal_status,
                    direction_angle=direction_angle_val,
                    surface_x_mm=surface_x_val,
                    surface_y_mm=surface_y_val,
                    surface_type=surface_type,
                )

                if selected_evidence:
                    database.update_evidence_status(selected_evidence, "Analyzed")

                flash(
                    f"Measurement for '{stain_label}' saved to history ({result['angle_degrees']}°).",
                    "success",
                )
            except Exception as exc:
                error = f"Failed to save measurement record: {exc}"

    return render_template(
        "measurement.html",
        active_page="measurement",
        cases=all_cases,
        evidence_list=all_evidence,
        recent_images=recent_images,
        selected_case=selected_case,
        selected_evidence=selected_evidence,
        image_id=image_id,
        active_image=active_image,
        vision_data=vision_data,
        processed_views=processed_views,
        vision_error=vision_error,
        threshold_method=threshold_method,
        invert_binary=invert_binary,
        blur_kernel=blur_kernel,
        min_area=min_area,
        selected_candidate_id=selected_candidate_id,
        cal_x1=cal_x1,
        cal_y1=cal_y1,
        cal_x2=cal_x2,
        cal_y2=cal_y2,
        known_length_mm=known_length_mm_raw,
        stain_label=stain_label,
        direction_angle=direction_angle_val,
        surface_x_mm=surface_x_val,
        surface_y_mm=surface_y_val,
        surface_type=surface_type,
        result=result,
        error=error,
        width=width,
        length=length,
    )


# ===========================================================================
# Results Dashboard (Subphase 2.6)
# ===========================================================================


@app.route("/results", methods=["GET"])
def results_dashboard():
    """
    Forensic results dashboard layout.

    Displays historical stain measurements and designated placeholders for
    area of convergence, 3D area of origin, and annotated evidence overlays.
    """
    case_filter = request.args.get("case_id", "").strip()
    measurements = database.get_recent_measurements(
        limit=20,
        case_id=case_filter if case_filter else None,
    )

    return render_template(
        "results.html",
        active_page="results",
        measurements=measurements,
        selected_case=case_filter,
    )


# ===========================================================================
# 3D Trajectory Reconstruction Laboratory (Phase 4.2)
# ===========================================================================

TRAJECTORY_PRESETS = {
    "floor_radial": {
        "surface_type": "Horizontal Floor",
        "incline_deg": None,
        "stains": [
            {"stain_id": "Stain-01", "width": 7.07, "length": 10.0, "surface_x_mm": 400.0, "surface_y_mm": 150.0, "gamma_deg": 0.0, "sigma_gamma_deg": 2.0},
            {"stain_id": "Stain-02", "width": 7.07, "length": 10.0, "surface_x_mm": 100.0, "surface_y_mm": 450.0, "gamma_deg": 90.0, "sigma_gamma_deg": 2.0},
            {"stain_id": "Stain-03", "width": 7.28, "length": 10.0, "surface_x_mm": 300.0, "surface_y_mm": 350.0, "gamma_deg": 45.0, "sigma_gamma_deg": 2.0},
        ],
    },
    "vertical_wall": {
        "surface_type": "Vertical Wall",
        "incline_deg": None,
        "stains": [
            {"stain_id": "Wall-S1", "width": 5.50, "length": 10.0, "surface_x_mm": 520.0, "surface_y_mm": 220.0, "gamma_deg": 15.0, "sigma_gamma_deg": 2.0},
            {"stain_id": "Wall-S2", "width": 6.10, "length": 10.0, "surface_x_mm": 480.0, "surface_y_mm": 390.0, "gamma_deg": 52.0, "sigma_gamma_deg": 2.0},
            {"stain_id": "Wall-S3", "width": 5.85, "length": 10.0, "surface_x_mm": 210.0, "surface_y_mm": 410.0, "gamma_deg": 118.0, "sigma_gamma_deg": 2.0},
            {"stain_id": "Wall-S4", "width": 5.30, "length": 10.0, "surface_x_mm": 160.0, "surface_y_mm": 240.0, "gamma_deg": 158.0, "sigma_gamma_deg": 2.0},
        ],
    },
    "inclined_45": {
        "surface_type": "Inclined Surface (45°)",
        "incline_deg": 45.0,
        "stains": [
            {"stain_id": "Inc-01", "width": 6.00, "length": 10.0, "surface_x_mm": 380.0, "surface_y_mm": 180.0, "gamma_deg": 12.0, "sigma_gamma_deg": 2.5},
            {"stain_id": "Inc-02", "width": 6.45, "length": 10.0, "surface_x_mm": 290.0, "surface_y_mm": 340.0, "gamma_deg": 55.0, "sigma_gamma_deg": 2.5},
            {"stain_id": "Inc-03", "width": 5.80, "length": 10.0, "surface_x_mm": 110.0, "surface_y_mm": 320.0, "gamma_deg": 115.0, "sigma_gamma_deg": 2.5},
        ],
    },
}


@app.route("/trajectory", methods=["GET", "POST"])
def trajectory_laboratory():
    """
    3D Trajectory Reconstruction & 2D Area of Convergence Visualization Laboratory.
    """
    all_cases = database.get_all_cases()
    preset_key = request.args.get("preset", "floor_radial").strip()
    if preset_key not in TRAJECTORY_PRESETS:
        preset_key = "floor_radial"

    preset = TRAJECTORY_PRESETS[preset_key]
    surface_type = preset["surface_type"]
    incline_deg = preset["incline_deg"]
    input_rows = [dict(r) for r in preset["stains"]]
    selected_case = request.values.get("case_id", "").strip()
    reconstruction_id = request.values.get("reconstruction_id", "").strip()
    traj_error = None

    if request.method == "GET" and (reconstruction_id or selected_case):
        rec_rows = []
        if reconstruction_id:
            rec_obj = database.get_reconstruction_by_id(reconstruction_id)
            if rec_obj:
                selected_case = rec_obj["case_id"]
                surface_type = rec_obj.get("surface_type") or surface_type
                rec_rows = database.get_stains_for_reconstruction(reconstruction_id)
        elif selected_case:
            recons = database.get_reconstructions_for_case(selected_case)
            if recons:
                rec_obj = recons[0]
                surface_type = rec_obj.get("surface_type") or surface_type
                rec_rows = database.get_stains_for_reconstruction(rec_obj["reconstruction_id"])

        valid_rec = [
            r for r in rec_rows
            if r.get("direction_angle") is not None
            and r.get("surface_x_mm") is not None
            and r.get("surface_y_mm") is not None
            and not r.get("is_perpendicular")
        ]
        if len(valid_rec) >= 2:
            input_rows = [
                {
                    "stain_id": r["stain_identifier"],
                    "width": r["width"],
                    "length": r["length"],
                    "surface_x_mm": r["surface_x_mm"],
                    "surface_y_mm": r["surface_y_mm"],
                    "gamma_deg": r["direction_angle"],
                    "sigma_gamma_deg": 2.0,
                }
                for r in valid_rec
            ]

    if request.method == "POST":
        surface_type = request.form.get("surface_type", "Horizontal Floor").strip()
        inc_raw = request.form.get("incline_deg", "").strip()
        if inc_raw:
            try:
                incline_deg = float(inc_raw)
            except ValueError:
                traj_error = f"Invalid custom incline angle: '{inc_raw}'"
                incline_deg = None
        else:
            incline_deg = None

        if request.form.get("load_case_stains") == "true" and selected_case:
            case_meas = database.get_recent_measurements(limit=30, case_id=selected_case)
            valid_rows = [
                m for m in case_meas
                if m.get("direction_angle") is not None
                and m.get("surface_x_mm") is not None
                and m.get("surface_y_mm") is not None
                and not m.get("is_perpendicular")
            ]
            if len(valid_rows) >= 2:
                input_rows = [
                    {
                        "stain_id": m["stain_label"],
                        "width": m["width"],
                        "length": m["length"],
                        "surface_x_mm": m["surface_x_mm"],
                        "surface_y_mm": m["surface_y_mm"],
                        "gamma_deg": m["direction_angle"],
                        "sigma_gamma_deg": 2.0,
                    }
                    for m in valid_rows
                ]
                if valid_rows[0].get("surface_type"):
                    surface_type = valid_rows[0]["surface_type"]
            else:
                traj_error = (
                    f"Case '{selected_case}' has fewer than 2 saved non-perpendicular stains with "
                    "supplied direction_angle and surface coordinates (X_s, Y_s)."
                )
        else:
            stain_ids = request.form.getlist("stain_id")
            widths = request.form.getlist("width")
            lengths = request.form.getlist("length")
            xs = request.form.getlist("surface_x_mm")
            ys = request.form.getlist("surface_y_mm")
            gammas = request.form.getlist("gamma_deg")
            sigmas = request.form.getlist("sigma_gamma_deg")

            parsed_rows = []
            for i in range(len(stain_ids)):
                parsed_rows.append({
                    "stain_id": stain_ids[i].strip() or f"Stain-{i+1:02d}",
                    "width": widths[i].strip() if i < len(widths) else "",
                    "length": lengths[i].strip() if i < len(lengths) else "",
                    "surface_x_mm": xs[i].strip() if i < len(xs) else "",
                    "surface_y_mm": ys[i].strip() if i < len(ys) else "",
                    "gamma_deg": gammas[i].strip() if i < len(gammas) else "",
                    "sigma_gamma_deg": sigmas[i].strip() if i < len(sigmas) and sigmas[i].strip() else "2.0",
                })
            if parsed_rows:
                input_rows = parsed_rows

    # Build StainTrajectoryInput objects and render visualizations
    scene = None
    scene_json = "{}"
    plot_2d_b64 = None
    plot_3d_b64 = None

    if not traj_error:
        try:
            stain_objs = []
            for r in input_rows:
                gamma_val = None if (r["gamma_deg"] is None or str(r["gamma_deg"]).strip() == "") else float(r["gamma_deg"])
                stain_objs.append(
                    StainTrajectoryInput(
                        stain_id=str(r["stain_id"]),
                        width=float(r["width"]),
                        length=float(r["length"]),
                        surface_x_mm=float(r["surface_x_mm"]),
                        surface_y_mm=float(r["surface_y_mm"]),
                        gamma_deg=gamma_val,
                        sigma_gamma_deg=float(r.get("sigma_gamma_deg", 2.0)),
                    )
                )

            vis_2d = trajectory_visualization.render_2d_convergence_plot(
                stains=stain_objs,
                surface_type=surface_type,
            )
            vis_3d = trajectory_visualization.render_3d_trajectory_plot(
                stains=stain_objs,
                surface_type=surface_type,
                incline_deg=incline_deg,
            )
            scene = vis_3d["scene"]
            scene_json = json.dumps(scene)
            plot_2d_b64 = vis_2d["base64_png"]
            plot_3d_b64 = vis_3d["base64_png"]
        except (TrajectoryError, ValueError, TypeError) as err:
            traj_error = str(err)
        except Exception as exc:
            traj_error = f"Unexpected trajectory visualization error: {exc}"

    return render_template(
        "trajectory_lab.html",
        active_page="trajectory",
        cases=all_cases,
        selected_case=selected_case,
        surface_type=surface_type,
        incline_deg=incline_deg,
        input_rows=input_rows,
        scene=scene,
        scene_json=scene_json,
        plot_2d_b64=plot_2d_b64,
        plot_3d_b64=plot_3d_b64,
        traj_error=traj_error,
    )


# ===========================================================================
# Phase 4.3: Measurement Integration & Case Trajectory Reconstruction Routes
# ===========================================================================


@app.route("/reconstruction/<case_id>", methods=["GET"])
def reconstruction_workspace(case_id: str):
    """
    Display the Case Trajectory Reconstruction Workspace.
    Allows examiners to select measurements, inspect calibration and eligibility,
    record surface coordinates and directionality, and calculate 2D/3D convergence.
    """
    reconstruction_id = request.args.get("reconstruction_id", "").strip() or None
    try:
        state = trajectory_integration.get_workspace_state(
            case_id=case_id,
            reconstruction_id=reconstruction_id,
        )
    except ReconstructionValidationError as err:
        if request.headers.get("Accept") == "application/json":
            return jsonify({"status": "error", "error": str(err)}), 404
        flash(str(err), "danger")
        return redirect(url_for("cases_list"))

    # If a previous calculation exists, render static 2D/3D plots for eligible stains
    plot_2d_b64 = None
    plot_3d_b64 = None
    if state["latest_result"] and state["can_calculate"]:
        try:
            stain_objs = [
                StainTrajectoryInput(
                    stain_id=str(s["stain_identifier"]),
                    width=float(s["width"]),
                    length=float(s["length"]),
                    surface_x_mm=float(s["surface_x_mm"]),
                    surface_y_mm=float(s["surface_y_mm"]),
                    gamma_deg=float(s["direction_angle"]),
                    sigma_gamma_deg=2.0,
                )
                for s in state["selected_stains"]
            ]
            surf_type = state["reconstruction"].get("surface_type") or "Horizontal Floor"
            vis_2d = trajectory_visualization.render_2d_convergence_plot(
                stains=stain_objs,
                surface_type=surf_type,
            )
            vis_3d = trajectory_visualization.render_3d_trajectory_plot(
                stains=stain_objs,
                surface_type=surf_type,
            )
            plot_2d_b64 = vis_2d["base64_png"]
            plot_3d_b64 = vis_3d["base64_png"]
        except Exception:
            pass

    return render_template(
        "reconstruction.html",
        active_page="cases",
        case=state["case"],
        workspace=state,
        active_scene=state["latest_scene"],
        plot_2d_b64=plot_2d_b64,
        plot_3d_b64=plot_3d_b64,
    )


@app.route("/reconstruction/<case_id>/add-measurement", methods=["POST"])
def reconstruction_add_measurement(case_id: str):
    """Associate a saved case measurement with the case's trajectory reconstruction workspace."""
    reconstruction_id = request.form.get("reconstruction_id", "").strip() or None
    measurement_id_raw = request.form.get("measurement_id", "").strip()
    stain_identifier = request.form.get("stain_identifier", "").strip() or None
    surface_x_raw = request.form.get("surface_x_mm", "").strip() or None
    surface_y_raw = request.form.get("surface_y_mm", "").strip() or None
    direction_angle_raw = request.form.get("direction_angle", "").strip() or None
    direction_source = request.form.get("direction_source", "").strip() or None

    try:
        if not measurement_id_raw:
            raise ReconstructionValidationError("Measurement ID is required to add a stain to the workspace.")
        measurement_id = int(measurement_id_raw)
        rec = trajectory_integration.get_or_create_case_reconstruction(case_id, reconstruction_id)
        added = trajectory_integration.add_measurement_to_case_reconstruction(
            case_id=case_id,
            reconstruction_id=rec["reconstruction_id"],
            measurement_id=measurement_id,
            stain_identifier=stain_identifier,
            surface_x_mm=surface_x_raw,
            surface_y_mm=surface_y_raw,
            direction_angle=direction_angle_raw,
            direction_source=direction_source,
        )
        if request.headers.get("Accept") == "application/json":
            return jsonify({"status": "ok", "stain": added}), 201
        flash(f"Added stain '{added['stain_identifier']}' (Measurement #{measurement_id}) to reconstruction.", "success")
    except (ReconstructionValidationError, ValueError) as err:
        if request.headers.get("Accept") == "application/json":
            return jsonify({"status": "error", "error": str(err)}), 400
        flash(f"Could not add measurement: {err}", "danger")

    return redirect(url_for("reconstruction_workspace", case_id=case_id, reconstruction_id=reconstruction_id))


@app.route("/reconstruction/<case_id>/remove-measurement", methods=["POST"])
def reconstruction_remove_measurement(case_id: str):
    """Remove a measurement from the case's trajectory reconstruction workspace."""
    reconstruction_id = request.form.get("reconstruction_id", "").strip() or None
    measurement_id_raw = request.form.get("measurement_id", "").strip()

    try:
        if not measurement_id_raw:
            raise ReconstructionValidationError("Measurement ID is required to remove a stain from the workspace.")
        measurement_id = int(measurement_id_raw)
        rec = trajectory_integration.get_or_create_case_reconstruction(case_id, reconstruction_id)
        trajectory_integration.remove_measurement_from_case_reconstruction(
            case_id=case_id,
            reconstruction_id=rec["reconstruction_id"],
            measurement_id=measurement_id,
        )
        if request.headers.get("Accept") == "application/json":
            return jsonify({"status": "ok", "removed_measurement_id": measurement_id}), 200
        flash(f"Removed Measurement #{measurement_id} from reconstruction workspace.", "info")
    except (ReconstructionValidationError, ValueError) as err:
        if request.headers.get("Accept") == "application/json":
            return jsonify({"status": "error", "error": str(err)}), 400
        flash(f"Could not remove measurement: {err}", "danger")

    return redirect(url_for("reconstruction_workspace", case_id=case_id, reconstruction_id=reconstruction_id))


@app.route("/reconstruction/<case_id>/update-stain", methods=["POST"])
def reconstruction_update_stain(case_id: str):
    """Update directional angle, surface position coordinates, or label for a workspace stain."""
    reconstruction_id = request.form.get("reconstruction_id", "").strip() or None
    measurement_id_raw = request.form.get("measurement_id", "").strip()
    stain_identifier = request.form.get("stain_identifier", "").strip() or None
    surface_x_raw = request.form.get("surface_x_mm", "").strip()
    surface_y_raw = request.form.get("surface_y_mm", "").strip()
    direction_angle_raw = request.form.get("direction_angle", "").strip()
    direction_source = request.form.get("direction_source", "").strip() or "Examiner Visual Inspection"

    try:
        if not measurement_id_raw:
            raise ReconstructionValidationError("Measurement ID is required to update stain parameters.")
        measurement_id = int(measurement_id_raw)
        rec = trajectory_integration.get_or_create_case_reconstruction(case_id, reconstruction_id)
        updated = trajectory_integration.update_stain_in_case_reconstruction(
            case_id=case_id,
            reconstruction_id=rec["reconstruction_id"],
            measurement_id=measurement_id,
            stain_identifier=stain_identifier,
            surface_x_mm=surface_x_raw if surface_x_raw != "" else None,
            surface_y_mm=surface_y_raw if surface_y_raw != "" else None,
            direction_angle=direction_angle_raw if direction_angle_raw != "" else None,
            direction_source=direction_source,
        )
        if request.headers.get("Accept") == "application/json":
            return jsonify({"status": "ok", "stain": updated}), 200
        flash(f"Updated parameters for stain '{updated['stain_identifier']}'.", "success")
    except (ReconstructionValidationError, ValueError) as err:
        if request.headers.get("Accept") == "application/json":
            return jsonify({"status": "error", "error": str(err)}), 400
        flash(f"Stain update failed: {err}", "danger")

    return redirect(url_for("reconstruction_workspace", case_id=case_id, reconstruction_id=reconstruction_id))


@app.route("/reconstruction/<case_id>/calculate", methods=["POST"])
def reconstruction_calculate(case_id: str):
    """
    Validate all stains in the reconstruction workspace, run 2D Area of Convergence
    and 3D Area of Origin via trajectory.py, and persist results to SQLite.
    """
    reconstruction_id = request.form.get("reconstruction_id", "").strip() or None
    surface_type = request.form.get("surface_type", "").strip() or None
    incline_raw = request.form.get("incline_deg", "").strip()
    notes = request.form.get("notes", "").strip() or None

    incline_deg = None
    if incline_raw:
        try:
            incline_deg = float(incline_raw)
        except ValueError:
            msg = f"Invalid incline angle: '{incline_raw}'"
            if request.headers.get("Accept") == "application/json":
                return jsonify({"status": "error", "error": msg}), 400
            flash(msg, "danger")
            return redirect(url_for("reconstruction_workspace", case_id=case_id, reconstruction_id=reconstruction_id))

    try:
        rec = trajectory_integration.get_or_create_case_reconstruction(case_id, reconstruction_id)
        calc_output = trajectory_integration.execute_case_reconstruction(
            case_id=case_id,
            reconstruction_id=rec["reconstruction_id"],
            surface_type=surface_type,
            incline_deg=incline_deg,
        )
        if request.headers.get("Accept") == "application/json":
            return jsonify(
                {
                    "status": "ok",
                    "reconstruction": calc_output["reconstruction"],
                    "result": calc_output["result"],
                }
            ), 200
        flash(
            f"Trajectory Reconstruction {rec['reconstruction_id']} computed and saved "
            f"({calc_output['result']['stain_count']} stains).",
            "success",
        )
    except ReconstructionValidationError as err:
        if request.headers.get("Accept") == "application/json":
            return jsonify({"status": "error", "error": str(err)}), 400
        flash(f"Reconstruction Validation Error: {err}", "danger")
    except Exception as exc:
        if request.headers.get("Accept") == "application/json":
            return jsonify({"status": "error", "error": str(exc)}), 500
        flash(f"Unexpected reconstruction calculation error: {exc}", "danger")

    return redirect(url_for("reconstruction_workspace", case_id=case_id, reconstruction_id=reconstruction_id))


@app.route("/reconstruction/<case_id>/results", methods=["GET"])
def reconstruction_results_view(case_id: str):
    """
    Retrieve the latest persisted trajectory reconstruction results for a case.
    Returns structured JSON when requested via Accept: application/json, or renders the workspace.
    """
    reconstruction_id = request.args.get("reconstruction_id", "").strip() or None
    try:
        state = trajectory_integration.get_workspace_state(
            case_id=case_id,
            reconstruction_id=reconstruction_id,
        )
    except ReconstructionValidationError as err:
        if request.headers.get("Accept") == "application/json":
            return jsonify({"status": "error", "error": str(err)}), 404
        flash(str(err), "danger")
        return redirect(url_for("cases_list"))

    if request.headers.get("Accept") == "application/json":
        return jsonify(
            {
                "status": "ok",
                "case_id": case_id,
                "reconstruction": state["reconstruction"],
                "stains": state["selected_stains"],
                "latest_result": state["latest_result"],
            }
        ), 200

    return redirect(url_for("reconstruction_workspace", case_id=case_id, reconstruction_id=reconstruction_id))


# ===========================================================================
# Phase 4.4: Professional Forensic PDF Report Generation & Archive Routes
# ===========================================================================


def _render_reports_page(selected_report=None):
    """Helper to render the Forensic PDF Reports dashboard."""
    all_cases = database.get_all_cases()
    all_measurements = database.get_recent_measurements(limit=100)
    all_reconstructions = []
    for c in all_cases:
        all_reconstructions.extend(database.get_reconstructions_for_case(c["case_id"]))
    reports = database.get_all_reports(limit=100)
    return render_template(
        "reports.html",
        active_page="reports",
        cases=all_cases,
        measurements=all_measurements,
        reconstructions=all_reconstructions,
        reports=reports,
        selected_report=selected_report,
    )


@app.route("/reports", methods=["GET"])
def reports_dashboard():
    """Display the Forensic PDF Report Generator & Archive Registry."""
    if request.headers.get("Accept") == "application/json":
        return jsonify({"status": "ok", "reports": database.get_all_reports(limit=100)}), 200
    return _render_reports_page(selected_report=None)


@app.route("/reports/generate", methods=["POST"])
def generate_report_dispatcher():
    """Form dispatcher for the 3 report generation stations on /reports."""
    dispatch_type = request.form.get("dispatch_type", "").strip().lower()
    target_id = request.form.get("target_id", "").strip()
    if dispatch_type == "measurement":
        return generate_measurement_report_route(target_id)
    if dispatch_type == "reconstruction":
        return generate_reconstruction_report_route(target_id)
    if dispatch_type == "case":
        return generate_case_report_route(target_id)
    flash("Please select a valid report type and target identifier.", "danger")
    return redirect(url_for("reports_dashboard"))


@app.route("/reports/generate/measurement/<measurement_id>", methods=["POST"])
def generate_measurement_report_route(measurement_id: str):
    """Generate a Type A Individual Bloodstain Examination PDF Report."""
    report_status = request.form.get("report_status", "DRAFT").strip()
    examiner_name = request.form.get("examiner_name", "").strip() or None
    laboratory_name = request.form.get("laboratory_name", "").strip() or None
    examiner_observations = request.form.get("examiner_observations", "").strip() or None

    try:
        record = report_generator.generate_measurement_report(
            measurement_id=measurement_id,
            output_dir=UPLOAD_REPORTS_DIR,
            report_status=report_status,
            examiner_name=examiner_name,
            laboratory_name=laboratory_name,
            examiner_observations=examiner_observations,
        )
        if request.headers.get("Accept") == "application/json":
            return jsonify({"status": "ok", "report": record}), 201
        flash(
            f"Forensic Measurement Report {record['report_id']} (Rev R{record['revision_number']}, {record['report_status']}) generated.",
            "success",
        )
        return redirect(url_for("report_detail_view", report_id=record["report_id"]))
    except ReportGenerationError as err:
        if request.headers.get("Accept") == "application/json":
            return jsonify({"status": "error", "error": str(err)}), 400
        flash(f"Report Generation Failed: {err}", "danger")
        return redirect(url_for("reports_dashboard"))


@app.route("/reports/generate/reconstruction/<reconstruction_id>", methods=["POST"])
def generate_reconstruction_report_route(reconstruction_id: str):
    """Generate a Type B Multiple Bloodstain Trajectory Reconstruction PDF Report."""
    report_status = request.form.get("report_status", "DRAFT").strip()
    examiner_name = request.form.get("examiner_name", "").strip() or None
    laboratory_name = request.form.get("laboratory_name", "").strip() or None
    examiner_observations = request.form.get("examiner_observations", "").strip() or None

    try:
        record = report_generator.generate_reconstruction_report(
            reconstruction_id=reconstruction_id,
            output_dir=UPLOAD_REPORTS_DIR,
            report_status=report_status,
            examiner_name=examiner_name,
            laboratory_name=laboratory_name,
            examiner_observations=examiner_observations,
        )
        if request.headers.get("Accept") == "application/json":
            return jsonify({"status": "ok", "report": record}), 201
        flash(
            f"Trajectory Reconstruction Report {record['report_id']} (Rev R{record['revision_number']}, {record['report_status']}) generated.",
            "success",
        )
        return redirect(url_for("report_detail_view", report_id=record["report_id"]))
    except ReportGenerationError as err:
        if request.headers.get("Accept") == "application/json":
            return jsonify({"status": "error", "error": str(err)}), 400
        flash(f"Report Generation Failed: {err}", "danger")
        return redirect(url_for("reports_dashboard"))


@app.route("/reports/generate/case/<case_id>", methods=["POST"])
def generate_case_report_route(case_id: str):
    """Generate a Type C Complete Forensic Case PDF Report."""
    report_status = request.form.get("report_status", "DRAFT").strip()
    examiner_name = request.form.get("examiner_name", "").strip() or None
    laboratory_name = request.form.get("laboratory_name", "").strip() or None
    examiner_observations = request.form.get("examiner_observations", "").strip() or None

    try:
        record = report_generator.generate_case_report(
            case_id=case_id,
            output_dir=UPLOAD_REPORTS_DIR,
            report_status=report_status,
            examiner_name=examiner_name,
            laboratory_name=laboratory_name,
            examiner_observations=examiner_observations,
        )
        if request.headers.get("Accept") == "application/json":
            return jsonify({"status": "ok", "report": record}), 201
        flash(
            f"Complete Forensic Case Report {record['report_id']} (Rev R{record['revision_number']}, {record['report_status']}) generated.",
            "success",
        )
        return redirect(url_for("report_detail_view", report_id=record["report_id"]))
    except ReportGenerationError as err:
        if request.headers.get("Accept") == "application/json":
            return jsonify({"status": "error", "error": str(err)}), 400
        flash(f"Report Generation Failed: {err}", "danger")
        return redirect(url_for("reports_dashboard"))


@app.route("/reports/<path:report_id>/download", methods=["GET"])
def download_report_pdf(report_id: str):
    """
    Securely serve a generated forensic PDF report by report_id.
    Blocks path traversal and verifies SHA-256 binary integrity before serving.
    """
    safe_id = secure_filename(report_id)
    if not safe_id or safe_id != report_id or ".." in report_id or "/" in report_id or "\\" in report_id:
        abort(404)

    record = database.get_report_by_id(safe_id)
    if not record:
        abort(404)

    safe_filename = secure_filename(record["filename"])
    if not safe_filename or safe_filename != record["filename"]:
        abort(404)

    filepath = record["filepath"]
    if not filepath or not os.path.isfile(filepath):
        abort(404)

    directory = os.path.dirname(os.path.abspath(filepath))
    return send_from_directory(
        directory,
        safe_filename,
        as_attachment=True,
        download_name=safe_filename,
        mimetype="application/pdf",
    )


@app.route("/reports/<path:report_id>", methods=["GET"])
def report_detail_view(report_id: str):
    """Inspect metadata and SHA-256 traceability for a generated forensic report."""
    safe_id = secure_filename(report_id)
    if not safe_id or safe_id != report_id or ".." in report_id or "/" in report_id or "\\" in report_id:
        if request.headers.get("Accept") == "application/json":
            return jsonify({"status": "error", "error": "Invalid report ID."}), 404
        abort(404)

    record = database.get_report_by_id(safe_id)
    if not record:
        if request.headers.get("Accept") == "application/json":
            return jsonify({"status": "error", "error": f"Report '{report_id}' not found."}), 404
        flash(f"Report '{report_id}' was not found in the archive.", "danger")
        return redirect(url_for("reports_dashboard"))

    if request.headers.get("Accept") == "application/json":
        return jsonify({"status": "ok", "report": record}), 200

    return _render_reports_page(selected_report=record)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
