"""
app.py
======
Forensic Laboratory Web Application for Bloodstain Pattern Analysis (BPA).

Provides interactive case management, physical evidence cataloging,
and droplet impact angle calculations using elliptical geometry.
"""

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
from calculator import InvalidMeasurementError, calculate_impact_angle
from image_processing import ImageProcessingError

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "bpa-forensic-lab-secret-2026")
app.config["MAX_CONTENT_LENGTH"] = image_processing.MAX_FILE_SIZE_BYTES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_ORIGINAL_DIR = os.path.join(BASE_DIR, "uploads", "original")
UPLOAD_PROCESSED_DIR = os.path.join(BASE_DIR, "uploads", "processed")
SAMPLE_IMAGES_DIR = os.path.join(BASE_DIR, "sample_images")

os.makedirs(UPLOAD_ORIGINAL_DIR, exist_ok=True)
os.makedirs(UPLOAD_PROCESSED_DIR, exist_ok=True)

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

    result = None
    error = None
    vision_error = None
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


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
