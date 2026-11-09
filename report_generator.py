"""
report_generator.py
===================
Professional Forensic PDF Report Generation Engine for the
Bloodstain Pattern Measurement Calculator — Forensic Laboratory Suite (Phase 4.4).

Responsibilities:
1. Generate publication-grade, A4-formatted forensic PDF reports using ReportLab:
   - Individual Bloodstain Examination Report (Type A)
   - Multiple Bloodstain Trajectory Reconstruction Report (Type B)
   - Complete Forensic Case Report (Type C)
2. Maintain strict independence from Flask routes while reusing existing validated
   modules (database.py, calculator.py, image_processing.py, trajectory.py,
   trajectory_integration.py, trajectory_visualization.py) without duplicating math.
3. Enforce forensic traceability, revision control (never overwriting existing FINAL
   or prior reports), SHA-256 cryptographic file hashing, proportional image scaling,
   repeated table headers, and page-numbered headers/footers.
"""

import base64
import hashlib
import io
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image as PILImage
from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from werkzeug.utils import secure_filename

import database
import image_processing
import trajectory
import trajectory_integration
import trajectory_visualization
from trajectory import StainTrajectoryInput

# Keep PDF content streams uncompressed so text/metadata are directly verifiable
# while remaining 100% compliant with ISO 32000 / PDF-1.4 readers.
rl_config.pageCompression = 0

SOFTWARE_VERSION = "Bloodstain Pattern Measurement Calculator — Forensic Laboratory Suite v2.4 (Phase 4.4)"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REPORTS_DIR = os.path.join(BASE_DIR, "uploads", "reports")
os.makedirs(DEFAULT_REPORTS_DIR, exist_ok=True)


class ReportGenerationError(ValueError):
    """Exception raised when forensic report validation or PDF generation fails."""
    pass


# ===========================================================================
# 1. Cryptographic SHA-256 & Image Scaling Utilities
# ===========================================================================


def compute_file_sha256(filepath: str) -> Optional[str]:
    """Compute the lowercase hexadecimal SHA-256 digest of a file on disk."""
    if not filepath or not os.path.isfile(filepath):
        return None
    hasher = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest().lower()
    except OSError:
        return None


def compute_bytes_sha256(data: bytes) -> str:
    """Compute the lowercase hexadecimal SHA-256 digest of raw bytes."""
    return hashlib.sha256(data).hexdigest().lower()


def _create_proportional_rl_image(
    image_source: Any,
    max_width_mm: float = 155.0,
    max_height_mm: float = 85.0,
) -> Optional[RLImage]:
    """
    Create a ReportLab Image flowable scaled proportionally within (max_width_mm, max_height_mm)
    without aspect-ratio distortion. Accepts either a file path or raw PNG bytes / BytesIO.
    """
    try:
        if isinstance(image_source, (bytes, bytearray)):
            bio = io.BytesIO(image_source)
            with PILImage.open(bio) as pil_img:
                w_px, h_px = pil_img.size
            bio.seek(0)
            img_stream = bio
        elif isinstance(image_source, str) and os.path.isfile(image_source):
            with PILImage.open(image_source) as pil_img:
                w_px, h_px = pil_img.size
            with open(image_source, "rb") as f:
                img_stream = io.BytesIO(f.read())
        else:
            return None

        if w_px <= 0 or h_px <= 0:
            return None

        max_w_pt = max_width_mm * mm
        max_h_pt = max_height_mm * mm
        scale = min(max_w_pt / float(w_px), max_h_pt / float(h_px))
        disp_w = float(w_px) * scale
        disp_h = float(h_px) * scale

        rl_img = RLImage(img_stream, width=disp_w, height=disp_h)
        rl_img.hAlign = "CENTER"
        return rl_img
    except Exception:
        return None


def _decode_data_uri_png(data_uri: Optional[str]) -> Optional[bytes]:
    """Decode a 'data:image/png;base64,...' string into raw PNG bytes."""
    if not data_uri or not isinstance(data_uri, str):
        return None
    if "," in data_uri:
        b64_part = data_uri.split(",", 1)[1]
    else:
        b64_part = data_uri
    try:
        return base64.b64decode(b64_part)
    except Exception:
        return None


# ===========================================================================
# 2. Custom Page-Numbered Forensic Canvas (Headers, Footers & Page X of Y)
# ===========================================================================


class NumberedForensicCanvas(canvas.Canvas):
    """
    Two-pass ReportLab Canvas that renders a restrained forensic header and footer
    with exact 'Page X of Y', Report ID, Case ID, Revision, and DRAFT/FINAL status.
    """

    def __init__(self, *args, **kwargs):
        self.report_meta: Dict[str, Any] = kwargs.pop("report_meta", {})
        super().__init__(*args, **kwargs)
        self._saved_page_states: List[Dict[str, Any]] = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self._draw_header_and_footer(num_pages)
            super().showPage()
        super().save()

    def _draw_header_and_footer(self, page_count: int):
        page_num = self._pageNumber
        width_pt, height_pt = A4
        margin_x = 18.0 * mm

        report_id = self.report_meta.get("report_id", "RPT-UNASSIGNED")
        case_id = self.report_meta.get("case_id") or "STANDALONE"
        status = self.report_meta.get("report_status", "DRAFT")
        revision = self.report_meta.get("revision_number", 1)
        gen_time = self.report_meta.get("generated_at", "")

        self.saveState()

        # Header bar on all pages
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.6)
        self.line(margin_x, height_pt - 14 * mm, width_pt - margin_x, height_pt - 14 * mm)

        self.setFont("Helvetica-Bold", 7.5)
        self.setFillColor(colors.HexColor("#1e293b"))
        self.drawString(
            margin_x,
            height_pt - 11.5 * mm,
            f"BLOODSTAIN PATTERN MEASUREMENT CALCULATOR  |  Report ID: {report_id} (Rev R{revision})",
        )

        # Status badge text on top right
        status_color = colors.HexColor("#b91c1c") if status == "FINAL" else colors.HexColor("#b45309")
        self.setFillColor(status_color)
        self.drawRightString(
            width_pt - margin_x,
            height_pt - 11.5 * mm,
            f"Case: {case_id}  |  Status: {status}",
        )

        # Footer bar on all pages
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.line(margin_x, 14 * mm, width_pt - margin_x, 14 * mm)

        self.setFont("Helvetica", 7.5)
        self.setFillColor(colors.HexColor("#475569"))
        self.drawString(
            margin_x,
            10 * mm,
            f"Generated: {gen_time}  |  Forensic Laboratory Examination Document",
        )
        self.setFont("Helvetica-Bold", 8.0)
        self.setFillColor(colors.HexColor("#0f172a"))
        self.drawRightString(
            width_pt - margin_x,
            10 * mm,
            f"Page {page_num} of {page_count}",
        )

        self.restoreState()


# ===========================================================================
# 3. Typography & Table Stylesheet Builder
# ===========================================================================


def _build_forensic_styles() -> Dict[str, ParagraphStyle]:
    """Create a cohesive, restrained scientific typography hierarchy for A4 reports."""
    base = getSampleStyleSheet()
    styles: Dict[str, ParagraphStyle] = {}

    styles["CoverAppTitle"] = ParagraphStyle(
        "CoverAppTitle",
        parent=base["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#991b1b"),
        alignment=TA_LEFT,
        spaceAfter=4,
    )
    styles["CoverMainTitle"] = ParagraphStyle(
        "CoverMainTitle",
        parent=base["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=25,
        textColor=colors.HexColor("#0f172a"),
        alignment=TA_LEFT,
        spaceAfter=8,
    )
    styles["CoverSubtitle"] = ParagraphStyle(
        "CoverSubtitle",
        parent=base["Normal"],
        fontName="Helvetica",
        fontSize=10.5,
        leading=15,
        textColor=colors.HexColor("#334155"),
        alignment=TA_LEFT,
        spaceAfter=14,
    )
    styles["SectionHeading"] = ParagraphStyle(
        "SectionHeading",
        parent=base["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=12.5,
        leading=16,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True,
    )
    styles["SubHeading"] = ParagraphStyle(
        "SubHeading",
        parent=base["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=10.0,
        leading=13.5,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True,
    )
    styles["Body"] = ParagraphStyle(
        "Body",
        parent=base["Normal"],
        fontName="Helvetica",
        fontSize=9.0,
        leading=13.0,
        textColor=colors.HexColor("#1e293b"),
        alignment=TA_JUSTIFY,
        spaceAfter=5,
    )
    styles["BodyLeft"] = ParagraphStyle(
        "BodyLeft",
        parent=styles["Body"],
        alignment=TA_LEFT,
    )
    styles["EquationBox"] = ParagraphStyle(
        "EquationBox",
        parent=base["Code"],
        fontName="Courier-Bold",
        fontSize=8.5,
        leading=12.0,
        textColor=colors.HexColor("#0f172a"),
        backColor=colors.HexColor("#f1f5f9"),
        borderPadding=6,
        spaceBefore=4,
        spaceAfter=6,
    )
    styles["Caption"] = ParagraphStyle(
        "Caption",
        parent=base["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8.0,
        leading=11.0,
        textColor=colors.HexColor("#475569"),
        alignment=TA_CENTER,
        spaceBefore=3,
        spaceAfter=8,
    )
    styles["TableCell"] = ParagraphStyle(
        "TableCell",
        parent=base["Normal"],
        fontName="Helvetica",
        fontSize=8.0,
        leading=10.5,
        textColor=colors.HexColor("#0f172a"),
    )
    styles["TableHeader"] = ParagraphStyle(
        "TableHeader",
        parent=base["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.0,
        leading=10.5,
        textColor=colors.HexColor("#ffffff"),
    )
    styles["NoticeBox"] = ParagraphStyle(
        "NoticeBox",
        parent=base["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12.0,
        textColor=colors.HexColor("#7f1d1d"),
        backColor=colors.HexColor("#fef2f2"),
        borderPadding=7,
        spaceBefore=5,
        spaceAfter=7,
    )
    return styles


def _standard_table_style() -> TableStyle:
    """Return a clean, high-contrast scientific TableStyle with repeated header support."""
    return TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#ffffff")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 8.0),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
            ("TOPPADDING", (0, 0), (-1, 0), 6),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 1), (-1, -1), 8.0),
            ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#0f172a")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#ffffff"), colors.HexColor("#f8fafc")]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 1), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 1), (-1, -1), 4),
        ]
    )


def _kv_table_style() -> TableStyle:
    """Return a key-value metadata table style."""
    return TableStyle(
        [
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f1f5f9")),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#0f172a")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]
    )


# ===========================================================================
# 4. Section Builders (Sections 1 through 12)
# ===========================================================================


def _build_section_1_cover(
    story: List[Any],
    styles: Dict[str, ParagraphStyle],
    meta: Dict[str, Any],
    case: Optional[Dict[str, Any]],
) -> None:
    """Section 1 — Cover Page."""
    story.append(Spacer(1, 6 * mm))
    story.append(Paragraph("BLOODSTAIN PATTERN MEASUREMENT CALCULATOR", styles["CoverAppTitle"]))
    story.append(Paragraph(meta["report_title"], styles["CoverMainTitle"]))
    story.append(
        Paragraph(
            f"Forensic Laboratory Technical Examination Document — {meta['report_type']}",
            styles["CoverSubtitle"],
        )
    )
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#991b1b"), spaceAfter=8))

    examiner_display = (
        meta["examiner_name"]
        if meta.get("examiner_name")
        else (case.get("examiner") if case and case.get("examiner") else "Not Supplied (Examiner Unspecified)")
    )
    lab_display = (
        meta["laboratory_name"]
        if meta.get("laboratory_name")
        else "Not Supplied (Laboratory Unspecified)"
    )

    cover_rows = [
        ["Application Name", "Bloodstain Pattern Measurement Calculator — Forensic Laboratory Suite"],
        ["Report Identifier", f"{meta['report_id']}  (Revision R{meta['revision_number']})"],
        ["Report Classification", f"{meta['report_type']}"],
        ["Report Status", f"{meta['report_status']}"],
        ["Case Identifier", case["case_id"] if case else "Standalone Examination (No Case Linked)"],
        ["Case Reference Number", case["reference_no"] if case and case.get("reference_no") else "N/A"],
        ["Date & Time Generated", meta["generated_at"]],
        ["Reporting Examiner", examiner_display],
        ["Laboratory Unit", lab_display],
    ]
    if meta.get("reconstruction_id"):
        cover_rows.insert(6, ["Reconstruction Identifier", meta["reconstruction_id"]])

    tbl = Table(cover_rows, colWidths=[52 * mm, 122 * mm])
    tbl.setStyle(_kv_table_style())
    story.append(tbl)
    story.append(Spacer(1, 6 * mm))

    story.append(
        Paragraph(
            "<b>Forensic Traceability &amp; Non-Fabrication Notice:</b> This document reports objective mathematical "
            "and optical measurements derived from recorded evidence records. Examiner credentials, institutional "
            "accreditation, and scene directional parameters are reported strictly as supplied and are never inferred "
            "or fabricated by the software.",
            styles["NoticeBox"],
        )
    )
    story.append(Spacer(1, 4 * mm))


def _build_section_2_case_info(
    story: List[Any],
    styles: Dict[str, ParagraphStyle],
    case: Optional[Dict[str, Any]],
    evidence_items: List[Dict[str, Any]],
    images: List[Dict[str, Any]],
) -> None:
    """Section 2 — Case Information."""
    story.append(Paragraph("SECTION 2 — CASE INFORMATION", styles["SectionHeading"]))
    story.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#cbd5e1"), spaceAfter=6))

    ev_ids = ", ".join(sorted({e["evidence_id"] for e in evidence_items if e.get("evidence_id")})) or "None Logged"
    img_ids = ", ".join(sorted({im["image_id"] for im in images if im.get("image_id")})) or "None Associated"

    if case:
        rows = [
            ["Case Number (ID)", case["case_id"]],
            ["Case Title / Name", case.get("title") or "N/A"],
            ["Case Reference No.", case.get("reference_no") or "N/A"],
            ["Date of Examination", case.get("examination_date") or "Not Specified"],
            ["Assigned Examiner", case.get("examiner") or "Not Specified"],
            ["Case Status", case.get("status") or "Active"],
            ["Case Description", Paragraph(case.get("description") or "No description recorded.", styles["TableCell"])],
            ["Associated Evidence IDs", ev_ids],
            ["Associated Image IDs", img_ids],
        ]
    else:
        rows = [
            ["Case Number (ID)", "Standalone Examination (Not linked to a registered case)"],
            ["Associated Evidence IDs", ev_ids],
            ["Associated Image IDs", img_ids],
        ]

    tbl = Table(rows, colWidths=[52 * mm, 122 * mm])
    tbl.setStyle(_kv_table_style())
    story.append(tbl)
    story.append(Spacer(1, 4 * mm))


def _build_section_3_evidence_documentation(
    story: List[Any],
    styles: Dict[str, ParagraphStyle],
    evidence_items: List[Dict[str, Any]],
    images: List[Dict[str, Any]],
    image_hashes: Dict[str, str],
    fig_counter: List[int],
) -> None:
    """Section 3 — Evidence Documentation & Unmodified Original Photographs."""
    story.append(Paragraph("SECTION 3 — EVIDENCE DOCUMENTATION & ORIGINAL PHOTOGRAPHS", styles["SectionHeading"]))
    story.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#cbd5e1"), spaceAfter=6))

    if evidence_items:
        ev_rows = [["Evidence ID", "Pattern Type", "Receiving Surface / Location", "Status", "Description"]]
        for ev in evidence_items:
            ev_rows.append(
                [
                    ev.get("evidence_id", ""),
                    ev.get("evidence_type", ""),
                    Paragraph(str(ev.get("location_surface") or "N/A"), styles["TableCell"]),
                    ev.get("status", ""),
                    Paragraph(str(ev.get("description") or ""), styles["TableCell"]),
                ]
            )
        ev_tbl = Table(ev_rows, colWidths=[26 * mm, 32 * mm, 42 * mm, 20 * mm, 54 * mm], repeatRows=1)
        ev_tbl.setStyle(_standard_table_style())
        story.append(ev_tbl)
        story.append(Spacer(1, 4 * mm))
    else:
        story.append(Paragraph("No physical evidence items were separately logged for this scope.", styles["BodyLeft"]))

    if images:
        story.append(Paragraph("3.1 Photographic Evidence Registry & Cryptographic Hashes", styles["SubHeading"]))
        img_rows = [
            ["Image ID", "Evidence ID", "Original Filename", "Dimensions", "Size / MIME", "Calibration", "Original File SHA-256"]
        ]
        for im in images:
            iid = im["image_id"]
            cal_txt = (
                f"Calibrated ({im['scale_factor']:.4f} mm/px)"
                if im.get("is_calibrated")
                else "Uncalibrated (Pixels)"
            )
            sha_val = image_hashes.get(iid) or "File Missing on Disk"
            sha_short = f"{sha_val[:14]}...{sha_val[-8:]}" if len(sha_val) == 64 else sha_val
            img_rows.append(
                [
                    Paragraph(iid, styles["TableCell"]),
                    Paragraph(str(im.get("evidence_id") or "Unlinked"), styles["TableCell"]),
                    Paragraph(str(im.get("original_filename") or ""), styles["TableCell"]),
                    Paragraph(f"{im.get('width_px', 0)}x{im.get('height_px', 0)} px", styles["TableCell"]),
                    Paragraph(f"{im.get('file_size', 0)} B<br/>{im.get('mime_type', '')}", styles["TableCell"]),
                    Paragraph(cal_txt, styles["TableCell"]),
                    Paragraph(f"<font name='Courier' size='6.5'>{sha_short}</font>", styles["TableCell"]),
                ]
            )
        img_tbl = Table(img_rows, colWidths=[23 * mm, 21 * mm, 29 * mm, 20 * mm, 21 * mm, 29 * mm, 31 * mm], repeatRows=1)
        img_tbl.setStyle(_standard_table_style())
        story.append(img_tbl)
        story.append(Spacer(1, 4 * mm))

        # Embed unmodified original images
        for im in images:
            iid = im["image_id"]
            orig_path = im.get("filepath")
            fig_num = fig_counter[0]
            fig_counter[0] += 1
            if orig_path and os.path.isfile(orig_path):
                rl_img = _create_proportional_rl_image(orig_path, max_width_mm=115.0, max_height_mm=65.0)
                if rl_img:
                    story.append(
                        KeepTogether(
                            [
                                rl_img,
                                Paragraph(
                                    f"Figure {fig_num}: Unmodified Original Evidence Photograph — {iid} "
                                    f"({im.get('original_filename')}, {im.get('width_px')}x{im.get('height_px')} px). "
                                    f"Preserved without alteration.",
                                    styles["Caption"],
                                ),
                            ]
                        )
                    )
            else:
                story.append(
                    Paragraph(
                        f"Figure {fig_num}: [Original image file for {iid} ({im.get('original_filename')}) is unavailable on disk; "
                        f"database metadata preserved.]",
                        styles["Caption"],
                    )
                )
    else:
        story.append(Paragraph("No photographic evidence images are associated with this examination scope.", styles["BodyLeft"]))


def _build_section_4_optical_methodology(
    story: List[Any],
    styles: Dict[str, ParagraphStyle],
    measurements: List[Dict[str, Any]],
    images_by_id: Dict[str, Dict[str, Any]],
    fig_counter: List[int],
) -> None:
    """Section 4 — Optical Measurement Methodology & Fitted Ellipse Overlays."""
    story.append(Paragraph("SECTION 4 — OPTICAL MEASUREMENT METHODOLOGY", styles["SectionHeading"]))
    story.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#cbd5e1"), spaceAfter=6))

    story.append(
        Paragraph(
            "Photographic bloodstain evidence is processed through a deterministic 8-stage computer vision and "
            "trigonometric pipeline: (1) 8-bit luminance grayscale conversion (cv2.cvtColor), (2) Gaussian spatial "
            "smoothing (cv2.GaussianBlur) to attenuate sensor grain, (3) Otsu global bimodal or Adaptive Gaussian "
            "thresholding, (4) morphological opening and closing (cv2.morphologyEx) to suppress satellite speckle "
            "while preserving primary stain perimeters, (5) external contour extraction (cv2.findContours), "
            "(6) Fitzgibbon direct least-squares ellipse fitting (cv2.fitEllipse, minimum 5 contour vertices), "
            "(7) two-point reference scale-bar calibration into physical millimeters (mm/px), and (8) impact angle "
            "derivation via established planar impact geometry:",
            styles["Body"],
        )
    )
    story.append(
        Paragraph(
            "Impact Angle Formula:   alpha = arcsin( W / L )     where 0 < Minor Axis (W) <= Major Axis (L)",
            styles["EquationBox"],
        )
    )

    # For measurements linked to an optical image, display optical telemetry and annotated overlay
    for m in measurements:
        iid = m.get("image_id")
        if not iid:
            continue
        im = images_by_id.get(iid)
        is_cal, cal_desc = trajectory_integration.is_measurement_calibrated(m)
        unit_str = "mm" if is_cal else "px (Uncalibrated)"

        opt_rows = [
            ["Stain Label / Measurement ID", f"{m.get('stain_label')}  (Measurement #{m['id']}, Image {iid})"],
            ["Minor Axis (Width W)", f"{m['width']:.2f} {unit_str}" + (f"  ({m['width_px']:.2f} px)" if m.get("width_px") else "")],
            ["Major Axis (Length L)", f"{m['length']:.2f} {unit_str}" + (f"  ({m['length_px']:.2f} px)" if m.get("length_px") else "")],
            ["Width-to-Length Ratio (W / L)", f"{m['ratio']:.4f}"],
            ["Calculated Impact Angle (alpha)", f"{m['angle_degrees']:.2f} deg  ({m['angle_radians']:.4f} rad)"],
            ["Fitted Ellipse Orientation", f"{m['ellipse_angle']:.2f} deg" if m.get("ellipse_angle") is not None else "N/A"],
            ["Ellipse Center (Image Pixels)", f"({m['ellipse_center_x']:.1f}, {m['ellipse_center_y']:.1f}) px" if m.get("ellipse_center_x") is not None else "N/A"],
            ["Calibration Scale & Status", f"{cal_desc}  (scale_factor = {m.get('scale_factor', 1.0):.4f})"],
        ]
        tbl = Table(opt_rows, colWidths=[55 * mm, 119 * mm])
        tbl.setStyle(_kv_table_style())
        story.append(tbl)
        story.append(Spacer(1, 3 * mm))

        # Check for processed overlay or generate on the fly from original image if available
        overlay_path = os.path.join(BASE_DIR, "uploads", "processed", f"{iid}_overlay.png")
        if not os.path.isfile(overlay_path) and im and im.get("filepath") and os.path.isfile(im["filepath"]):
            try:
                cv_res = image_processing.process_bloodstain_pipeline(
                    image_source=im["filepath"],
                    stain_label=m.get("stain_label") or "Stain",
                    direction_angle_deg=m.get("direction_angle"),
                )
                image_processing.save_processed_views(
                    pipeline_result=cv_res,
                    output_dir=os.path.join(BASE_DIR, "uploads", "processed"),
                    image_id=iid,
                )
            except Exception:
                pass

        fig_num = fig_counter[0]
        fig_counter[0] += 1
        if os.path.isfile(overlay_path):
            rl_ov = _create_proportional_rl_image(overlay_path, max_width_mm=115.0, max_height_mm=65.0)
            if rl_ov:
                story.append(
                    KeepTogether(
                        [
                            rl_ov,
                            Paragraph(
                                f"Figure {fig_num}: Processed Computer-Vision Annotated Overlay — {iid} "
                                f"(Fitted Least-Squares Ellipse, Minor/Major Axes, and Contour Boundary for {m.get('stain_label')}). "
                                f"Distinct from unmodified original evidence image.",
                                styles["Caption"],
                            ),
                        ]
                    )
                )


def _build_section_5_individual_measurements_table(
    story: List[Any],
    styles: Dict[str, ParagraphStyle],
    measurements: List[Dict[str, Any]],
) -> None:
    """Section 5 — Individual Measurement Results Table."""
    story.append(Paragraph("SECTION 5 — INDIVIDUAL MEASUREMENT RESULTS", styles["SectionHeading"]))
    story.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#cbd5e1"), spaceAfter=6))

    story.append(
        Paragraph(
            "The table below summarizes all bloodstain measurements within this examination scope. "
            "Calibrated physical millimeter measurements (mm) are explicitly distinguished from uncalibrated "
            "optical pixel measurements (px). First-order propagated impact angle uncertainty (±sigma_alpha) "
            "is computed via sigma_alpha = (1 / (L * sqrt(1 - (W/L)^2))) * sqrt(sigma_W^2 + (W/L)^2 * sigma_L^2).",
            styles["Body"],
        )
    )

    header = [
        "Stain ID",
        "Width (W)",
        "Length (L)",
        "Ratio (W/L)",
        "Impact Angle",
        "Uncertainty",
        "Source / Image ID",
        "Calibration Status",
    ]
    rows: List[List[Any]] = [header]

    for m in measurements:
        is_cal, cal_reason = trajectory_integration.is_measurement_calibrated(m)
        unit = "mm" if is_cal else "px"
        try:
            unc = trajectory.propagate_uncertainty(width=m["width"], length=m["length"])
            sigma_str = f"±{unc['sigma_angle_degrees']:.2f} deg"
        except Exception:
            sigma_str = "N/A (90 deg)"

        src_str = f"Optical ({m['image_id']})" if m.get("image_id") else "Manual Entry"
        cal_badge = "CALIBRATED (mm)" if is_cal else "UNCALIBRATED (px)"

        rows.append(
            [
                Paragraph(f"{m.get('stain_label', 'Stain')} (#{m['id']})", styles["TableCell"]),
                f"{m['width']:.2f} {unit}",
                f"{m['length']:.2f} {unit}",
                f"{m['ratio']:.4f}",
                f"{m['angle_degrees']:.2f} deg",
                sigma_str,
                Paragraph(src_str, styles["TableCell"]),
                Paragraph(f"<b>{cal_badge}</b><br/>{cal_reason}", styles["TableCell"]),
            ]
        )

    tbl = Table(
        rows,
        colWidths=[24 * mm, 18 * mm, 18 * mm, 17 * mm, 20 * mm, 20 * mm, 25 * mm, 32 * mm],
        repeatRows=1,
    )
    tbl.setStyle(_standard_table_style())
    story.append(tbl)
    story.append(Spacer(1, 4 * mm))


def _build_section_6_7_8_9_trajectory_sections(
    story: List[Any],
    styles: Dict[str, ParagraphStyle],
    reconstruction: Optional[Dict[str, Any]],
    rec_stains: List[Dict[str, Any]],
    rec_result: Optional[Dict[str, Any]],
    fig_counter: List[int],
) -> None:
    """
    Sections 6, 7, 8, and 9:
    - Section 6: Trajectory Reconstruction Methodology
    - Section 7: Area of Convergence (2D)
    - Section 8: Estimated Area of Origin (3D)
    - Section 9: Visual Reconstruction (2D & 3D Publication Diagrams)
    """
    # Section 6
    story.append(Paragraph("SECTION 6 — TRAJECTORY RECONSTRUCTION METHODOLOGY", styles["SectionHeading"]))
    story.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#cbd5e1"), spaceAfter=6))
    story.append(
        Paragraph(
            "Multi-stain trajectory reconstruction combines individual impact angles (alpha = arcsin(W/L)) with "
            "examiner-verified direction-of-travel angles (gamma in [0, 360) deg measured counter-clockwise from "
            "+X_surface) and calibrated receiving-surface coordinates (X_s, Y_s in mm). Because an elliptical "
            "outline alone carries a 180-degree directional ambiguity, reverse source bearing is derived from "
            "examiner-supplied tail/spine orientation as beta = (gamma + 180 deg) mod 360 deg.",
            styles["Body"],
        )
    )
    story.append(
        Paragraph(
            "1. 2D Orthogonal Least-Squares Convergence:  min_(X_c, Y_c) sum_i ( (p_i - c) . n_perp,i )^2\n"
            "2. Classical BPA Tangent Height Estimate:    Z_i = D_i * tan(alpha_i),   D_i = ||(X_i - X_c, Y_i - Y_c)||\n"
            "3. 3D Skew-Ray Closest Point of Approach:    min_P sum_i || (I - u_i u_i^T) (P - P_0,i) ||^2",
            styles["EquationBox"],
        )
    )

    if not reconstruction or not rec_result:
        story.append(
            Paragraph(
                "Sections 7–9 Notice: No multi-stain trajectory reconstruction calculation was executed for this "
                "individual measurement examination scope.",
                styles["BodyLeft"],
            )
        )
        return

    scene: Dict[str, Any] = {}
    if rec_result.get("result_json"):
        try:
            scene = json.loads(rec_result["result_json"])
        except Exception:
            scene = {}

    # Section 7 — Area of Convergence
    story.append(Paragraph("SECTION 7 — 2D AREA OF CONVERGENCE", styles["SectionHeading"]))
    story.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#cbd5e1"), spaceAfter=6))

    conv_rows = [
        ["Reconstruction ID & Name", f"{reconstruction['reconstruction_id']} — {reconstruction.get('reconstruction_name', '')}"],
        ["Receiving Surface Orientation", f"{rec_result['surface_type']}"],
        ["Included Calibrated Stains", f"{rec_result['stain_count']} stains"],
        ["Estimated Convergence X_c", f"{rec_result['convergence_x_mm']:.2f} mm"],
        ["Estimated Convergence Y_c", f"{rec_result['convergence_y_mm']:.2f} mm"],
        ["RMS Perpendicular Residual", f"{rec_result['rms_residual_mm']:.2f} mm"],
        ["Maximum Perpendicular Residual", f"{rec_result['max_residual_mm']:.2f} mm"],
        ["Convergence Uncertainty Radius", f"±{rec_result['convergence_radius_mm']:.2f} mm"],
        ["Geometric Condition Number", f"{rec_result.get('condition_number', 1.0):.2f}"],
    ]
    c_tbl = Table(conv_rows, colWidths=[58 * mm, 116 * mm])
    c_tbl.setStyle(_kv_table_style())
    story.append(c_tbl)
    story.append(Spacer(1, 4 * mm))

    if scene.get("stains"):
        story.append(Paragraph("7.1 Individual Stain Directionality, Surface Coordinates & Miss Distances", styles["SubHeading"]))
        s_rows = [
            ["Stain ID", "Surface Pos (X_s, Y_s)", "Impact Angle (α)", "Travel Dir (γ)", "Back-Ray (β)", "Planar Dist D_i", "2D Miss Residual"]
        ]
        for st in scene["stains"]:
            s_rows.append(
                [
                    st["stain_id"],
                    f"({st['surface_x_mm']:.1f}, {st['surface_y_mm']:.1f}) mm",
                    f"{st['angle_degrees']:.2f} deg",
                    f"{st['gamma_deg']:.1f} deg",
                    f"{st['reverse_azimuth_deg']:.1f} deg",
                    f"{st['planar_distance_mm']:.1f} mm",
                    f"{st['perp_residual_2d_mm']:.2f} mm",
                ]
            )
        st_tbl = Table(s_rows, colWidths=[22 * mm, 36 * mm, 24 * mm, 22 * mm, 22 * mm, 24 * mm, 24 * mm], repeatRows=1)
        st_tbl.setStyle(_standard_table_style())
        story.append(st_tbl)
        story.append(Spacer(1, 4 * mm))

    # Section 8 — Estimated Area of Origin
    story.append(Paragraph("SECTION 8 — ESTIMATED 3D AREA OF ORIGIN", styles["SectionHeading"]))
    story.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#cbd5e1"), spaceAfter=6))

    story.append(
        Paragraph(
            "<b>Forensic Interpretation Requirement:</b> The coordinates below represent an <i>approximate volumetric "
            "Area of Origin</i> under linear stringing geometry and must never be interpreted as an exact point source.",
            styles["NoticeBox"],
        )
    )

    orig_rows = [
        ["Method 1: Classical BPA Tangent Method", f"X = {rec_result['tangent_origin_x_mm']:.2f} mm,  Y = {rec_result['tangent_origin_y_mm']:.2f} mm"],
        ["Tangent Mean Height (Z_mean ± σ_Z)", f"{rec_result['tangent_origin_z_mm']:.2f} ± {rec_result['tangent_z_uncertainty_mm']:.2f} mm  (Std Dev: {rec_result['tangent_z_std_mm']:.2f} mm)"],
        ["Tangent Height Envelope [Z_min, Z_max]", f"[{rec_result['tangent_z_min_mm']:.2f}, {rec_result['tangent_z_max_mm']:.2f}] mm"],
        ["Method 2: 3D Skew-Ray CPA Origin", f"X_lab = {rec_result['estimated_origin_x_mm']:.2f} mm,  Y_lab = {rec_result['estimated_origin_y_mm']:.2f} mm,  Z_lab = {rec_result['estimated_origin_z_mm']:.2f} mm"],
        ["3D Skew-Ray RMS Miss Distance", f"{rec_result['rms_3d_miss_mm']:.2f} mm"],
    ]
    o_tbl = Table(orig_rows, colWidths=[68 * mm, 106 * mm])
    o_tbl.setStyle(_kv_table_style())
    story.append(o_tbl)
    story.append(Spacer(1, 4 * mm))

    # Section 9 — Visual Reconstruction
    story.append(Paragraph("SECTION 9 — VISUAL RECONSTRUCTION DIAGRAMS", styles["SectionHeading"]))
    story.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#cbd5e1"), spaceAfter=6))

    try:
        stain_inputs = [
            StainTrajectoryInput(
                stain_id=str(s["stain_identifier"]),
                width=float(s["width"]),
                length=float(s["length"]),
                surface_x_mm=float(s["surface_x_mm"]),
                surface_y_mm=float(s["surface_y_mm"]),
                gamma_deg=float(s["direction_angle"]),
                sigma_gamma_deg=2.0,
                is_calibrated=True,
            )
            for s in rec_stains
            if s.get("direction_angle") is not None
            and s.get("surface_x_mm") is not None
            and s.get("surface_y_mm") is not None
        ]
        if len(stain_inputs) >= 2:
            vis_2d = trajectory_visualization.render_2d_convergence_plot(
                stains=stain_inputs,
                surface_type=rec_result["surface_type"],
            )
            vis_3d = trajectory_visualization.render_3d_trajectory_plot(
                stains=stain_inputs,
                surface_type=rec_result["surface_type"],
                incline_deg=reconstruction.get("incline_deg"),
            )
            png_2d = _decode_data_uri_png(vis_2d["base64_png"])
            png_3d = _decode_data_uri_png(vis_3d["base64_png"])

            if png_2d:
                rl_2d = _create_proportional_rl_image(png_2d, max_width_mm=150.0, max_height_mm=82.0)
                if rl_2d:
                    f2d = fig_counter[0]
                    fig_counter[0] += 1
                    story.append(
                        KeepTogether(
                            [
                                rl_2d,
                                Paragraph(
                                    f"Figure {f2d}: 2D Area of Convergence Diagram ({rec_result['surface_type']}) — "
                                    f"Showing stain positions (mm), reverse trajectory rays, perpendicular miss residuals, "
                                    f"and uncertainty radius (±{rec_result['convergence_radius_mm']:.2f} mm).",
                                    styles["Caption"],
                                ),
                            ]
                        )
                    )
            if png_3d:
                rl_3d = _create_proportional_rl_image(png_3d, max_width_mm=150.0, max_height_mm=85.0)
                if rl_3d:
                    f3d = fig_counter[0]
                    fig_counter[0] += 1
                    story.append(
                        KeepTogether(
                            [
                                rl_3d,
                                Paragraph(
                                    f"Figure {f3d}: 3D Trajectory Reconstruction Scene ({rec_result['surface_type']}) — "
                                    f"Showing 3D laboratory axes (mm), receiving surface plane, back-projected rays, "
                                    f"Tangent Origin, and 3D Closest-Point-of-Approach (CPA) region.",
                                    styles["Caption"],
                                ),
                            ]
                        )
                    )
    except Exception as exc:
        story.append(Paragraph(f"Visual diagram rendering note: {exc}", styles["BodyLeft"]))


def _build_section_pattern_analysis(
    story: List[Any],
    styles: Dict[str, ParagraphStyle],
    pattern_analyses: Optional[List[Dict[str, Any]]],
) -> None:
    """
    Phase 5.1 — Bloodstain Pattern Classification & Analytical Decision Support Section.
    Renders when pattern_analyses records are associated with the case or report scope.
    Clearly distinguishes Automatically Measured features, Software Suggested candidates,
    Examiner Observed parameters, and Examiner Approved conclusions.
    """
    if not pattern_analyses:
        return

    story.append(
        Paragraph(
            "SECTION 9B — BLOODSTAIN PATTERN CLASSIFICATION & ANALYTICAL DECISION SUPPORT",
            styles["SectionHeading"],
        )
    )
    story.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#cbd5e1"), spaceAfter=6))

    story.append(
        Paragraph(
            "<b>Pattern Analysis Methodology:</b> Quantitative contour morphology (Circularity = 4*pi*A/P^2, "
            "Aspect Ratio = L/W, Solidity = A/A_hull, Eccentricity = sqrt(1 - (W/L)^2), spatial density, "
            "nearest-neighbor dispersion, and centroid linearity R^2) is extracted deterministically from "
            "segmented evidence imagery and evaluated alongside structured examiner observations. "
            "<b>Software suggestions are preliminary analytical hypotheses only and never establish a "
            "definitive deposition mechanism without examiner verification.</b>",
            styles["Body"],
        )
    )

    for pa in pattern_analyses:
        try:
            feats = json.loads(pa.get("features_json") or "{}")
            suggs = json.loads(pa.get("suggestions_json") or "{}")
            obs = json.loads(pa.get("observations_json") or "{}")
        except Exception:
            continue

        summary = feats.get("pattern_summary", {})
        is_cal = bool(pa.get("is_calibrated"))
        unit_lbl = f"Calibrated Metric ({pa.get('mm_per_pixel'):.4f} mm/px)" if is_cal and pa.get("mm_per_pixel") else "Uncalibrated (Pixel Units Only)"

        meta_rows = [
            ["Pattern Analysis ID & Revision", f"{pa.get('analysis_id')}  (Revision R{pa.get('revision_number', 1)})"],
            ["Source Evidence Image ID", f"{pa.get('image_id') or 'N/A'}  ({pa.get('image_filename') or 'No filename'})"],
            ["Calibration & Unit Discipline", f"[Examiner Supplied] {unit_lbl}"],
            ["Detected Contours (Accepted / Flagged)", f"[Automatically Measured] {feats.get('accepted_stain_count', 0)} accepted stains / {feats.get('rejected_or_uncertain_count', 0)} rejected or uncertain contours"],
            ["Spatial Distribution & Linearity", f"[Automatically Measured] {summary.get('spatial_distribution_type', 'N/A')}  (Linearity R^2 = {summary.get('centroid_linearity_r2') if summary.get('centroid_linearity_r2') is not None else 'N/A'})"],
            [
                "Mean Morphological Ratios",
                f"[Automatically Measured] Circularity = {summary.get('mean_circularity') or 'N/A'}, "
                f"Aspect Ratio = {summary.get('mean_aspect_ratio') or 'N/A'}, "
                f"Solidity = {summary.get('mean_solidity') or 'N/A'}, "
                f"Eccentricity = {summary.get('mean_eccentricity') or 'N/A'}",
            ],
            ["Software Preliminary Determination", f"[Software Suggested] {pa.get('overall_determination')}"],
            ["Examiner Suggestion Decision", f"[Examiner Observed] {pa.get('suggestion_decision') or 'Pending Examiner Review'}"],
            ["Examiner Review Status", f"[{pa.get('review_status') or 'Preliminary — Examiner Review Required'}]"],
            [
                "Final Examiner Interpretation",
                Paragraph(
                    f"<b>[Examiner Approved / Recorded]:</b> {pa.get('examiner_interpretation') or 'Not yet recorded — Pending examiner sign-off.'}",
                    styles["TableCell"],
                ),
            ],
            [
                "Examiner Decision Rationale & Notes",
                Paragraph(
                    f"<b>[Examiner Observed]:</b> {pa.get('examiner_rationale') or 'No rationale recorded.'} | "
                    f"Substrate: {obs.get('surface_material', {}).get('value', 'N/A')} ({obs.get('surface_texture', {}).get('value', 'N/A')}) | "
                    f"Notes: {obs.get('examination_notes', {}).get('value', 'None')}",
                    styles["TableCell"],
                ),
            ],
        ]
        pa_tbl = Table(meta_rows, colWidths=[58 * mm, 116 * mm])
        pa_tbl.setStyle(_kv_table_style())
        story.append(pa_tbl)
        story.append(Spacer(1, 3 * mm))

        cand_list = suggs.get("candidate_suggestions", [])
        if cand_list:
            story.append(
                Paragraph(
                    f"<b>Candidate Pattern Interpretations for {pa.get('analysis_id')} ([Software Suggested] — Require Examiner Review):</b>",
                    styles["SubHeading"],
                )
            )
            c_rows: List[List[Any]] = [
                ["Candidate Category", "Evidence Indicator", "Supporting Characteristics", "Contradicting / Limiting Factors"]
            ]
            for c in cand_list[:5]:
                supp_txt = "; ".join(c.get("supporting_features", [])) or "None"
                contra_txt = "; ".join(c.get("contradicting_features", []) + [c.get("distinguishing_limitations", "")])
                c_rows.append(
                    [
                        Paragraph(f"<b>{c.get('category_name')}</b> ({c.get('group_name')})", styles["TableCell"]),
                        Paragraph(str(c.get("evidence_indicator", "")), styles["TableCell"]),
                        Paragraph(supp_txt, styles["TableCell"]),
                        Paragraph(contra_txt, styles["TableCell"]),
                    ]
                )
            c_tbl = Table(c_rows, colWidths=[36 * mm, 34 * mm, 52 * mm, 52 * mm], repeatRows=1)
            c_tbl.setStyle(_standard_table_style())
            story.append(c_tbl)
            story.append(Spacer(1, 3 * mm))

    story.append(
        Paragraph(
            "<b>Classification Limitations Notice:</b> Bloodstain morphology and spatial distribution alone "
            "cannot establish impact velocity, applied force, specific weapon type, or exact chronological "
            "sequence of events. Morphological overlap exists between impact spatter, expirated blood, "
            "and satellite spatter, as well as between linear cast-off and drip trails.",
            styles["NoticeBox"],
        )
    )


def _build_section_overlap_analysis(
    story: List[Any],
    styles: Dict[str, ParagraphStyle],
    overlap_analyses: Optional[List[Dict[str, Any]]],
) -> None:
    """
    Section 9C: Overlapping & Complex Bloodstain Analysis (Phase 5.2).
    Documents connected component detection, marker-controlled watershed candidate separations,
    examiner review decisions (accepted, rejected, indeterminate), and area conservation.
    """
    if not overlap_analyses:
        return

    story.append(Spacer(1, 4 * mm))
    story.append(
        Paragraph(
            "SECTION 9C — OVERLAPPING & COMPLEX BLOODSTAIN PATTERN ANALYSIS",
            styles["SectionHeading"],
        )
    )
    story.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#cbd5e1"), spaceAfter=6))

    story.append(
        Paragraph(
            "<b>Methodological Purpose & Separation Governance:</b> Topological distance-transform and "
            "marker-controlled watershed algorithms identify connected stain regions and propose candidate "
            "subdivisions. In accordance with forensic standards, merged contours are never automatically declared "
            "individual stains without visual corroboration; uncertain regions remain preserved as unresolved merged entities.",
            styles["BodyLeft"],
        )
    )
    story.append(Spacer(1, 3 * mm))

    for oa in overlap_analyses:
        regions_data: List[Dict[str, Any]] = []
        try:
            regions_data = json.loads(oa.get("regions_json", "[]"))
        except Exception:
            regions_data = []

        is_cal = bool(oa.get("is_calibrated"))
        unit_lbl = f"Calibrated Metric ({oa.get('mm_per_pixel'):.4f} mm/px)" if is_cal and oa.get("mm_per_pixel") else "Uncalibrated (Pixel Units Only)"

        meta_rows = [
            ["Overlap Analysis ID & Revision", f"{oa.get('analysis_id')}  (Revision R{oa.get('revision_number', 1)})"],
            ["Source Evidence Image ID", f"{oa.get('image_id') or 'N/A'}  ({oa.get('image_filename') or 'No filename'})"],
            ["Algorithm & Version", f"{oa.get('algorithm_name', 'Marker Watershed')} {oa.get('algorithm_version', 'v5.2')}"],
            ["Calibration & Unit Discipline", f"[Examiner Supplied] {unit_lbl}"],
            ["Total Connected Regions Detected", f"[Automatically Measured] {len(regions_data)} regions"],
            ["Review Status & Examiner", f"[{oa.get('review_status', 'Preliminary')}] — {oa.get('examiner_name') or 'Duty Examiner'}"],
            ["Examiner Examination Notes", f"{oa.get('examiner_notes') or 'No specific notes recorded.'}"],
        ]
        meta_tbl = Table(meta_rows, colWidths=[58 * mm, 116 * mm])
        meta_tbl.setStyle(_kv_table_style())
        story.append(meta_tbl)
        story.append(Spacer(1, 3 * mm))

        if regions_data:
            story.append(
                Paragraph(
                    f"<b>Connected Region Classifications & Segmentation Candidates ({oa.get('analysis_id')}):</b>",
                    styles["SubHeading"],
                )
            )
            r_rows: List[List[Any]] = [
                ["Region ID", "Classification", "Area", "Aspect Ratio", "Solidity", "Peaks", "Status", "Decision"]
            ]
            for r in regions_data[:12]:
                area_str = f"{r.get('parent_area_mm2')} mm²" if is_cal and r.get("parent_area_mm2") is not None else f"{r.get('parent_area_px2')} px²"
                r_rows.append(
                    [
                        r.get("region_id", "REG"),
                        r.get("region_type", "Unknown"),
                        area_str,
                        f"{r.get('parent_aspect_ratio', 1.0):.2f}",
                        f"{r.get('parent_solidity', 1.0):.2f}",
                        str(r.get("peak_count", 1)),
                        r.get("segmentation_status", "Unresolved"),
                        r.get("examiner_decision", "Pending"),
                    ]
                )
            r_tbl = Table(r_rows, colWidths=[20 * mm, 34 * mm, 26 * mm, 20 * mm, 16 * mm, 14 * mm, 26 * mm, 18 * mm], repeatRows=1)
            r_tbl.setStyle(_standard_table_style())
            story.append(r_tbl)
            story.append(Spacer(1, 3 * mm))

    story.append(
        Paragraph(
            "<b>Overlapping Analysis Limitations Notice:</b> 2D photograph segmentation cannot establish droplet "
            "temporal deposition sequence or 3D physical vertical layering without independent microscopic or biological "
            "indicators. Merged stain regions with unresolved boundaries must not be used for individual trajectory reconstruction.",
            styles["NoticeBox"],
        )
    )


def _build_section_10_11_12_closing(
    story: List[Any],
    styles: Dict[str, ParagraphStyle],
    meta: Dict[str, Any],
    measurements: List[Dict[str, Any]],
    reconstruction: Optional[Dict[str, Any]],
    rec_result: Optional[Dict[str, Any]],
    image_hashes: Dict[str, str],
) -> None:
    """
    Sections 10, 11, and 12:
    - Section 10: Context-Aware Scientific Limitations
    - Section 11: Neutral Factual Conclusion & Examiner Observations
    - Section 12: Appendices, Uncertainty Parameters & Cryptographic Integrity Notice
    """
    # Section 10 — Scientific Limitations
    story.append(Paragraph("SECTION 10 — SCIENTIFIC LIMITATIONS", styles["SectionHeading"]))
    story.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#cbd5e1"), spaceAfter=6))

    has_uncal = any(not trajectory_integration.is_measurement_calibrated(m)[0] for m in measurements)
    has_optical = any(bool(m.get("image_id")) for m in measurements)

    limitations = [
        "1. Elliptical Impact Geometry & Surface Planarity: The relation alpha = arcsin(W/L) assumes a single droplet "
        "striking a smooth, non-absorbent planar target surface without post-impact distortion or dripping.",
    ]
    if has_optical:
        limitations.append(
            "2. Optical Resolution, Thresholding & Contour Selection: Digital pixel discretization, illumination gradients, "
            "and satellite tail/spine protrusions influence contour segmentation and least-squares ellipse fitting (cv2.fitEllipse)."
        )
    if has_uncal:
        limitations.append(
            "3. Uncalibrated Pixel Measurements Present: One or more measurements in this scope were recorded in pixel units "
            "without a physical millimeter scale factor. Uncalibrated measurements are valid for dimensionless ratio (W/L) "
            "and impact angle (alpha) only and cannot be used for spatial distance or trajectory convergence."
        )
    else:
        limitations.append(
            "3. Spatial Scale Calibration Uncertainty: Physical millimeter dimensions depend on two-point reference scale-bar "
            "placement coplanar with the bloodstain receiving surface and free of perspective foreshortening."
        )

    if reconstruction and rec_result:
        limitations.extend(
            [
                "4. Directionality Ambiguity (180-Degree Ellipse Symmetry): An elliptical stain boundary alone does not uniquely "
                "establish travel direction. The 2D/3D convergence solution depends directly on examiner-supplied direction "
                "angles (gamma in [0, 360) deg) determined from physical tail/scalloping morphology.",
                "5. Linear Stringing Model (Neglect of Gravity & Aerodynamic Drag): Straight-line back-projection models "
                "droplet flight paths as linear rays rather than parabolic ballistic trajectories. Because airborne droplets "
                "experience downward gravitational acceleration and air drag, linear stringing systematically overestimates "
                "the true vertical height (Z) of the blood source.",
                "6. Volumetric Area of Origin vs. Point Source: Numerical ray residuals and angular uncertainties (±sigma_alpha, "
                "±sigma_gamma) mean that calculated intersections define an approximate 3D region of origin, never an exact point.",
            ]
        )

    for lim in limitations:
        story.append(Paragraph(lim, styles["BodyLeft"]))

    # Section 11 — Conclusion
    story.append(Paragraph("SECTION 11 — FACTUAL SUMMARY & EXAMINER OBSERVATIONS", styles["SectionHeading"]))
    story.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#cbd5e1"), spaceAfter=6))

    summary_lines = [
        f"This report documents {len(measurements)} bloodstain measurement(s) under Report ID {meta['report_id']} "
        f"(Revision R{meta['revision_number']}, Status: {meta['report_status']})."
    ]
    if measurements:
        angles = [float(m["angle_degrees"]) for m in measurements]
        summary_lines.append(
            f"Measured impact angles (alpha) across the analyzed stain set range from {min(angles):.2f} deg to {max(angles):.2f} deg."
        )
    if reconstruction and rec_result:
        summary_lines.append(
            f"Multi-stain trajectory reconstruction ({reconstruction['reconstruction_id']} on {rec_result['surface_type']}, "
            f"N={rec_result['stain_count']} calibrated stains) yielded an estimated 2D Area of Convergence at "
            f"(X_c = {rec_result['convergence_x_mm']:.2f} mm, Y_c = {rec_result['convergence_y_mm']:.2f} mm; "
            f"RMS residual = {rec_result['rms_residual_mm']:.2f} mm) and an approximate 3D Area of Origin height "
            f"of Z_mean = {rec_result['tangent_origin_z_mm']:.2f} ± {rec_result['tangent_z_uncertainty_mm']:.2f} mm "
            f"(3D CPA Lab Coordinates: [{rec_result['estimated_origin_x_mm']:.2f}, {rec_result['estimated_origin_y_mm']:.2f}, "
            f"{rec_result['estimated_origin_z_mm']:.2f}] mm; 3D RMS miss = {rec_result['rms_3d_miss_mm']:.2f} mm)."
        )
    summary_lines.append(
        "Scope Restriction: This technical report states mathematical and geometric findings only and makes no automated "
        "claims regarding the identity of any individual, sequence of events, intent, cause of death, or criminality."
    )
    story.append(Paragraph(" ".join(summary_lines), styles["Body"]))

    obs_text = (
        meta.get("examiner_observations")
        if meta.get("examiner_observations")
        else "No additional narrative observations were entered by the examiner at the time of report generation."
    )
    story.append(Paragraph("<b>Examiner Supplied Observations:</b>", styles["SubHeading"]))
    story.append(Paragraph(obs_text, styles["BodyLeft"]))

    # Section 12 — Appendices
    story.append(Paragraph("SECTION 12 — APPENDICES & CRYPTOGRAPHIC TRACEABILITY", styles["SectionHeading"]))
    story.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#cbd5e1"), spaceAfter=6))

    src_mids = ", ".join(str(m["id"]) for m in measurements) or "None"
    src_iids = ", ".join(sorted(image_hashes.keys())) or "None"

    app_rows = [
        ["Software Engine & Version", SOFTWARE_VERSION],
        ["Report ID & Revision", f"{meta['report_id']}  (Revision R{meta['revision_number']})"],
        ["Source Measurement Record IDs", src_mids],
        ["Source Evidence Image IDs", src_iids],
        ["Default Uncertainty Model", "First-order Taylor propagation (sigma_W = 0.15 mm, sigma_L = 0.15 mm, sigma_gamma = 2.0 deg)"],
        [
            "Cryptographic Hash Scope Notice",
            Paragraph(
                "The SHA-256 hash stored in the laboratory database verifies the binary integrity of the generated PDF file "
                "and unmodified source evidence photographs; it does not certify the scientific validity of examiner-supplied inputs.",
                styles["TableCell"],
            ),
        ],
    ]
    app_tbl = Table(app_rows, colWidths=[58 * mm, 116 * mm])
    app_tbl.setStyle(_kv_table_style())
    story.append(app_tbl)


# ===========================================================================
# 5. Core Atomic PDF Document Compiler
# ===========================================================================


def _compile_pdf_document(
    output_dir: str,
    meta: Dict[str, Any],
    case: Optional[Dict[str, Any]],
    evidence_items: List[Dict[str, Any]],
    images: List[Dict[str, Any]],
    measurements: List[Dict[str, Any]],
    reconstruction: Optional[Dict[str, Any]],
    rec_stains: List[Dict[str, Any]],
    rec_result: Optional[Dict[str, Any]],
    pattern_analyses: Optional[List[Dict[str, Any]]] = None,
    overlap_analyses: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Build the 12-section PDF document into a temporary file, validate the PDF header/EOF,
    compute the SHA-256 digest, atomically move to the final path, and register in SQLite.
    """
    os.makedirs(output_dir, exist_ok=True)

    safe_report_id = secure_filename(meta["report_id"])
    if not safe_report_id:
        raise ReportGenerationError("Invalid report identifier generated.")

    filename = f"{safe_report_id}_{meta['report_status']}.pdf"
    final_filepath = os.path.join(output_dir, filename)
    temp_filepath = os.path.join(output_dir, f".tmp_{filename}")

    # Never overwrite an existing FINAL report file on disk
    if os.path.exists(final_filepath):
        raise ReportGenerationError(
            f"Report file '{filename}' already exists on disk. Existing reports are never overwritten."
        )

    # Compute SHA-256 hashes of original evidence images
    image_hashes: Dict[str, str] = {}
    images_by_id: Dict[str, Dict[str, Any]] = {}
    for im in images:
        iid = im["image_id"]
        images_by_id[iid] = im
        digest = compute_file_sha256(im.get("filepath", ""))
        image_hashes[iid] = digest if digest else "MISSING_ON_DISK"

    styles = _build_forensic_styles()
    story: List[Any] = []
    fig_counter = [1]

    # Assemble all forensic sections
    _build_section_1_cover(story, styles, meta, case)
    _build_section_2_case_info(story, styles, case, evidence_items, images)
    _build_section_3_evidence_documentation(story, styles, evidence_items, images, image_hashes, fig_counter)
    _build_section_4_optical_methodology(story, styles, measurements, images_by_id, fig_counter)
    _build_section_5_individual_measurements_table(story, styles, measurements)
    _build_section_6_7_8_9_trajectory_sections(story, styles, reconstruction, rec_stains, rec_result, fig_counter)
    _build_section_pattern_analysis(story, styles, pattern_analyses)
    _build_section_overlap_analysis(story, styles, overlap_analyses)
    _build_section_10_11_12_closing(story, styles, meta, measurements, reconstruction, rec_result, image_hashes)

    try:
        doc = SimpleDocTemplate(
            temp_filepath,
            pagesize=A4,
            leftMargin=18 * mm,
            rightMargin=18 * mm,
            topMargin=18 * mm,
            bottomMargin=18 * mm,
            title=meta["report_title"],
            author=meta.get("examiner_name") or "Forensic Laboratory Suite",
            subject=f"Forensic Bloodstain Pattern Report — {meta['report_id']}",
            creator=SOFTWARE_VERSION,
        )
        doc.build(
            story,
            canvasmaker=lambda *args, **kwargs: NumberedForensicCanvas(*args, report_meta=meta, **kwargs),
        )

        # Verify that the generated file is a valid non-empty PDF before promoting
        if not os.path.isfile(temp_filepath):
            raise ReportGenerationError("PDF builder did not produce an output file.")

        with open(temp_filepath, "rb") as f:
            pdf_bytes = f.read()

        if len(pdf_bytes) < 512 or not pdf_bytes.startswith(b"%PDF-") or b"%%EOF" not in pdf_bytes[-1024:]:
            raise ReportGenerationError("Generated file failed PDF header/trailer integrity verification.")

        pdf_sha256 = compute_bytes_sha256(pdf_bytes)
        os.replace(temp_filepath, final_filepath)

    except Exception as exc:
        if os.path.exists(temp_filepath):
            try:
                os.remove(temp_filepath)
            except OSError:
                pass
        raise ReportGenerationError(f"Failed to generate forensic PDF report: {exc}") from exc

    saved_record = database.create_report_record(
        report_id=meta["report_id"],
        report_type=meta["report_type"],
        report_title=meta["report_title"],
        filename=filename,
        filepath=final_filepath,
        file_size=len(pdf_bytes),
        sha256_hash=pdf_sha256,
        report_status=meta["report_status"],
        revision_number=meta["revision_number"],
        case_id=case["case_id"] if case else None,
        reconstruction_id=reconstruction["reconstruction_id"] if reconstruction else None,
        measurement_id=meta.get("measurement_id"),
        examiner_name=meta.get("examiner_name"),
        laboratory_name=meta.get("laboratory_name"),
        examiner_observations=meta.get("examiner_observations"),
        source_measurement_ids=json.dumps([m["id"] for m in measurements]),
        source_image_ids=json.dumps(sorted(image_hashes.keys())),
        image_hashes_json=json.dumps(image_hashes),
    )
    return saved_record


# ===========================================================================
# 6. Public Report Generation Entry Points (Types A, B, and C)
# ===========================================================================


def _normalize_report_status(status: Optional[str]) -> str:
    """Validate and normalize report_status to strictly 'DRAFT' or 'FINAL'."""
    if not status or not str(status).strip():
        return "DRAFT"
    cleaned = str(status).strip().upper()
    if cleaned not in ("DRAFT", "FINAL"):
        raise ReportGenerationError(f"Invalid report status '{status}'. Must be 'DRAFT' or 'FINAL'.")
    return cleaned


def generate_measurement_report(
    measurement_id: Any,
    output_dir: str = DEFAULT_REPORTS_DIR,
    report_status: str = "DRAFT",
    examiner_name: Optional[str] = None,
    laboratory_name: Optional[str] = None,
    examiner_observations: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Type A — Generate an Individual Bloodstain Examination Report for a single measurement_id.
    """
    try:
        mid = int(measurement_id)
    except (ValueError, TypeError):
        raise ReportGenerationError(f"Invalid measurement_id: '{measurement_id}'.")

    meas = database.get_measurement_by_id(mid)
    if not meas:
        raise ReportGenerationError(f"Measurement ID {mid} was not found in the database.")

    status_norm = _normalize_report_status(report_status)
    case = database.get_case_by_id(meas["case_id"]) if meas.get("case_id") else None

    evidence_items: List[Dict[str, Any]] = []
    if meas.get("evidence_id") and case:
        all_ev = database.get_evidence_for_case(case["case_id"])
        evidence_items = [e for e in all_ev if e["evidence_id"] == meas["evidence_id"]]

    images: List[Dict[str, Any]] = []
    if meas.get("image_id"):
        im = database.get_image_by_id(meas["image_id"])
        if im:
            images.append(im)

    report_type = "Individual Measurement"
    rev = database.get_next_report_revision(
        report_type=report_type,
        case_id=case["case_id"] if case else None,
        measurement_id=mid,
    )
    report_id = database.generate_report_id(revision_number=rev)

    meta = {
        "report_id": report_id,
        "report_type": report_type,
        "report_title": f"Individual Bloodstain Examination Report — {meas.get('stain_label', f'Stain #{mid}')}",
        "report_status": status_norm,
        "revision_number": rev,
        "case_id": case["case_id"] if case else None,
        "measurement_id": mid,
        "reconstruction_id": None,
        "examiner_name": examiner_name.strip() if examiner_name else None,
        "laboratory_name": laboratory_name.strip() if laboratory_name else None,
        "examiner_observations": examiner_observations.strip() if examiner_observations else None,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }

    return _compile_pdf_document(
        output_dir=output_dir,
        meta=meta,
        case=case,
        evidence_items=evidence_items,
        images=images,
        measurements=[meas],
        reconstruction=None,
        rec_stains=[],
        rec_result=None,
    )


def generate_reconstruction_report(
    reconstruction_id: str,
    output_dir: str = DEFAULT_REPORTS_DIR,
    report_status: str = "DRAFT",
    examiner_name: Optional[str] = None,
    laboratory_name: Optional[str] = None,
    examiner_observations: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Type B — Generate a Multiple Bloodstain Trajectory Reconstruction Report for a reconstruction_id.
    Automatically executes calculation if not yet calculated and stains are eligible.
    """
    if not reconstruction_id or not str(reconstruction_id).strip():
        raise ReportGenerationError("A valid reconstruction_id is required.")

    rec = database.get_reconstruction_by_id(reconstruction_id.strip())
    if not rec:
        raise ReportGenerationError(f"Reconstruction '{reconstruction_id}' was not found in the database.")

    status_norm = _normalize_report_status(report_status)
    case = database.get_case_by_id(rec["case_id"])
    if not case:
        raise ReportGenerationError(f"Associated Case '{rec['case_id']}' was not found.")

    rec_stains = database.get_stains_for_reconstruction(rec["reconstruction_id"])
    if len(rec_stains) < 2:
        raise ReportGenerationError(
            f"Reconstruction '{rec['reconstruction_id']}' contains fewer than 2 stains and cannot be reported."
        )

    rec_result = database.get_latest_reconstruction_result(rec["reconstruction_id"])
    if not rec_result:
        try:
            calc_out = trajectory_integration.execute_case_reconstruction(
                case_id=case["case_id"],
                reconstruction_id=rec["reconstruction_id"],
            )
            rec = calc_out["reconstruction"]
            rec_result = calc_out["result"]
        except Exception as exc:
            raise ReportGenerationError(
                f"Reconstruction '{rec['reconstruction_id']}' could not be calculated: {exc}"
            ) from exc

    # Gather only the measurements, evidence items, and images belonging to this reconstruction
    measurements: List[Dict[str, Any]] = []
    img_ids = set()
    ev_ids = set()
    for rs in rec_stains:
        m = database.get_measurement_by_id(rs["measurement_id"])
        if m:
            measurements.append(m)
            if m.get("image_id"):
                img_ids.add(m["image_id"])
            if m.get("evidence_id"):
                ev_ids.add(m["evidence_id"])

    all_case_ev = database.get_evidence_for_case(case["case_id"])
    evidence_items = [e for e in all_case_ev if e["evidence_id"] in ev_ids]

    images: List[Dict[str, Any]] = []
    for iid in sorted(img_ids):
        im = database.get_image_by_id(iid)
        if im:
            images.append(im)

    report_type = "Trajectory Reconstruction"
    rev = database.get_next_report_revision(
        report_type=report_type,
        case_id=case["case_id"],
        reconstruction_id=rec["reconstruction_id"],
    )
    report_id = database.generate_report_id(revision_number=rev)

    meta = {
        "report_id": report_id,
        "report_type": report_type,
        "report_title": f"Bloodstain Trajectory Reconstruction Report — {rec['reconstruction_id']}",
        "report_status": status_norm,
        "revision_number": rev,
        "case_id": case["case_id"],
        "measurement_id": None,
        "reconstruction_id": rec["reconstruction_id"],
        "examiner_name": examiner_name.strip() if examiner_name else None,
        "laboratory_name": laboratory_name.strip() if laboratory_name else None,
        "examiner_observations": examiner_observations.strip() if examiner_observations else None,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }

    return _compile_pdf_document(
        output_dir=output_dir,
        meta=meta,
        case=case,
        evidence_items=evidence_items,
        images=images,
        measurements=measurements,
        reconstruction=rec,
        rec_stains=rec_stains,
        rec_result=rec_result,
    )


def generate_case_report(
    case_id: str,
    output_dir: str = DEFAULT_REPORTS_DIR,
    report_status: str = "DRAFT",
    examiner_name: Optional[str] = None,
    laboratory_name: Optional[str] = None,
    examiner_observations: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Type C — Generate a Complete Forensic Case Report combining all case evidence,
    optical/manual stain measurements, and trajectory reconstruction results for case_id.
    Strictly excludes any unrelated cases or evidence.
    """
    if not case_id or not str(case_id).strip():
        raise ReportGenerationError("A valid case_id is required.")

    case = database.get_case_by_id(case_id.strip())
    if not case:
        raise ReportGenerationError(f"Forensic Case '{case_id}' was not found in the database.")

    status_norm = _normalize_report_status(report_status)
    evidence_items = database.get_evidence_for_case(case["case_id"])
    images = database.get_images_for_case(case["case_id"])
    measurements = database.get_measurements_for_case(case["case_id"])

    reconstructions = database.get_reconstructions_for_case(case["case_id"])
    active_rec: Optional[Dict[str, Any]] = None
    rec_stains: List[Dict[str, Any]] = []
    rec_result: Optional[Dict[str, Any]] = None

    if reconstructions:
        active_rec = reconstructions[0]
        rec_stains = database.get_stains_for_reconstruction(active_rec["reconstruction_id"])
        rec_result = database.get_latest_reconstruction_result(active_rec["reconstruction_id"])

    report_type = "Complete Case Report"
    rev = database.get_next_report_revision(
        report_type=report_type,
        case_id=case["case_id"],
        reconstruction_id=active_rec["reconstruction_id"] if active_rec else None,
    )
    report_id = database.generate_report_id(revision_number=rev)

    meta = {
        "report_id": report_id,
        "report_type": report_type,
        "report_title": f"Complete Forensic Case Examination Report — {case['case_id']}",
        "report_status": status_norm,
        "revision_number": rev,
        "case_id": case["case_id"],
        "measurement_id": None,
        "reconstruction_id": active_rec["reconstruction_id"] if active_rec else None,
        "examiner_name": examiner_name.strip() if examiner_name else None,
        "laboratory_name": laboratory_name.strip() if laboratory_name else None,
        "examiner_observations": examiner_observations.strip() if examiner_observations else None,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }

    pattern_analyses = database.get_pattern_analyses_for_case(case["case_id"])
    overlap_analyses = database.get_overlap_analyses_for_case(case["case_id"])

    return _compile_pdf_document(
        output_dir=output_dir,
        meta=meta,
        case=case,
        evidence_items=evidence_items,
        images=images,
        measurements=measurements,
        reconstruction=active_rec,
        rec_stains=rec_stains,
        rec_result=rec_result,
        pattern_analyses=pattern_analyses,
        overlap_analyses=overlap_analyses,
    )
