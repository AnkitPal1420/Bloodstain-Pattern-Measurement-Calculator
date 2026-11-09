"""
database.py
===========
SQLite data layer and persistence models for the Forensic Case Management system.

This module provides data access for cases, evidence items, and stain measurement records.
Uses Python's standard library sqlite3 for robust, dependency-free persistence.
"""

import os
import json
import random
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

DB_PATH = os.path.join(os.path.dirname(__file__), "database", "forensic.db")


def get_db_connection() -> sqlite3.Connection:
    """Return an active connection to the SQLite forensic database with Row factory."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    return conn


def init_db(seed_demo: bool = False) -> None:
    """Initialize database tables and indexes."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Forensic Cases table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id TEXT UNIQUE NOT NULL,
            title TEXT NOT NULL,
            reference_no TEXT NOT NULL,
            examination_date TEXT NOT NULL,
            examiner TEXT NOT NULL,
            description TEXT,
            status TEXT NOT NULL DEFAULT 'Active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )

    # 2. Evidence Records table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS evidence (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            evidence_id TEXT UNIQUE NOT NULL,
            case_id TEXT NOT NULL,
            description TEXT NOT NULL,
            evidence_type TEXT NOT NULL,
            location_surface TEXT,
            status TEXT NOT NULL DEFAULT 'Logged',
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (case_id) REFERENCES cases (case_id) ON DELETE CASCADE
        );
        """
    )

    # 3. Stain Measurements table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS measurements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id TEXT,
            evidence_id TEXT,
            stain_label TEXT DEFAULT 'Stain-01',
            width REAL NOT NULL,
            length REAL NOT NULL,
            ratio REAL NOT NULL,
            angle_degrees REAL NOT NULL,
            angle_radians REAL NOT NULL,
            is_perpendicular INTEGER DEFAULT 0,
            interpretation TEXT,
            calibration_status TEXT DEFAULT 'Uncalibrated (Manual Pixels/mm)',
            image_id TEXT,
            width_px REAL,
            length_px REAL,
            scale_factor REAL DEFAULT 1.0,
            ellipse_center_x REAL,
            ellipse_center_y REAL,
            ellipse_angle REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (case_id) REFERENCES cases (case_id) ON DELETE SET NULL,
            FOREIGN KEY (evidence_id) REFERENCES evidence (evidence_id) ON DELETE SET NULL
        );
        """
    )

    # Ensure existing measurements table has Phase 3 & Phase 4 columns (safe schema migration)
    cursor.execute("PRAGMA table_info(measurements);")
    existing_cols = {row["name"] for row in cursor.fetchall()}
    migration_columns = [
        ("image_id", "TEXT"),
        ("width_px", "REAL"),
        ("length_px", "REAL"),
        ("scale_factor", "REAL DEFAULT 1.0"),
        ("ellipse_center_x", "REAL"),
        ("ellipse_center_y", "REAL"),
        ("ellipse_angle", "REAL"),
        ("calibration_status", "TEXT DEFAULT 'Uncalibrated (Manual Pixels/mm)'"),
        ("direction_angle", "REAL"),
        ("surface_x_mm", "REAL"),
        ("surface_y_mm", "REAL"),
        ("surface_type", "TEXT DEFAULT 'Horizontal Floor'"),
    ]
    for col_name, col_type in migration_columns:
        if col_name not in existing_cols:
            cursor.execute(f"ALTER TABLE measurements ADD COLUMN {col_name} {col_type};")

    # 4. Evidence Images table (Phase 3 Optical Acquisition)
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS evidence_images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id TEXT UNIQUE NOT NULL,
            case_id TEXT,
            evidence_id TEXT,
            filename TEXT NOT NULL,
            filepath TEXT NOT NULL,
            original_filename TEXT NOT NULL,
            file_size INTEGER NOT NULL,
            mime_type TEXT NOT NULL,
            width_px INTEGER,
            height_px INTEGER,
            scale_factor REAL DEFAULT 1.0,
            is_calibrated INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (case_id) REFERENCES cases (case_id) ON DELETE SET NULL,
            FOREIGN KEY (evidence_id) REFERENCES evidence (evidence_id) ON DELETE SET NULL
        );
        """
    )

    # 5. Trajectory Reconstructions table (Phase 4.3 Multi-Stain Workspace)
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS reconstructions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            reconstruction_id TEXT UNIQUE NOT NULL,
            case_id TEXT NOT NULL,
            reconstruction_name TEXT NOT NULL,
            surface_type TEXT NOT NULL DEFAULT 'Horizontal Floor',
            incline_deg REAL,
            status TEXT NOT NULL DEFAULT 'Draft',
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (case_id) REFERENCES cases (case_id) ON DELETE CASCADE
        );
        """
    )

    # 6. Reconstruction Stains association table (prevents duplicate measurement links)
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS reconstruction_stains (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            reconstruction_id TEXT NOT NULL,
            measurement_id INTEGER NOT NULL,
            stain_identifier TEXT NOT NULL,
            surface_x_mm REAL,
            surface_y_mm REAL,
            direction_angle REAL,
            surface_type TEXT DEFAULT 'Horizontal Floor',
            direction_source TEXT DEFAULT 'Examiner Supplied (Tail/Spine Morphology)',
            sigma_gamma_deg REAL DEFAULT 2.0,
            examiner_notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (reconstruction_id, measurement_id),
            FOREIGN KEY (reconstruction_id) REFERENCES reconstructions (reconstruction_id) ON DELETE CASCADE,
            FOREIGN KEY (measurement_id) REFERENCES measurements (id) ON DELETE CASCADE
        );
        """
    )

    # 7. Reconstruction Results table (persisted 2D Convergence & 3D Origin solutions)
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS reconstruction_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            reconstruction_id TEXT NOT NULL,
            stain_count INTEGER NOT NULL,
            surface_type TEXT NOT NULL,
            convergence_x_mm REAL NOT NULL,
            convergence_y_mm REAL NOT NULL,
            rms_residual_mm REAL NOT NULL,
            max_residual_mm REAL NOT NULL,
            convergence_radius_mm REAL NOT NULL,
            condition_number REAL,
            tangent_origin_x_mm REAL NOT NULL,
            tangent_origin_y_mm REAL NOT NULL,
            tangent_origin_z_mm REAL NOT NULL,
            tangent_z_std_mm REAL NOT NULL,
            tangent_z_uncertainty_mm REAL NOT NULL,
            tangent_z_min_mm REAL NOT NULL,
            tangent_z_max_mm REAL NOT NULL,
            estimated_origin_x_mm REAL NOT NULL,
            estimated_origin_y_mm REAL NOT NULL,
            estimated_origin_z_mm REAL NOT NULL,
            rms_3d_miss_mm REAL NOT NULL,
            result_json TEXT,
            calculation_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (reconstruction_id) REFERENCES reconstructions (reconstruction_id) ON DELETE CASCADE
        );
        """
    )

    # 8. Forensic PDF Reports table (Phase 4.4 — Report Integrity, Revisions & SHA-256 Traceability)
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_id TEXT UNIQUE NOT NULL,
            case_id TEXT,
            reconstruction_id TEXT,
            measurement_id INTEGER,
            report_type TEXT NOT NULL,
            report_title TEXT NOT NULL,
            examiner_name TEXT,
            laboratory_name TEXT,
            examiner_observations TEXT,
            report_status TEXT NOT NULL DEFAULT 'DRAFT',
            revision_number INTEGER NOT NULL DEFAULT 1,
            source_measurement_ids TEXT,
            source_image_ids TEXT,
            image_hashes_json TEXT,
            filename TEXT NOT NULL,
            filepath TEXT NOT NULL,
            file_size INTEGER NOT NULL DEFAULT 0,
            sha256_hash TEXT NOT NULL,
            generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (case_id) REFERENCES cases (case_id) ON DELETE SET NULL,
            FOREIGN KEY (reconstruction_id) REFERENCES reconstructions (reconstruction_id) ON DELETE SET NULL,
            FOREIGN KEY (measurement_id) REFERENCES measurements (id) ON DELETE SET NULL
        );
        """
    )

    # Indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_cases_case_id ON cases(case_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_cases_status ON cases(status);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_evidence_case_id ON evidence(case_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_measurements_case_id ON measurements(case_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_images_image_id ON evidence_images(image_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_images_case_id ON evidence_images(case_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_rec_case_id ON reconstructions(case_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_rec_stains_rec_id ON reconstruction_stains(reconstruction_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_rec_results_rec_id ON reconstruction_results(reconstruction_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_reports_report_id ON reports(report_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_reports_case_id ON reports(case_id);")

    # 9. Forensic Audit Trail table (Phase 4.5 — Non-Destructive Operational Traceability)
    # Note: Records operational timestamps and file SHA-256 integrity hashes at ingestion/generation
    # time; does not replace physical/administrative chain-of-custody documentation.
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            action_type TEXT NOT NULL,
            case_id TEXT,
            entity_type TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            details TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_case_id ON audit_logs(case_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_action_type ON audit_logs(action_type);")

    # 10. Bloodstain Pattern Classification & Decision Support table (Phase 5.1)
    # Preserves strict separation between Automatically Measured features, Software Suggested
    # candidate interpretations, Examiner Observed parameters, and Examiner Approved conclusions.
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS pattern_analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id TEXT UNIQUE NOT NULL,
            case_id TEXT NOT NULL,
            evidence_id TEXT,
            image_id TEXT,
            revision_number INTEGER NOT NULL DEFAULT 1,
            software_version TEXT NOT NULL,
            is_calibrated INTEGER NOT NULL DEFAULT 0,
            mm_per_pixel REAL,
            features_json TEXT NOT NULL,
            suggestions_json TEXT NOT NULL,
            observations_json TEXT NOT NULL,
            overall_determination TEXT NOT NULL,
            suggestion_decision TEXT NOT NULL DEFAULT 'Pending Examiner Review',
            examiner_interpretation TEXT,
            examiner_rationale TEXT,
            examiner_name TEXT,
            review_status TEXT NOT NULL DEFAULT 'Preliminary — Examiner Review Required',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (case_id) REFERENCES cases (case_id) ON DELETE CASCADE,
            FOREIGN KEY (evidence_id) REFERENCES evidence (evidence_id) ON DELETE SET NULL,
            FOREIGN KEY (image_id) REFERENCES evidence_images (image_id) ON DELETE SET NULL
        );
        """
    )
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_pat_case_id ON pattern_analyses(case_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_pat_image_id ON pattern_analyses(image_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_pat_analysis_id ON pattern_analyses(analysis_id);")

    # 11. Overlapping & Complex Bloodstain Analysis table (Phase 5.2)
    # Tracks connected stain regions, marker-watershed candidate subdivisions,
    # and examiner review decisions (Accepted / Rejected / Indeterminate) across non-destructive revisions.
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS overlap_analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id TEXT UNIQUE NOT NULL,
            case_id TEXT NOT NULL,
            evidence_id TEXT,
            image_id TEXT,
            revision_number INTEGER NOT NULL DEFAULT 1,
            algorithm_name TEXT NOT NULL,
            algorithm_version TEXT NOT NULL,
            is_calibrated INTEGER NOT NULL DEFAULT 0,
            mm_per_pixel REAL,
            parameters_json TEXT NOT NULL,
            summary_json TEXT NOT NULL,
            regions_json TEXT NOT NULL,
            accepted_stains_json TEXT,
            review_status TEXT NOT NULL DEFAULT 'Preliminary — Examiner Review Required',
            examiner_name TEXT,
            examiner_notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (case_id) REFERENCES cases (case_id) ON DELETE CASCADE,
            FOREIGN KEY (evidence_id) REFERENCES evidence (evidence_id) ON DELETE SET NULL,
            FOREIGN KEY (image_id) REFERENCES evidence_images (image_id) ON DELETE SET NULL
        );
        """
    )
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_ovr_case_id ON overlap_analyses(case_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_ovr_image_id ON overlap_analyses(image_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_ovr_analysis_id ON overlap_analyses(analysis_id);")

    # 12. Advanced Uncertainty & Sensitivity Analysis table (Phase 5.3)
    # Stores analytical error propagation, Monte Carlo simulation quantiles,
    # and sensitivity ranking metrics with non-destructive revision tracking.
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS uncertainty_analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id TEXT UNIQUE NOT NULL,
            case_id TEXT NOT NULL,
            stain_id INTEGER,
            stain_label TEXT,
            revision_number INTEGER NOT NULL DEFAULT 1,
            parent_analysis_id TEXT,
            width REAL NOT NULL,
            length REAL NOT NULL,
            width_uncertainty REAL,
            length_uncertainty REAL,
            correlation REAL DEFAULT 0.0,
            scale_factor REAL DEFAULT 1.0,
            scale_uncertainty REAL DEFAULT 0.0,
            uncertainty_source TEXT,
            is_manual INTEGER DEFAULT 0,
            iterations INTEGER DEFAULT 10000,
            random_seed INTEGER DEFAULT 42,
            distribution TEXT DEFAULT 'normal',
            perturbation_percentage REAL DEFAULT 5.0,
            analysis_data_json TEXT NOT NULL,
            review_status TEXT NOT NULL DEFAULT 'Preliminary — Examiner Review Required',
            examiner_name TEXT,
            examiner_notes TEXT,
            reviewed_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (case_id) REFERENCES cases (case_id) ON DELETE CASCADE,
            FOREIGN KEY (stain_id) REFERENCES measurements (id) ON DELETE SET NULL
        );
        """
    )
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_unc_case_id ON uncertainty_analyses(case_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_unc_stain_id ON uncertainty_analyses(stain_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_unc_analysis_id ON uncertainty_analyses(analysis_id);")

    conn.commit()
    conn.close()

    if seed_demo:
        seed_demo_data()


def clear_all_data() -> None:
    """Clear all records from uncertainty_analyses, overlap_analyses, pattern_analyses, audit_logs, reports, reconstruction tables, measurements, evidence_images, evidence, and cases."""
    conn = get_db_connection()
    try:
        conn.execute("DELETE FROM uncertainty_analyses;")
        conn.execute("DELETE FROM overlap_analyses;")
        conn.execute("DELETE FROM pattern_analyses;")
        conn.execute("DELETE FROM audit_logs;")
        conn.execute("DELETE FROM reports;")
        conn.execute("DELETE FROM reconstruction_results;")
        conn.execute("DELETE FROM reconstruction_stains;")
        conn.execute("DELETE FROM reconstructions;")
        conn.execute("DELETE FROM measurements;")
        conn.execute("DELETE FROM evidence_images;")
        conn.execute("DELETE FROM evidence;")
        conn.execute("DELETE FROM cases;")
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def log_audit_event(
    action_type: str,
    entity_type: str,
    entity_id: Any,
    case_id: Optional[str] = None,
    details: str = "",
    conn: Optional[sqlite3.Connection] = None,
) -> int:
    """
    Record a forensic operational audit trail entry.
    Note: SHA-256 hashes logged in details verify digital file integrity at ingestion/generation
    time and do not independently establish physical chain of custody.
    """
    own_conn = conn is None
    active_conn = conn if conn is not None else get_db_connection()
    try:
        cursor = active_conn.cursor()
        cursor.execute(
            """
            INSERT INTO audit_logs (action_type, case_id, entity_type, entity_id, details)
            VALUES (?, ?, ?, ?, ?);
            """,
            (
                action_type.strip(),
                case_id.strip() if case_id and str(case_id).strip() else None,
                entity_type.strip(),
                str(entity_id).strip(),
                details.strip() if details else "",
            ),
        )
        row_id = cursor.lastrowid
        if own_conn:
            active_conn.commit()
        return int(row_id)
    except Exception:
        if own_conn:
            active_conn.rollback()
        raise
    finally:
        if own_conn:
            active_conn.close()


def get_audit_logs(
    case_id: Optional[str] = None,
    action_type: Optional[str] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """Retrieve operational audit log entries ordered newest first."""
    conn = get_db_connection()
    query = "SELECT * FROM audit_logs WHERE 1=1"
    params: List[Any] = []
    if case_id and case_id.strip():
        query += " AND case_id = ?"
        params.append(case_id.strip())
    if action_type and action_type.strip():
        query += " AND action_type = ?"
        params.append(action_type.strip())
    query += " ORDER BY id DESC LIMIT ?;"
    params.append(int(limit))
    cursor = conn.execute(query, params)
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def seed_demo_data() -> None:
    """Populate initial baseline demonstration cases and evidence records."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM cases;")
        count = cursor.fetchone()[0]

        if count == 0:
            initial_cases = [
                (
                    "BPA-2026-0101",
                    "Homicide Scene — 142 Elm Street",
                    "REF-CR-2026-8812",
                    "2026-11-02",
                    "Senior Analyst S. Morgan",
                    "Medium-velocity spatter cluster documented on living room north drywall. Surface is smooth latex paint.",
                    "Active",
                ),
                (
                    "BPA-2026-0102",
                    "Aggravated Assault Investigation — Harbor Warehouse",
                    "REF-CR-2026-9041",
                    "2026-10-28",
                    "Forensic Specialist J. Vance",
                    "Cast-off bloodstain pattern along polished concrete flooring and adjacent metal shelving unit.",
                    "Under Review",
                ),
                (
                    "BPA-2026-0103",
                    "Laboratory Reference Pattern Verification Study",
                    "REF-LAB-2026-004",
                    "2026-10-15",
                    "Dr. A. Chen, Lead BPA",
                    "Controlled laboratory calibration drops on non-porous glazed ceramic tiles at known 30°, 45°, and 90° angles.",
                    "Completed",
                ),
            ]

            cursor.executemany(
                """
                INSERT INTO cases (case_id, title, reference_no, examination_date, examiner, description, status)
                VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                initial_cases,
            )

            initial_evidence = [
                (
                    "EVD-0101-01",
                    "BPA-2026-0101",
                    "Impact spatter group on north wall drywall, 1.2m above floor level",
                    "Impact Spatter",
                    "Smooth painted vertical drywall",
                    "Logged",
                    "Swabbed for DNA, photographs taken with scale bar",
                ),
                (
                    "EVD-0101-02",
                    "BPA-2026-0101",
                    "Elongated directional stains with distinct tail directionality pointing west",
                    "Blood Spatter",
                    "Hardwood parquet flooring",
                    "Pending Analysis",
                    "Directionality indicates droplet traveled eastward before impact",
                ),
                (
                    "EVD-0102-01",
                    "BPA-2026-0102",
                    "Linear cast-off trail comprising 7 distinct elliptical droplets",
                    "Cast-off Pattern",
                    "Polished industrial concrete",
                    "Pending Analysis",
                    "Apparent weapon swing trajectory from south to north",
                ),
                (
                    "EVD-0103-01",
                    "BPA-2026-0103",
                    "Synthetic blood test drop group A (known 30.0° impact)",
                    "Controlled Spatter",
                    "Glazed non-porous ceramic tile",
                    "Analyzed",
                    "Reference verification standard",
                ),
            ]

            cursor.executemany(
                """
                INSERT INTO evidence (evidence_id, case_id, description, evidence_type, location_surface, status, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                initial_evidence,
            )

            # Initial baseline measurement
            cursor.execute(
                """
                INSERT INTO measurements (case_id, evidence_id, stain_label, width, length, ratio, angle_degrees, angle_radians, is_perpendicular, interpretation)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    "BPA-2026-0103",
                    "EVD-0103-01",
                    "Stain-A1",
                    5.0,
                    10.0,
                    0.5,
                    30.0,
                    0.5236,
                    0,
                    "An estimated impact angle of 30.00° indicates an oblique strike. The droplet struck at an intermediate incline.",
                ),
            )
            conn.commit()
    finally:
        conn.close()


def generate_case_id() -> str:
    """Generate a unique forensic case ID in the format BPA-YYYY-NNNN."""
    conn = get_db_connection()
    current_year = datetime.now().year
    prefix = f"BPA-{current_year}-"

    while True:
        num = random.randint(1000, 9999)
        candidate = f"{prefix}{num}"
        cursor = conn.execute("SELECT 1 FROM cases WHERE case_id = ?;", (candidate,))
        if not cursor.fetchone():
            conn.close()
            return candidate


def generate_evidence_id(case_id: str) -> str:
    """Generate a unique evidence ID associated with a case ID (e.g. EVD-0101-03)."""
    conn = get_db_connection()
    # Extract numeric part of case_id if available
    parts = case_id.split("-")
    tag = parts[-1] if len(parts) >= 2 else "0001"

    cursor = conn.execute(
        "SELECT COUNT(*) FROM evidence WHERE case_id = ?;",
        (case_id,),
    )
    count = cursor.fetchone()[0] + 1
    evidence_id = f"EVD-{tag}-{count:02d}"

    # Ensure uniqueness
    while True:
        check = conn.execute("SELECT 1 FROM evidence WHERE evidence_id = ?;", (evidence_id,)).fetchone()
        if not check:
            break
        count += 1
        evidence_id = f"EVD-{tag}-{count:02d}"

    conn.close()
    return evidence_id


def generate_image_id() -> str:
    """Generate a unique evidence image ID in the format IMG-YYYY-NNNN."""
    conn = get_db_connection()
    current_year = datetime.now().year
    prefix = f"IMG-{current_year}-"

    while True:
        num = random.randint(1000, 9999)
        candidate = f"{prefix}{num}"
        cursor = conn.execute("SELECT 1 FROM evidence_images WHERE image_id = ?;", (candidate,))
        if not cursor.fetchone():
            conn.close()
            return candidate


# ---------------------------------------------------------------------------
# Case Queries & Mutations
# ---------------------------------------------------------------------------


def create_case(
    case_id: str,
    title: str,
    reference_no: str,
    examination_date: str,
    examiner: str,
    description: str = "",
    status: str = "Active",
) -> Dict[str, Any]:
    """Insert a new forensic case record."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO cases (case_id, title, reference_no, examination_date, examiner, description, status)
            VALUES (?, ?, ?, ?, ?, ?, ?);
            """,
            (
                case_id.strip(),
                title.strip(),
                reference_no.strip(),
                examination_date.strip(),
                examiner.strip(),
                description.strip() if description else "",
                status.strip(),
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return get_case_by_id(case_id)


def get_all_cases(
    search_query: Optional[str] = None,
    status_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Retrieve all cases with optional search and status filtering."""
    conn = get_db_connection()
    sql = """
        SELECT c.*, COUNT(e.id) AS evidence_count
        FROM cases c
        LEFT JOIN evidence e ON c.case_id = e.case_id
    """
    params: List[Any] = []
    conditions: List[str] = []

    if status_filter and status_filter.lower() != "all":
        conditions.append("c.status = ?")
        params.append(status_filter)

    if search_query and search_query.strip():
        term = f"%{search_query.strip()}%"
        conditions.append(
            "(c.case_id LIKE ? OR c.title LIKE ? OR c.reference_no LIKE ? OR c.examiner LIKE ?)"
        )
        params.extend([term, term, term, term])

    if conditions:
        sql += " WHERE " + " AND ".join(conditions)

    sql += " GROUP BY c.id ORDER BY c.created_at DESC;"

    cursor = conn.execute(sql, params)
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def get_case_by_id(case_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a single case by its unique case_id, including its evidence count."""
    conn = get_db_connection()
    cursor = conn.execute(
        """
        SELECT c.*, COUNT(e.id) AS evidence_count
        FROM cases c
        LEFT JOIN evidence e ON c.case_id = e.case_id
        WHERE c.case_id = ?
        GROUP BY c.id;
        """,
        (case_id.strip(),),
    )
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def update_case_status(case_id: str, new_status: str) -> bool:
    """Update the status of an existing case."""
    conn = get_db_connection()
    cursor = conn.execute(
        """
        UPDATE cases
        SET status = ?, updated_at = CURRENT_TIMESTAMP
        WHERE case_id = ?;
        """,
        (new_status, case_id.strip()),
    )
    updated = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return updated


def get_case_statistics() -> Dict[str, int]:
    """Retrieve actual counts for dashboard metric widgets."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM cases;")
    total_cases = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM cases WHERE status = 'Active';")
    active_cases = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM cases WHERE status = 'Completed';")
    completed_cases = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM cases WHERE status = 'Under Review';")
    under_review_cases = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM evidence;")
    total_evidence = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM measurements;")
    total_measurements = cursor.fetchone()[0]

    conn.close()
    return {
        "total_cases": total_cases,
        "active_cases": active_cases,
        "completed_cases": completed_cases,
        "under_review_cases": under_review_cases,
        "total_evidence": total_evidence,
        "total_measurements": total_measurements,
    }


def get_recent_cases(limit: int = 5) -> List[Dict[str, Any]]:
    """Retrieve the most recently updated or created cases."""
    conn = get_db_connection()
    cursor = conn.execute(
        """
        SELECT c.*, COUNT(e.id) AS evidence_count
        FROM cases c
        LEFT JOIN evidence e ON c.case_id = e.case_id
        GROUP BY c.id
        ORDER BY c.updated_at DESC, c.id DESC
        LIMIT ?;
        """,
        (limit,),
    )
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


# ---------------------------------------------------------------------------
# Evidence Queries & Mutations
# ---------------------------------------------------------------------------


def create_evidence(
    evidence_id: str,
    case_id: str,
    description: str,
    evidence_type: str,
    location_surface: str = "",
    status: str = "Logged",
    notes: str = "",
) -> Dict[str, Any]:
    """Insert a new evidence record linked to a case."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO evidence (evidence_id, case_id, description, evidence_type, location_surface, status, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?);
            """,
            (
                evidence_id.strip(),
                case_id.strip(),
                description.strip(),
                evidence_type.strip(),
                location_surface.strip() if location_surface else "",
                status.strip(),
                notes.strip() if notes else "",
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return get_evidence_by_id(evidence_id)


def get_all_evidence(
    case_id: Optional[str] = None,
    status_filter: Optional[str] = None,
    search_query: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Retrieve all evidence records with optional filtering."""
    conn = get_db_connection()
    sql = "SELECT * FROM evidence"
    params: List[Any] = []
    conditions: List[str] = []

    if case_id and case_id.strip():
        conditions.append("case_id = ?")
        params.append(case_id.strip())

    if status_filter and status_filter.lower() != "all":
        conditions.append("status = ?")
        params.append(status_filter.strip())

    if search_query and search_query.strip():
        term = f"%{search_query.strip()}%"
        conditions.append(
            "(evidence_id LIKE ? OR description LIKE ? OR location_surface LIKE ? OR evidence_type LIKE ?)"
        )
        params.extend([term, term, term, term])

    if conditions:
        sql += " WHERE " + " AND ".join(conditions)

    sql += " ORDER BY created_at DESC;"

    cursor = conn.execute(sql, params)
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def get_evidence_by_id(evidence_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a single evidence record by its unique evidence_id."""
    conn = get_db_connection()
    cursor = conn.execute(
        "SELECT * FROM evidence WHERE evidence_id = ?;",
        (evidence_id.strip(),),
    )
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_evidence_for_case(case_id: str) -> List[Dict[str, Any]]:
    """Retrieve all evidence records attached to a specific case."""
    conn = get_db_connection()
    cursor = conn.execute(
        "SELECT * FROM evidence WHERE case_id = ? ORDER BY created_at ASC;",
        (case_id.strip(),),
    )
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def update_evidence_status(evidence_id: str, new_status: str) -> bool:
    """Update the status of an evidence record."""
    conn = get_db_connection()
    cursor = conn.execute(
        "UPDATE evidence SET status = ? WHERE evidence_id = ?;",
        (new_status, evidence_id.strip()),
    )
    updated = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return updated


# ---------------------------------------------------------------------------
# Measurement Queries & Mutations
# ---------------------------------------------------------------------------


def create_measurement(
    case_id: Optional[str],
    evidence_id: Optional[str],
    stain_label: str,
    width: float,
    length: float,
    ratio: float,
    angle_degrees: float,
    angle_radians: float,
    is_perpendicular: bool,
    interpretation: str,
    image_id: Optional[str] = None,
    width_px: Optional[float] = None,
    length_px: Optional[float] = None,
    scale_factor: float = 1.0,
    ellipse_center_x: Optional[float] = None,
    ellipse_center_y: Optional[float] = None,
    ellipse_angle: Optional[float] = None,
    calibration_status: Optional[str] = None,
    direction_angle: Optional[float] = None,
    surface_x_mm: Optional[float] = None,
    surface_y_mm: Optional[float] = None,
    surface_type: str = "Horizontal Floor",
) -> int:
    """Insert a verified stain calculation record into the history log."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO measurements (
                case_id, evidence_id, stain_label, width, length, ratio,
                angle_degrees, angle_radians, is_perpendicular, interpretation,
                image_id, width_px, length_px, scale_factor,
                ellipse_center_x, ellipse_center_y, ellipse_angle, calibration_status,
                direction_angle, surface_x_mm, surface_y_mm, surface_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                case_id if case_id else None,
                evidence_id if evidence_id else None,
                stain_label if stain_label else "Stain-01",
                width,
                length,
                ratio,
                angle_degrees,
                angle_radians,
                1 if is_perpendicular else 0,
                interpretation,
                image_id if image_id else None,
                width_px,
                length_px,
                scale_factor,
                ellipse_center_x,
                ellipse_center_y,
                ellipse_angle,
                calibration_status if calibration_status else "Uncalibrated (Manual Pixels/mm)",
                direction_angle,
                surface_x_mm,
                surface_y_mm,
                surface_type if surface_type else "Horizontal Floor",
            ),
        )
        row_id = cursor.lastrowid
        log_audit_event(
            action_type="Measurement Creation",
            entity_type="measurement",
            entity_id=row_id,
            case_id=case_id,
            details=f"Stain={stain_label or 'Stain-01'}, W={width:.2f}mm, L={length:.2f}mm, alpha={angle_degrees:.2f}deg",
            conn=conn,
        )
        conn.commit()
        return row_id
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_recent_measurements(
    limit: int = 10,
    case_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Retrieve recent measurement history records."""
    conn = get_db_connection()
    sql = "SELECT * FROM measurements"
    params: List[Any] = []

    if case_id and case_id.strip():
        sql += " WHERE case_id = ?"
        params.append(case_id.strip())

    sql += " ORDER BY created_at DESC LIMIT ?;"
    params.append(limit)

    cursor = conn.execute(sql, params)
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


# ---------------------------------------------------------------------------
# Evidence Image Queries & Mutations (Phase 3 Optical Staging)
# ---------------------------------------------------------------------------


def create_evidence_image(
    image_id: str,
    case_id: Optional[str],
    evidence_id: Optional[str],
    filename: str,
    filepath: str,
    original_filename: str,
    file_size: int,
    mime_type: str,
    width_px: int,
    height_px: int,
    scale_factor: float = 1.0,
    is_calibrated: int = 0,
) -> Dict[str, Any]:
    """Insert a photographic evidence record linked to a case and/or evidence item."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO evidence_images (
                image_id, case_id, evidence_id, filename, filepath,
                original_filename, file_size, mime_type, width_px, height_px,
                scale_factor, is_calibrated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                image_id.strip(),
                case_id.strip() if case_id else None,
                evidence_id.strip() if evidence_id else None,
                filename.strip(),
                filepath.strip(),
                original_filename.strip(),
                file_size,
                mime_type.strip(),
                width_px,
                height_px,
                scale_factor,
                is_calibrated,
            ),
        )
        log_audit_event(
            action_type="Evidence Upload",
            entity_type="evidence_image",
            entity_id=image_id.strip(),
            case_id=case_id,
            details=f"File={original_filename.strip()} ({width_px}x{height_px}px, {file_size} bytes)",
            conn=conn,
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return get_image_by_id(image_id)


def get_image_by_id(image_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve an evidence image record by image_id."""
    conn = get_db_connection()
    cursor = conn.execute(
        "SELECT * FROM evidence_images WHERE image_id = ?;",
        (image_id.strip(),),
    )
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_images_for_case(case_id: str) -> List[Dict[str, Any]]:
    """Retrieve all images uploaded for a specific case."""
    conn = get_db_connection()
    cursor = conn.execute(
        "SELECT * FROM evidence_images WHERE case_id = ? ORDER BY created_at DESC;",
        (case_id.strip(),),
    )
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def get_images_for_evidence(evidence_id: str) -> List[Dict[str, Any]]:
    """Retrieve all images uploaded for a specific evidence item."""
    conn = get_db_connection()
    cursor = conn.execute(
        "SELECT * FROM evidence_images WHERE evidence_id = ? ORDER BY created_at DESC;",
        (evidence_id.strip(),),
    )
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def get_all_images(limit: int = 25) -> List[Dict[str, Any]]:
    """Retrieve recent evidence images across all cases."""
    conn = get_db_connection()
    cursor = conn.execute(
        "SELECT * FROM evidence_images ORDER BY created_at DESC LIMIT ?;",
        (limit,),
    )
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def update_image_calibration(image_id: str, scale_factor: float, is_calibrated: int = 1) -> bool:
    """Update scale factor (mm per pixel) and calibration state for an image."""
    conn = get_db_connection()
    cursor = conn.execute(
        """
        UPDATE evidence_images
        SET scale_factor = ?, is_calibrated = ?
        WHERE image_id = ?;
        """,
        (scale_factor, is_calibrated, image_id.strip()),
    )
    updated = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return updated


# ---------------------------------------------------------------------------
# Phase 4.3: Trajectory Reconstruction Queries & Mutations
# ---------------------------------------------------------------------------


def generate_reconstruction_id() -> str:
    """Generate a unique trajectory reconstruction ID in the format REC-YYYY-NNNN."""
    conn = get_db_connection()
    current_year = datetime.now().year
    prefix = f"REC-{current_year}-"

    while True:
        num = random.randint(1000, 9999)
        candidate = f"{prefix}{num}"
        cursor = conn.execute("SELECT 1 FROM reconstructions WHERE reconstruction_id = ?;", (candidate,))
        if not cursor.fetchone():
            conn.close()
            return candidate


def get_measurement_by_id(measurement_id: int) -> Optional[Dict[str, Any]]:
    """Retrieve a single measurement record by its integer primary key id."""
    conn = get_db_connection()
    cursor = conn.execute(
        """
        SELECT m.*, ei.is_calibrated AS image_is_calibrated, ei.scale_factor AS image_scale_factor,
               ei.original_filename AS image_filename
        FROM measurements m
        LEFT JOIN evidence_images ei ON m.image_id = ei.image_id
        WHERE m.id = ?;
        """,
        (int(measurement_id),),
    )
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_measurements_for_case(case_id: str) -> List[Dict[str, Any]]:
    """Retrieve all measurements associated with a specific case, joined with image calibration metadata."""
    conn = get_db_connection()
    cursor = conn.execute(
        """
        SELECT m.*, ei.is_calibrated AS image_is_calibrated, ei.scale_factor AS image_scale_factor,
               ei.original_filename AS image_filename
        FROM measurements m
        LEFT JOIN evidence_images ei ON m.image_id = ei.image_id
        WHERE m.case_id = ?
        ORDER BY m.created_at DESC, m.id DESC;
        """,
        (case_id.strip(),),
    )
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def create_reconstruction(
    case_id: str,
    reconstruction_name: str = "Primary Scene Reconstruction",
    surface_type: str = "Horizontal Floor",
    incline_deg: Optional[float] = None,
    notes: str = "",
    reconstruction_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a new trajectory reconstruction workspace linked to a forensic case."""
    rec_id = reconstruction_id.strip() if reconstruction_id else generate_reconstruction_id()
    conn = get_db_connection()
    try:
        conn.execute(
            """
            INSERT INTO reconstructions (
                reconstruction_id, case_id, reconstruction_name, surface_type, incline_deg, status, notes
            ) VALUES (?, ?, ?, ?, ?, 'Draft', ?);
            """,
            (
                rec_id,
                case_id.strip(),
                reconstruction_name.strip(),
                surface_type.strip() if surface_type else "Horizontal Floor",
                incline_deg,
                notes.strip() if notes else "",
            ),
        )
        log_audit_event(
            action_type="Reconstruction Creation",
            entity_type="reconstruction",
            entity_id=rec_id,
            case_id=case_id,
            details=f"Name={reconstruction_name.strip()}, Surface={surface_type.strip() if surface_type else 'Horizontal Floor'}",
            conn=conn,
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return get_reconstruction_by_id(rec_id)


def get_reconstruction_by_id(reconstruction_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a reconstruction record by its unique reconstruction_id."""
    conn = get_db_connection()
    cursor = conn.execute(
        "SELECT * FROM reconstructions WHERE reconstruction_id = ?;",
        (reconstruction_id.strip(),),
    )
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_reconstructions_for_case(case_id: str) -> List[Dict[str, Any]]:
    """Retrieve all trajectory reconstructions for a given case_id."""
    conn = get_db_connection()
    cursor = conn.execute(
        "SELECT * FROM reconstructions WHERE case_id = ? ORDER BY updated_at DESC, id DESC;",
        (case_id.strip(),),
    )
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def update_reconstruction_metadata(
    reconstruction_id: str,
    surface_type: Optional[str] = None,
    incline_deg: Optional[float] = None,
    status: Optional[str] = None,
    reconstruction_name: Optional[str] = None,
) -> bool:
    """Update receiving surface configuration or status on a reconstruction workspace."""
    rec = get_reconstruction_by_id(reconstruction_id)
    if not rec:
        return False
    new_surf = surface_type.strip() if surface_type else rec["surface_type"]
    new_inc = incline_deg if incline_deg is not None else rec["incline_deg"]
    new_stat = status.strip() if status else rec["status"]
    new_name = reconstruction_name.strip() if reconstruction_name else rec["reconstruction_name"]

    conn = get_db_connection()
    cursor = conn.execute(
        """
        UPDATE reconstructions
        SET surface_type = ?, incline_deg = ?, status = ?, reconstruction_name = ?, updated_at = CURRENT_TIMESTAMP
        WHERE reconstruction_id = ?;
        """,
        (new_surf, new_inc, new_stat, new_name, reconstruction_id.strip()),
    )
    updated = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return updated


def add_stain_to_reconstruction(
    reconstruction_id: str,
    measurement_id: int,
    stain_identifier: Optional[str] = None,
    surface_x_mm: Optional[float] = None,
    surface_y_mm: Optional[float] = None,
    direction_angle: Optional[float] = None,
    surface_type: Optional[str] = None,
    direction_source: str = "Examiner Supplied (Tail/Spine Morphology)",
    sigma_gamma_deg: float = 2.0,
    examiner_notes: str = "",
) -> Dict[str, Any]:
    """
    Associate an existing measurement with a reconstruction workspace.
    Raises sqlite3.IntegrityError if the measurement is already added to this reconstruction.
    """
    meas = get_measurement_by_id(measurement_id)
    if not meas:
        raise ValueError(f"Measurement ID {measurement_id} does not exist.")

    eff_label = (stain_identifier or meas.get("stain_label") or f"Stain-{measurement_id}").strip()
    eff_x = surface_x_mm if surface_x_mm is not None else meas.get("surface_x_mm")
    eff_y = surface_y_mm if surface_y_mm is not None else meas.get("surface_y_mm")
    eff_dir = direction_angle if direction_angle is not None else meas.get("direction_angle")
    eff_surf = (surface_type or meas.get("surface_type") or "Horizontal Floor").strip()

    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO reconstruction_stains (
                reconstruction_id, measurement_id, stain_identifier,
                surface_x_mm, surface_y_mm, direction_angle, surface_type,
                direction_source, sigma_gamma_deg, examiner_notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                reconstruction_id.strip(),
                int(measurement_id),
                eff_label,
                eff_x,
                eff_y,
                eff_dir,
                eff_surf,
                direction_source.strip() if direction_source else "Examiner Supplied",
                sigma_gamma_deg,
                examiner_notes.strip() if examiner_notes else "",
            ),
        )
        row_id = cursor.lastrowid
        conn.execute(
            "UPDATE reconstructions SET updated_at = CURRENT_TIMESTAMP WHERE reconstruction_id = ?;",
            (reconstruction_id.strip(),),
        )
        conn.commit()
    finally:
        conn.close()

    stains = get_stains_for_reconstruction(reconstruction_id)
    for st in stains:
        if st["id"] == row_id:
            return st
    return stains[-1]


def remove_stain_from_reconstruction(reconstruction_id: str, measurement_id: int) -> bool:
    """Remove a measurement association from a reconstruction workspace."""
    conn = get_db_connection()
    cursor = conn.execute(
        "DELETE FROM reconstruction_stains WHERE reconstruction_id = ? AND measurement_id = ?;",
        (reconstruction_id.strip(), int(measurement_id)),
    )
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted


def update_reconstruction_stain(
    reconstruction_id: str,
    measurement_id: int,
    stain_identifier: Optional[str] = None,
    surface_x_mm: Optional[float] = None,
    surface_y_mm: Optional[float] = None,
    direction_angle: Optional[float] = None,
    surface_type: Optional[str] = None,
    direction_source: Optional[str] = None,
    sigma_gamma_deg: Optional[float] = None,
    examiner_notes: Optional[str] = None,
) -> bool:
    """Update directional, coordinate, or surface parameters for a stain in a reconstruction."""
    conn = get_db_connection()
    cursor = conn.execute(
        "SELECT * FROM reconstruction_stains WHERE reconstruction_id = ? AND measurement_id = ?;",
        (reconstruction_id.strip(), int(measurement_id)),
    )
    existing = cursor.fetchone()
    if not existing:
        conn.close()
        return False

    new_ident = stain_identifier.strip() if stain_identifier else existing["stain_identifier"]
    new_x = surface_x_mm if surface_x_mm is not None else existing["surface_x_mm"]
    new_y = surface_y_mm if surface_y_mm is not None else existing["surface_y_mm"]
    new_dir = direction_angle if direction_angle is not None else existing["direction_angle"]
    new_surf = surface_type.strip() if surface_type else existing["surface_type"]
    new_src = direction_source.strip() if direction_source else existing["direction_source"]
    new_sig = sigma_gamma_deg if sigma_gamma_deg is not None else existing["sigma_gamma_deg"]
    new_notes = examiner_notes.strip() if examiner_notes is not None else existing["examiner_notes"]

    try:
        cursor.execute(
            """
            UPDATE reconstruction_stains
            SET stain_identifier = ?, surface_x_mm = ?, surface_y_mm = ?,
                direction_angle = ?, surface_type = ?, direction_source = ?,
                sigma_gamma_deg = ?, examiner_notes = ?
            WHERE reconstruction_id = ? AND measurement_id = ?;
            """,
            (
                new_ident,
                new_x,
                new_y,
                new_dir,
                new_surf,
                new_src,
                new_sig,
                new_notes,
                reconstruction_id.strip(),
                int(measurement_id),
            ),
        )
        # Also synchronize back to measurements row for consistency
        cursor.execute(
            """
            UPDATE measurements
            SET surface_x_mm = ?, surface_y_mm = ?, direction_angle = ?, surface_type = ?
            WHERE id = ?;
            """,
            (new_x, new_y, new_dir, new_surf, int(measurement_id)),
        )
        rec_row = conn.execute(
            "SELECT case_id FROM reconstructions WHERE reconstruction_id = ?;",
            (reconstruction_id.strip(),),
        ).fetchone()
        rec_case_id = rec_row["case_id"] if rec_row else None
        log_audit_event(
            action_type="Measurement Modification",
            entity_type="measurement",
            entity_id=int(measurement_id),
            case_id=rec_case_id,
            details=f"Reconstruction={reconstruction_id.strip()}, Stain={new_ident}, X={new_x}, Y={new_y}, gamma={new_dir}",
            conn=conn,
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return True


def get_stains_for_reconstruction(reconstruction_id: str) -> List[Dict[str, Any]]:
    """Retrieve all stains in a reconstruction joined with measurement and optical calibration metadata."""
    conn = get_db_connection()
    cursor = conn.execute(
        """
        SELECT rs.*,
               m.case_id, m.evidence_id, m.stain_label, m.width, m.length, m.ratio,
               m.angle_degrees, m.angle_radians, m.is_perpendicular, m.interpretation,
               m.calibration_status, m.image_id, m.width_px, m.length_px,
               m.scale_factor AS measurement_scale_factor,
               ei.is_calibrated AS image_is_calibrated,
               ei.scale_factor AS image_scale_factor,
               ei.original_filename AS image_filename
        FROM reconstruction_stains rs
        JOIN measurements m ON rs.measurement_id = m.id
        LEFT JOIN evidence_images ei ON m.image_id = ei.image_id
        WHERE rs.reconstruction_id = ?
        ORDER BY rs.id ASC;
        """,
        (reconstruction_id.strip(),),
    )
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def save_reconstruction_result(
    reconstruction_id: str,
    stain_count: int,
    surface_type: str,
    convergence_x_mm: float,
    convergence_y_mm: float,
    rms_residual_mm: float,
    max_residual_mm: float,
    convergence_radius_mm: float,
    condition_number: float,
    tangent_origin_x_mm: float,
    tangent_origin_y_mm: float,
    tangent_origin_z_mm: float,
    tangent_z_std_mm: float,
    tangent_z_uncertainty_mm: float,
    tangent_z_min_mm: float,
    tangent_z_max_mm: float,
    estimated_origin_x_mm: float,
    estimated_origin_y_mm: float,
    estimated_origin_z_mm: float,
    rms_3d_miss_mm: float,
    result_json: str,
) -> Dict[str, Any]:
    """Persist a verified 2D convergence and 3D origin reconstruction result."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO reconstruction_results (
                reconstruction_id, stain_count, surface_type,
                convergence_x_mm, convergence_y_mm, rms_residual_mm, max_residual_mm,
                convergence_radius_mm, condition_number,
                tangent_origin_x_mm, tangent_origin_y_mm, tangent_origin_z_mm,
                tangent_z_std_mm, tangent_z_uncertainty_mm, tangent_z_min_mm, tangent_z_max_mm,
                estimated_origin_x_mm, estimated_origin_y_mm, estimated_origin_z_mm,
                rms_3d_miss_mm, result_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                reconstruction_id.strip(),
                stain_count,
                surface_type.strip(),
                convergence_x_mm,
                convergence_y_mm,
                rms_residual_mm,
                max_residual_mm,
                convergence_radius_mm,
                condition_number,
                tangent_origin_x_mm,
                tangent_origin_y_mm,
                tangent_origin_z_mm,
                tangent_z_std_mm,
                tangent_z_uncertainty_mm,
                tangent_z_min_mm,
                tangent_z_max_mm,
                estimated_origin_x_mm,
                estimated_origin_y_mm,
                estimated_origin_z_mm,
                rms_3d_miss_mm,
                result_json,
            ),
        )
        conn.execute(
            "UPDATE reconstructions SET status = 'Calculated', updated_at = CURRENT_TIMESTAMP WHERE reconstruction_id = ?;",
            (reconstruction_id.strip(),),
        )
        rec_row = conn.execute(
            "SELECT case_id FROM reconstructions WHERE reconstruction_id = ?;",
            (reconstruction_id.strip(),),
        ).fetchone()
        rec_case_id = rec_row["case_id"] if rec_row else None
        log_audit_event(
            action_type="Reconstruction Calculation",
            entity_type="reconstruction",
            entity_id=reconstruction_id.strip(),
            case_id=rec_case_id,
            details=f"Stains={stain_count}, Conv=({convergence_x_mm:.1f},{convergence_y_mm:.1f})mm, Z_tan={tangent_origin_z_mm:.1f}mm",
            conn=conn,
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return get_latest_reconstruction_result(reconstruction_id)


def get_latest_reconstruction_result(reconstruction_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve the most recent calculated result for a reconstruction workspace."""
    conn = get_db_connection()
    cursor = conn.execute(
        """
        SELECT * FROM reconstruction_results
        WHERE reconstruction_id = ?
        ORDER BY calculation_timestamp DESC, id DESC
        LIMIT 1;
        """,
        (reconstruction_id.strip(),),
    )
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


# ===========================================================================
# Phase 4.4: Forensic PDF Report Registry, Revisions & Traceability
# ===========================================================================


def get_next_report_revision(
    report_type: str,
    case_id: Optional[str] = None,
    reconstruction_id: Optional[str] = None,
    measurement_id: Optional[int] = None,
) -> int:
    """
    Determine the next sequential revision number (1, 2, 3...) for a given report scope.
    Ensures regenerated reports increment revision numbers and never overwrite prior reports.
    """
    conn = get_db_connection()
    query = "SELECT MAX(revision_number) FROM reports WHERE report_type = ?"
    params: List[Any] = [report_type.strip()]

    if case_id:
        query += " AND case_id = ?"
        params.append(case_id.strip())
    else:
        query += " AND case_id IS NULL"

    if reconstruction_id:
        query += " AND reconstruction_id = ?"
        params.append(reconstruction_id.strip())
    else:
        query += " AND reconstruction_id IS NULL"

    if measurement_id is not None:
        query += " AND measurement_id = ?"
        params.append(int(measurement_id))
    else:
        query += " AND measurement_id IS NULL"

    cursor = conn.execute(query + ";", params)
    row = cursor.fetchone()
    conn.close()
    max_rev = row[0] if row and row[0] is not None else 0
    return int(max_rev) + 1


def generate_report_id(revision_number: int = 1) -> str:
    """Generate a unique forensic report ID in the format RPT-YYYY-NNNN-R<rev>."""
    conn = get_db_connection()
    current_year = datetime.now().year
    while True:
        num = random.randint(1000, 9999)
        candidate = f"RPT-{current_year}-{num}-R{int(revision_number)}"
        cursor = conn.execute("SELECT 1 FROM reports WHERE report_id = ?;", (candidate,))
        if not cursor.fetchone():
            conn.close()
            return candidate


def create_report_record(
    report_id: str,
    report_type: str,
    report_title: str,
    filename: str,
    filepath: str,
    file_size: int,
    sha256_hash: str,
    report_status: str = "DRAFT",
    revision_number: int = 1,
    case_id: Optional[str] = None,
    reconstruction_id: Optional[str] = None,
    measurement_id: Optional[int] = None,
    examiner_name: Optional[str] = None,
    laboratory_name: Optional[str] = None,
    examiner_observations: Optional[str] = None,
    source_measurement_ids: Optional[str] = None,
    source_image_ids: Optional[str] = None,
    image_hashes_json: Optional[str] = None,
) -> Dict[str, Any]:
    """Insert a generated forensic PDF report record into SQLite and return the saved row."""
    conn = get_db_connection()
    try:
        conn.execute(
            """
            INSERT INTO reports (
                report_id, case_id, reconstruction_id, measurement_id,
                report_type, report_title, examiner_name, laboratory_name,
                examiner_observations, report_status, revision_number,
                source_measurement_ids, source_image_ids, image_hashes_json,
                filename, filepath, file_size, sha256_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                report_id.strip(),
                case_id.strip() if case_id else None,
                reconstruction_id.strip() if reconstruction_id else None,
                int(measurement_id) if measurement_id is not None else None,
                report_type.strip(),
                report_title.strip(),
                examiner_name.strip() if examiner_name else None,
                laboratory_name.strip() if laboratory_name else None,
                examiner_observations.strip() if examiner_observations else None,
                report_status.strip().upper(),
                int(revision_number),
                source_measurement_ids,
                source_image_ids,
                image_hashes_json,
                filename.strip(),
                filepath.strip(),
                int(file_size),
                sha256_hash.strip().lower(),
            ),
        )
        act_type = "Report Revision" if int(revision_number) > 1 else "Report Generation"
        log_audit_event(
            action_type=act_type,
            entity_type="report",
            entity_id=report_id.strip(),
            case_id=case_id,
            details=f"Type={report_type.strip()}, Rev={int(revision_number)}, Status={report_status.strip().upper()}, SHA256={sha256_hash.strip().lower()[:16]}...",
            conn=conn,
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return get_report_by_id(report_id)


def get_report_by_id(report_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a single forensic report record by its unique report_id."""
    if not report_id or not str(report_id).strip():
        return None
    conn = get_db_connection()
    cursor = conn.execute(
        """
        SELECT r.*, c.title AS case_title, c.reference_no AS case_reference_no
        FROM reports r
        LEFT JOIN cases c ON r.case_id = c.case_id
        WHERE r.report_id = ?;
        """,
        (str(report_id).strip(),),
    )
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_reports(
    case_id: Optional[str] = None,
    report_type: Optional[str] = None,
    report_status: Optional[str] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """Retrieve generated forensic PDF reports ordered newest first with optional filters."""
    conn = get_db_connection()
    query = """
        SELECT r.*, c.title AS case_title, c.reference_no AS case_reference_no
        FROM reports r
        LEFT JOIN cases c ON r.case_id = c.case_id
        WHERE 1=1
    """
    params: List[Any] = []

    if case_id and case_id != "All":
        query += " AND r.case_id = ?"
        params.append(case_id.strip())
    if report_type and report_type != "All":
        query += " AND r.report_type = ?"
        params.append(report_type.strip())
    if report_status and report_status != "All":
        query += " AND r.report_status = ?"
        params.append(report_status.strip().upper())

    query += " ORDER BY r.generated_at DESC, r.id DESC LIMIT ?;"
    params.append(int(limit))

    cursor = conn.execute(query, params)
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


# ===========================================================================
# Phase 5.1: Pattern Analysis & Decision Support Persistence & Audit Logging
# ===========================================================================


def get_next_pattern_analysis_revision(
    case_id: str,
    image_id: Optional[str] = None,
) -> int:
    """
    Determine the next sequential revision number (1, 2, 3...) for a pattern analysis
    within a case (and optional image_id). Never overwrites prior examinations.
    """
    conn = get_db_connection()
    if image_id and str(image_id).strip():
        cursor = conn.execute(
            "SELECT MAX(revision_number) FROM pattern_analyses WHERE case_id = ? AND image_id = ?;",
            (case_id.strip(), str(image_id).strip()),
        )
    else:
        cursor = conn.execute(
            "SELECT MAX(revision_number) FROM pattern_analyses WHERE case_id = ? AND image_id IS NULL;",
            (case_id.strip(),),
        )
    row = cursor.fetchone()
    conn.close()
    max_rev = row[0] if row and row[0] is not None else 0
    return int(max_rev) + 1


def generate_pattern_analysis_id(revision_number: int = 1) -> str:
    """Generate a unique pattern analysis ID in the format PAT-YYYY-NNNN-R<rev>."""
    conn = get_db_connection()
    current_year = datetime.now().year
    while True:
        num = random.randint(1000, 9999)
        candidate = f"PAT-{current_year}-{num}-R{int(revision_number)}"
        cursor = conn.execute("SELECT 1 FROM pattern_analyses WHERE analysis_id = ?;", (candidate,))
        if not cursor.fetchone():
            conn.close()
            return candidate


def create_pattern_analysis_record(
    case_id: str,
    features_json: str,
    suggestions_json: str,
    observations_json: str,
    overall_determination: str,
    software_version: str = "BPMS Pattern Analysis Decision-Support Engine v5.1",
    evidence_id: Optional[str] = None,
    image_id: Optional[str] = None,
    is_calibrated: bool = False,
    mm_per_pixel: Optional[float] = None,
    suggestion_decision: str = "Pending Examiner Review",
    examiner_interpretation: Optional[str] = None,
    examiner_rationale: Optional[str] = None,
    examiner_name: Optional[str] = None,
    review_status: str = "Preliminary — Examiner Review Required",
    analysis_id: Optional[str] = None,
    revision_number: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Insert a new Bloodstain Pattern Analysis record into SQLite without overwriting
    any previous examination, and record the required forensic audit trail entries:
    - Pattern Analysis Creation (or Analysis Revision if revision_number > 1)
    - Feature Extraction
    - Classification Suggestion Generation
    - Examiner Interpretation Modification / Analysis Approval (if supplied)
    """
    rev = (
        int(revision_number)
        if revision_number is not None
        else get_next_pattern_analysis_revision(case_id=case_id, image_id=image_id)
    )
    aid = analysis_id.strip() if analysis_id else generate_pattern_analysis_id(revision_number=rev)

    # Enforce rule: Never mark an analysis as examiner-approved automatically unless explicitly signed off
    eff_review_status = review_status.strip() if review_status else "Preliminary — Examiner Review Required"

    conn = get_db_connection()
    try:
        conn.execute(
            """
            INSERT INTO pattern_analyses (
                analysis_id, case_id, evidence_id, image_id, revision_number,
                software_version, is_calibrated, mm_per_pixel,
                features_json, suggestions_json, observations_json,
                overall_determination, suggestion_decision,
                examiner_interpretation, examiner_rationale, examiner_name,
                review_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                aid,
                case_id.strip(),
                evidence_id.strip() if evidence_id and str(evidence_id).strip() else None,
                image_id.strip() if image_id and str(image_id).strip() else None,
                rev,
                software_version.strip(),
                1 if is_calibrated else 0,
                float(mm_per_pixel) if (is_calibrated and mm_per_pixel is not None) else None,
                features_json,
                suggestions_json,
                observations_json,
                overall_determination.strip(),
                suggestion_decision.strip() if suggestion_decision else "Pending Examiner Review",
                examiner_interpretation.strip() if examiner_interpretation else None,
                examiner_rationale.strip() if examiner_rationale else None,
                examiner_name.strip() if examiner_name else None,
                eff_review_status,
            ),
        )

        # Log creation / revision
        create_action = "Analysis Revision" if rev > 1 else "Pattern Analysis Creation"
        log_audit_event(
            action_type=create_action,
            entity_type="pattern_analysis",
            entity_id=aid,
            case_id=case_id,
            details=f"Image={image_id or 'None'}, Rev=R{rev}, Calibrated={bool(is_calibrated)}",
            conn=conn,
        )
        log_audit_event(
            action_type="Feature Extraction",
            entity_type="pattern_analysis",
            entity_id=aid,
            case_id=case_id,
            details=f"Image={image_id or 'None'}, Unit={'mm' if is_calibrated else 'px'}",
            conn=conn,
        )
        log_audit_event(
            action_type="Classification Suggestion Generation",
            entity_type="pattern_analysis",
            entity_id=aid,
            case_id=case_id,
            details=f"Determination={overall_determination.strip()}",
            conn=conn,
        )
        if examiner_interpretation and str(examiner_interpretation).strip():
            log_audit_event(
                action_type="Examiner Interpretation Modification",
                entity_type="pattern_analysis",
                entity_id=aid,
                case_id=case_id,
                details=f"Decision={suggestion_decision}, Interpretation={examiner_interpretation.strip()[:60]}",
                conn=conn,
            )
        if eff_review_status in ("Examiner Approved", "Examiner Rejected"):
            log_audit_event(
                action_type="Analysis Approval",
                entity_type="pattern_analysis",
                entity_id=aid,
                case_id=case_id,
                details=f"Status={eff_review_status}, Decision={suggestion_decision}",
                conn=conn,
            )

        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    return get_pattern_analysis_by_id(aid)


def update_pattern_analysis_examiner_review(
    analysis_id: str,
    suggestion_decision: str,
    examiner_interpretation: str,
    examiner_rationale: str,
    examiner_name: Optional[str] = None,
    review_status: str = "Examiner Reviewed",
    observations_json: Optional[str] = None,
    suggestions_json: Optional[str] = None,
    overall_determination: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Record or update the examiner's independent observations, decision on software suggestions
    (Accepted / Modified / Rejected / Indeterminate), final interpretation, and review status.
    Logs 'Examiner Interpretation Modification' and (when approved/rejected) 'Analysis Approval'.
    """
    existing = get_pattern_analysis_by_id(analysis_id)
    if not existing:
        return None

    new_obs = observations_json if observations_json is not None else existing["observations_json"]
    new_sugg = suggestions_json if suggestions_json is not None else existing["suggestions_json"]
    new_det = overall_determination.strip() if overall_determination else existing["overall_determination"]
    new_dec = suggestion_decision.strip() if suggestion_decision else existing["suggestion_decision"]
    new_interp = examiner_interpretation.strip() if examiner_interpretation is not None else existing["examiner_interpretation"]
    new_rat = examiner_rationale.strip() if examiner_rationale is not None else existing["examiner_rationale"]
    new_exam = examiner_name.strip() if examiner_name is not None else existing["examiner_name"]
    new_stat = review_status.strip() if review_status else existing["review_status"]

    conn = get_db_connection()
    try:
        conn.execute(
            """
            UPDATE pattern_analyses
            SET observations_json = ?,
                suggestions_json = ?,
                overall_determination = ?,
                suggestion_decision = ?,
                examiner_interpretation = ?,
                examiner_rationale = ?,
                examiner_name = ?,
                review_status = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE analysis_id = ?;
            """,
            (
                new_obs,
                new_sugg,
                new_det,
                new_dec,
                new_interp,
                new_rat,
                new_exam,
                new_stat,
                analysis_id.strip(),
            ),
        )

        log_audit_event(
            action_type="Examiner Interpretation Modification",
            entity_type="pattern_analysis",
            entity_id=analysis_id.strip(),
            case_id=existing["case_id"],
            details=f"Decision={new_dec}, Interpretation={new_interp[:60] if new_interp else 'None'}",
            conn=conn,
        )
        if new_stat in ("Examiner Approved", "Examiner Rejected", "Reviewed — Indeterminate"):
            log_audit_event(
                action_type="Analysis Approval",
                entity_type="pattern_analysis",
                entity_id=analysis_id.strip(),
                case_id=existing["case_id"],
                details=f"ReviewStatus={new_stat}, Decision={new_dec}",
                conn=conn,
            )

        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    return get_pattern_analysis_by_id(analysis_id)


def get_pattern_analysis_by_id(analysis_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a single pattern analysis record by its unique analysis_id."""
    if not analysis_id or not str(analysis_id).strip():
        return None
    conn = get_db_connection()
    cursor = conn.execute(
        """
        SELECT pa.*, ei.original_filename AS image_filename, ei.filepath AS image_filepath
        FROM pattern_analyses pa
        LEFT JOIN evidence_images ei ON pa.image_id = ei.image_id
        WHERE pa.analysis_id = ?;
        """,
        (str(analysis_id).strip(),),
    )
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_pattern_analyses_for_case(case_id: str) -> List[Dict[str, Any]]:
    """Retrieve all pattern analyses for a given case_id ordered newest first."""
    if not case_id or not str(case_id).strip():
        return []
    conn = get_db_connection()
    cursor = conn.execute(
        """
        SELECT pa.*, ei.original_filename AS image_filename, ei.filepath AS image_filepath
        FROM pattern_analyses pa
        LEFT JOIN evidence_images ei ON pa.image_id = ei.image_id
        WHERE pa.case_id = ?
        ORDER BY pa.id DESC;
        """,
        (str(case_id).strip(),),
    )
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


# ===========================================================================
# Phase 5.2 — Overlapping & Complex Bloodstain Analysis CRUD & Audit
# ===========================================================================


def get_next_overlap_analysis_revision(case_id: str, image_id: Optional[str] = None) -> int:
    """Calculate the next revision number for an overlap analysis on a case/image."""
    conn = get_db_connection()
    if image_id and str(image_id).strip():
        row = conn.execute(
            """
            SELECT COALESCE(MAX(revision_number), 0) + 1 AS next_rev
            FROM overlap_analyses
            WHERE case_id = ? AND image_id = ?;
            """,
            (case_id.strip(), str(image_id).strip()),
        ).fetchone()
    else:
        row = conn.execute(
            """
            SELECT COALESCE(MAX(revision_number), 0) + 1 AS next_rev
            FROM overlap_analyses
            WHERE case_id = ?;
            """,
            (case_id.strip(),),
        ).fetchone()
    conn.close()
    return int(row["next_rev"]) if row else 1


def generate_overlap_analysis_id(revision_number: int = 1) -> str:
    """Generate a unique overlap analysis ID, e.g., OVR-2026-4812-R1."""
    year = datetime.now().year
    rand_part = random.randint(1000, 9999)
    return f"OVR-{year}-{rand_part}-R{revision_number}"


def create_overlap_analysis_record(
    case_id: str,
    evidence_id: Optional[str],
    image_id: Optional[str],
    algorithm_name: str,
    algorithm_version: str,
    is_calibrated: bool,
    mm_per_pixel: Optional[float],
    parameters_json: str,
    summary_json: str,
    regions_json: str,
    accepted_stains_json: Optional[str] = None,
    review_status: str = "Preliminary — Examiner Review Required",
    examiner_name: Optional[str] = None,
    examiner_notes: Optional[str] = None,
    analysis_id: Optional[str] = None,
    revision_number: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Persist a newly computed complex/overlapping stain analysis record and record audit logs:
    - Complex Image Analysis Created
    - Connected Regions Detected
    - Segmentation Generated
    """
    rev = (
        int(revision_number)
        if revision_number is not None
        else get_next_overlap_analysis_revision(case_id=case_id, image_id=image_id)
    )
    aid = analysis_id.strip() if analysis_id else generate_overlap_analysis_id(revision_number=rev)

    conn = get_db_connection()
    try:
        conn.execute(
            """
            INSERT INTO overlap_analyses (
                analysis_id, case_id, evidence_id, image_id, revision_number,
                algorithm_name, algorithm_version, is_calibrated, mm_per_pixel,
                parameters_json, summary_json, regions_json, accepted_stains_json,
                review_status, examiner_name, examiner_notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                aid,
                case_id.strip(),
                evidence_id.strip() if evidence_id and str(evidence_id).strip() else None,
                image_id.strip() if image_id and str(image_id).strip() else None,
                rev,
                algorithm_name.strip(),
                algorithm_version.strip(),
                1 if is_calibrated else 0,
                float(mm_per_pixel) if (is_calibrated and mm_per_pixel is not None) else None,
                parameters_json,
                summary_json,
                regions_json,
                accepted_stains_json,
                review_status.strip() if review_status else "Preliminary — Examiner Review Required",
                examiner_name.strip() if examiner_name else None,
                examiner_notes.strip() if examiner_notes else None,
            ),
        )

        create_action = "Segmentation Revision Saved" if rev > 1 else "Complex Image Analysis Created"
        log_audit_event(
            action_type=create_action,
            entity_type="overlap_analysis",
            entity_id=aid,
            case_id=case_id,
            details=f"Image={image_id or 'None'}, Rev=R{rev}, Calibrated={bool(is_calibrated)}",
            conn=conn,
        )
        log_audit_event(
            action_type="Connected Regions Detected",
            entity_type="overlap_analysis",
            entity_id=aid,
            case_id=case_id,
            details=f"Analysis={aid}, Image={image_id or 'None'}",
            conn=conn,
        )
        log_audit_event(
            action_type="Segmentation Generated",
            entity_type="overlap_analysis",
            entity_id=aid,
            case_id=case_id,
            details=f"Algorithm={algorithm_name} {algorithm_version}",
            conn=conn,
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    return get_overlap_analysis_by_id(aid)


def update_overlap_analysis_review(
    analysis_id: str,
    regions_json: str,
    accepted_stains_json: str,
    summary_json: Optional[str] = None,
    review_status: str = "Examiner Reviewed",
    examiner_name: Optional[str] = None,
    examiner_notes: Optional[str] = None,
    action_type: str = "Segmentation Accepted",
) -> Optional[Dict[str, Any]]:
    """
    Update examiner decisions on an existing overlap analysis and log audit events:
    - Segmentation Accepted
    - Segmentation Rejected
    - Region Marked Indeterminate
    - Measurements Updated Following Approved Segmentation
    """
    existing = get_overlap_analysis_by_id(analysis_id)
    if not existing:
        return None

    new_sum = summary_json if summary_json is not None else existing["summary_json"]
    new_name = examiner_name.strip() if examiner_name is not None else existing["examiner_name"]
    new_notes = examiner_notes.strip() if examiner_notes is not None else existing["examiner_notes"]

    conn = get_db_connection()
    try:
        conn.execute(
            """
            UPDATE overlap_analyses
            SET regions_json = ?,
                accepted_stains_json = ?,
                summary_json = ?,
                review_status = ?,
                examiner_name = ?,
                examiner_notes = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE analysis_id = ?;
            """,
            (
                regions_json,
                accepted_stains_json,
                new_sum,
                review_status.strip(),
                new_name,
                new_notes,
                analysis_id.strip(),
            ),
        )

        log_audit_event(
            action_type=action_type,
            entity_type="overlap_analysis",
            entity_id=analysis_id.strip(),
            case_id=existing["case_id"],
            details=f"Status={review_status}, Examiner={new_name or 'Unknown'}",
            conn=conn,
        )
        if action_type == "Segmentation Accepted":
            log_audit_event(
                action_type="Measurements Updated Following Approved Segmentation",
                entity_type="overlap_analysis",
                entity_id=analysis_id.strip(),
                case_id=existing["case_id"],
                details=f"Approved individual child stains registered for {analysis_id.strip()}",
                conn=conn,
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    return get_overlap_analysis_by_id(analysis_id)


def get_overlap_analysis_by_id(analysis_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve an overlap analysis record by its unique analysis_id."""
    if not analysis_id or not str(analysis_id).strip():
        return None
    conn = get_db_connection()
    cursor = conn.execute(
        """
        SELECT oa.*, ei.original_filename AS image_filename, ei.filepath AS image_filepath
        FROM overlap_analyses oa
        LEFT JOIN evidence_images ei ON oa.image_id = ei.image_id
        WHERE oa.analysis_id = ?;
        """,
        (str(analysis_id).strip(),),
    )
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_overlap_analyses_for_case(case_id: str) -> List[Dict[str, Any]]:
    """Retrieve all overlap analyses for a given case_id ordered newest first."""
    if not case_id or not str(case_id).strip():
        return []
    conn = get_db_connection()
    cursor = conn.execute(
        """
        SELECT oa.*, ei.original_filename AS image_filename, ei.filepath AS image_filepath
        FROM overlap_analyses oa
        LEFT JOIN evidence_images ei ON oa.image_id = ei.image_id
        WHERE oa.case_id = ?
        ORDER BY oa.id DESC;
        """,
        (str(case_id).strip(),),
    )
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


# ===========================================================================
# Phase 5.3: Advanced Uncertainty & Sensitivity Analysis Records
# ===========================================================================


def get_next_uncertainty_analysis_revision(case_id: str, base_id: Optional[str] = None) -> int:
    """Calculate the next non-destructive revision number for an uncertainty analysis."""
    conn = get_db_connection()
    if base_id and str(base_id).strip():
        prefix = str(base_id).strip().split("-R")[0]
        cursor = conn.execute(
            "SELECT MAX(revision_number) FROM uncertainty_analyses WHERE analysis_id LIKE ?;",
            (f"{prefix}-R%",),
        )
    else:
        cursor = conn.execute(
            "SELECT MAX(revision_number) FROM uncertainty_analyses WHERE case_id = ?;",
            (case_id,),
        )
    row = cursor.fetchone()
    conn.close()
    curr = row[0] if (row and row[0] is not None) else 0
    return int(curr) + 1


def generate_uncertainty_analysis_id(case_id: str, revision: int = 1) -> str:
    """Generate unique ID: UNC-YYYY-NNNN-R<rev>."""
    year = datetime.now().year
    rand_num = random.randint(1000, 9999)
    return f"UNC-{year}-{rand_num:04d}-R{int(revision)}"


def create_uncertainty_analysis_record(
    case_id: str,
    width: float,
    length: float,
    analysis_data: Dict[str, Any],
    stain_id: Optional[int] = None,
    stain_label: Optional[str] = None,
    width_uncertainty: Optional[float] = None,
    length_uncertainty: Optional[float] = None,
    correlation: float = 0.0,
    scale_factor: float = 1.0,
    scale_uncertainty: float = 0.0,
    uncertainty_source: str = "Automated Estimation",
    is_manual: bool = False,
    iterations: int = 10000,
    random_seed: int = 42,
    distribution: str = "normal",
    perturbation_percentage: float = 5.0,
    parent_analysis_id: Optional[str] = None,
    examiner_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Insert a new Phase 5.3 uncertainty analysis record with audit log tracking."""
    conn = get_db_connection()
    try:
        rev = get_next_uncertainty_analysis_revision(case_id, base_id=parent_analysis_id)
        if parent_analysis_id and "-R" in parent_analysis_id:
            base_prefix = parent_analysis_id.split("-R")[0]
            analysis_id = f"{base_prefix}-R{rev}"
        else:
            analysis_id = generate_uncertainty_analysis_id(case_id, revision=rev)

        label = stain_label.strip() if stain_label else "Stain-01"
        data_json = json.dumps(analysis_data)

        conn.execute(
            """
            INSERT INTO uncertainty_analyses (
                analysis_id, case_id, stain_id, stain_label, revision_number,
                parent_analysis_id, width, length, width_uncertainty, length_uncertainty,
                correlation, scale_factor, scale_uncertainty, uncertainty_source,
                is_manual, iterations, random_seed, distribution,
                perturbation_percentage, analysis_data_json, review_status,
                examiner_name
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                analysis_id,
                case_id,
                stain_id,
                label,
                rev,
                parent_analysis_id,
                float(width),
                float(length),
                float(width_uncertainty) if width_uncertainty is not None else None,
                float(length_uncertainty) if length_uncertainty is not None else None,
                float(correlation),
                float(scale_factor),
                float(scale_uncertainty),
                str(uncertainty_source),
                1 if is_manual else 0,
                int(iterations),
                int(random_seed) if random_seed is not None else 42,
                str(distribution),
                float(perturbation_percentage),
                data_json,
                "Preliminary — Examiner Review Required",
                examiner_name,
            ),
        )

        log_audit_event(
            action_type="UNCERTAINTY_ANALYSIS_EXECUTED",
            entity_type="uncertainty_analysis",
            entity_id=analysis_id,
            case_id=case_id,
            details=f"Uncertainty analysis created for {label} (W={width}, L={length}, rev=R{rev})",
            conn=conn,
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    res = get_uncertainty_analysis_by_id(analysis_id)
    if not res:
        raise RuntimeError(f"Failed to retrieve newly created uncertainty analysis '{analysis_id}'.")
    return res


def update_uncertainty_analysis_review(
    analysis_id: str,
    review_status: str,
    examiner_name: str,
    examiner_notes: str = "",
) -> Dict[str, Any]:
    """Record forensic examiner review sign-off on an uncertainty analysis."""
    if not analysis_id or not str(analysis_id).strip():
        raise ValueError("analysis_id is required.")

    existing = get_uncertainty_analysis_by_id(analysis_id)
    if not existing:
        raise ValueError(f"Uncertainty analysis '{analysis_id}' not found.")

    conn = get_db_connection()
    try:
        now_ts = datetime.now().isoformat()
        new_name = examiner_name.strip() if examiner_name else existing.get("examiner_name")

        conn.execute(
            """
            UPDATE uncertainty_analyses
            SET review_status = ?,
                examiner_name = ?,
                examiner_notes = ?,
                reviewed_at = ?
            WHERE analysis_id = ?;
            """,
            (review_status.strip(), new_name, examiner_notes.strip(), now_ts, analysis_id.strip()),
        )

        log_audit_event(
            action_type="UNCERTAINTY_REVIEW_SAVED",
            entity_type="uncertainty_analysis",
            entity_id=analysis_id.strip(),
            case_id=existing["case_id"],
            details=f"Review updated: Status={review_status}, Examiner={new_name or 'Unknown'}",
            conn=conn,
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    res = get_uncertainty_analysis_by_id(analysis_id)
    if not res:
        raise ValueError(f"Could not retrieve updated analysis '{analysis_id}'.")
    return res


def get_uncertainty_analysis_by_id(analysis_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve an uncertainty analysis record by analysis_id."""
    if not analysis_id or not str(analysis_id).strip():
        return None
    conn = get_db_connection()
    cursor = conn.execute(
        """
        SELECT ua.*, c.title AS case_title, c.case_id AS case_ref
        FROM uncertainty_analyses ua
        JOIN cases c ON ua.case_id = c.case_id
        WHERE ua.analysis_id = ?;
        """,
        (str(analysis_id).strip(),),
    )
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_uncertainty_analyses_for_case(case_id: str) -> List[Dict[str, Any]]:
    """Retrieve all uncertainty analyses for a given case_id ordered newest first."""
    if not case_id or not str(case_id).strip():
        return []
    conn = get_db_connection()
    cursor = conn.execute(
        """
        SELECT ua.*, c.title AS case_title
        FROM uncertainty_analyses ua
        JOIN cases c ON ua.case_id = c.case_id
        WHERE ua.case_id = ?
        ORDER BY ua.id DESC;
        """,
        (str(case_id).strip(),),
    )
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def get_uncertainty_analyses_for_stain(stain_id: int) -> List[Dict[str, Any]]:
    """Retrieve all uncertainty analyses for a specific stain/measurement record."""
    if not stain_id:
        return []
    conn = get_db_connection()
    cursor = conn.execute(
        """
        SELECT ua.*, c.title AS case_title
        FROM uncertainty_analyses ua
        JOIN cases c ON ua.case_id = c.case_id
        WHERE ua.stain_id = ?
        ORDER BY ua.id DESC;
        """,
        (int(stain_id),),
    )
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows





