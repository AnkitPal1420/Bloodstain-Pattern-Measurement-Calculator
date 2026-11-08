"""
database.py
===========
SQLite data layer and persistence models for the Forensic Case Management system.

This module provides data access for cases, evidence items, and stain measurement records.
Uses Python's standard library sqlite3 for robust, dependency-free persistence.
"""

import os
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

    # Indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_cases_case_id ON cases(case_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_cases_status ON cases(status);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_evidence_case_id ON evidence(case_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_measurements_case_id ON measurements(case_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_images_image_id ON evidence_images(image_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_images_case_id ON evidence_images(case_id);")

    conn.commit()
    conn.close()

    if seed_demo:
        seed_demo_data()


def clear_all_data() -> None:
    """Clear all records from measurements, evidence_images, evidence, and cases tables."""
    conn = get_db_connection()
    try:
        conn.execute("DELETE FROM measurements;")
        conn.execute("DELETE FROM evidence_images;")
        conn.execute("DELETE FROM evidence;")
        conn.execute("DELETE FROM cases;")
        conn.commit()
    finally:
        conn.close()


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
        conn.commit()
        return row_id
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
        conn.commit()
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
