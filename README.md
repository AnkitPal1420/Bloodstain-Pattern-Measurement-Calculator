# Bloodstain Pattern Measurement Calculator — Forensic Laboratory Suite

A scientific forensic analysis platform for **Bloodstain Pattern Analysis (BPA)**, combining analytical trigonometric mathematics, OpenCV digital optical contour & ellipse extraction, multi-stain 2D/3D trajectory reconstruction, interactive 3D visualization, and verifiable ReportLab forensic PDF report compilation.

---

## System Architecture

The application enforces strict separation of concerns across 9 specialized Python modules:

- **[`calculator.py`](calculator.py) (Phase 1 — Core Trigonometry & Validation):**
  Pure analytical implementation of the bloodstain impact angle equation $\alpha = \arcsin(W / L)$, returning radians, degrees, $W/L$ ratio, perpendicular-drop classification, and forensic morphological interpretation.
- **[`image_processing.py`](image_processing.py) (Phase 3 — OpenCV Optical Inspection):**
  Digital image validation, CLAHE contrast enhancement, Gaussian denoising, adaptive/Otsu thresholding, external contour segmentation, Fitzgibbon direct least-squares ellipse fitting (`cv2.fitEllipse`), metric scale calibration ($\text{mm}/\text{px}$), and directional arrow overlay rendering.
- **[`trajectory.py`](trajectory.py) (Phase 4.1 — 3D Trajectory & Convergence Mathematics):**
  - 2D Area of Convergence via weighted least-squares minimization of perpendicular ray distances ($I - \hat{\mathbf{u}}_i \hat{\mathbf{u}}_i^T$).
  - 3D Area of Origin via both the **Linear Stringing / Tangent Method** ($Z_i = D_i \tan\alpha_i$) and **3D Closest Point of Approach (CPA)**.
  - Orthogonal coordinate transformations ($R^T R = I, \det(R) = +1$) between Image ($x_{\text{px}}, y_{\text{px}}$), Receiving Surface ($X_s, Y_s, Z_s$), and 3D Laboratory ($X, Y, Z$) frames (Floor, North Wall, East Wall, and Inclined Plane).
  - First-order Jacobian uncertainty propagation ($\sigma_W, \sigma_L, \sigma_\gamma \to \sigma_\alpha, \sigma_{\text{origin}}$).
- **[`trajectory_visualization.py`](trajectory_visualization.py) (Phase 4.2 — 2D/3D Forensic Visualization):**
  Publication-grade dark-slate Matplotlib 2D convergence and 3D trajectory projections alongside JSON scene builders for the interactive Three.js WebGL/Canvas 3D laboratory viewer.
- **[`trajectory_integration.py`](trajectory_integration.py) (Phase 4.3 — Measurement-to-Trajectory Bridge):**
  Converts calibrated SQLite optical measurements and examiner-supplied directionality/surface coordinates into validated `StainTrajectoryInput` records, executes full 2D/3D reconstruction, and persists results.
- **[`report_generator.py`](report_generator.py) (Phase 4.4 — Forensic PDF Reporting Engine):**
  Flask-independent ReportLab A4 PDF compiler supporting three report types (**Individual Bloodstain Examination**, **Trajectory Reconstruction**, and **Complete Forensic Case Report**) with proportional image embedding, atomic file writing, revision control (`R1, R2, ...`), and SHA-256 integrity hashing.
- **[`pattern_analysis.py`](pattern_analysis.py) (Phase 5.1 — Bloodstain Pattern Classification & Decision Support):**
  Examiner-assisted morphological feature extraction (Circularity $4\pi A/P^2$, Aspect Ratio $L/W$, Solidity $A/A_{\text{hull}}$, Eccentricity $\sqrt{1-(W/L)^2}$, spatial linearity $R^2$, nearest-neighbor distances, and non-silent contour rejection logs) and an explainable rule-based decision-support engine across 15 BPA pattern categories in 4 groups.
- **[`database.py`](database.py) (Phase 2–5.1 — SQLite Persistence & Operational Audit Trail):**
  Parameterized SQLite data layer managing 10 relational tables (`cases`, `evidence`, `measurements`, `evidence_images`, `reconstructions`, `reconstruction_stains`, `reconstruction_results`, `reports`, `audit_logs`, and `pattern_analyses`) with `PRAGMA foreign_keys = ON` and automatic rollback safety.
- **[`app.py`](app.py) (Flask Web Laboratory Controller):**
  Routes and REST endpoints serving the Forensic Dashboard, Case & Evidence Registry, Optical Measurement Lab, Multi-Stain Reconstruction Workspace, Interactive 3D Trajectory Lab, Pattern Classification & Decision Support Lab (`/pattern-analysis/<case_id>`), and Forensic PDF Report Registry.

---

## Installation & Startup Instructions

### 1. Environment Setup (Windows PowerShell)
```powershell
python -m venv .venv314
.\.venv314\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Initialize Database & Run Application
```powershell
.\.venv314\Scripts\python.exe app.py
```
Then open **`http://127.0.0.1:5000`** in your web browser.

### 3. Run the Automated Verification Suite
```powershell
.\.venv314\Scripts\python.exe -m pytest -v
```

---

## Scientific Methodology & Limitations

1. **Impact Angle Geometry:**
   $$\alpha = \arcsin\left(\frac{W}{L}\right)$$
   Valid only for well-formed elliptical bloodstains on a macroscopically planar receiving surface where $0 < W \le L$ (excluding satellite spatter, spines, or post-impact flow distortion).
2. **180° Directional Ambiguity:**
   An ellipse contour defines its principal major axis modulo $180^\circ$ and cannot by itself distinguish forward vs. reverse travel along that axis. Direction of travel ($\gamma \in [0^\circ, 360^\circ)$) must be independently established by the forensic examiner from tail, spine, or satellite morphology.
3. **Straight-Line Stringing Approximation:**
   Linear back-projection neglects aerodynamic drag and gravitational parabolic drop ($g = 9.81\text{ m/s}^2$). Consequently, straight-line triangulation systematically **overestimates** the true height of origin ($Z_{\text{linear}} \ge Z_{\text{true}}$). Computed coordinates represent an **approximate region of origin**, not an exact mathematical point source.
4. **Digital Integrity vs. Chain of Custody:**
   SHA-256 cryptographic digests recorded in `reports` and `audit_logs` verify digital file integrity at the time of ingestion or PDF compilation; they do not by themselves establish physical chain of custody.

---

## Project Milestones

- [x] **Milestone 1:** Core Trigonometric Calculation Engine, Input Validation & CLI (`calculator.py`, `main.py`)
- [x] **Milestone 2:** Forensic Laboratory Web UI, Case & Evidence Management (`app.py`, `database.py`, `templates/`)
- [x] **Milestone 3:** OpenCV Optical Contour Detection, Least-Squares Ellipse Fitting & Metric Scale Calibration (`image_processing.py`)
- [x] **Milestone 4.1:** 3D Trajectory Reconstruction & Convergence Mathematics (`trajectory.py`)
- [x] **Milestone 4.2:** Interactive 3D Trajectory Laboratory & Scientific Visualization (`trajectory_visualization.py`)
- [x] **Milestone 4.3:** Measurement-to-Trajectory Integration & Multi-Stain Workspace (`trajectory_integration.py`)
- [x] **Milestone 4.4:** Professional Forensic PDF Report Generation (`report_generator.py`)
- [x] **Milestone 4.5:** Complete System Integration, Scientific Validation, Security Audit & Final Release (`tests/test_phase4_final_validation.py`)
- [x] **Milestone 5.1:** Advanced Bloodstain Pattern Classification & Analytical Decision Support (`pattern_analysis.py`, `templates/pattern_analysis.html`, `tests/test_phase5_1_pattern_analysis.py`)
