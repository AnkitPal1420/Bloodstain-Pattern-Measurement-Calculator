# PHASE 5.3 COMPLETION AND VALIDATION REPORT
## Advanced Uncertainty & Sensitivity Analysis
**Bloodstain Pattern Measurement Calculator — Forensic Laboratory Suite**

---

## 1. Project Overview

- **Project Path:** `C:\Users\ANKIT\.gemini\antigravity\scratch\bloodstain-pattern-measurement-calculator`
- **Development Environment:** Python 3.14.10, Flask 3.1.3, SQLite with WAL mode, OpenCV 5.0.0.93, NumPy 2.5.4, SciPy 1.18.1, Matplotlib 3.11.2, ReportLab 5.1.0
- **Git Branch:** `main`
- **Release Baseline:** Phase 5.2 (109 / 109 automated tests passing)

### Objective & Core Mandate
Phase 5.3 develops an advanced forensic module for **Uncertainty & Sensitivity Analysis** that quantifies measurement dispersion, propagates analytical error bounds through the non-linear arcsin impact angle transformation ($\theta = \arcsin(W/L)$), executes high-performance vectorized Monte Carlo simulations, and evaluates parameter sensitivity via normalized elasticity indices and tornado charts.

### Scientific & Forensic Principles Enforced
1. **Zero Artificial Alteration:** Never silently correct invalid measurements ($W > L$, $W \le 0$, $L \le 0$). Invalid inputs must raise explicit forensic validation errors.
2. **Explicit Missing Uncertainty Handling:** When measurement uncertainty is not provided, the system explicitly reports it as *"Unavailable / Not Provided"* rather than fabricating synthetic error numbers.
3. **Isotropic Calibration Cancellation:** When stain dimensions are scaled by an isotropic factor $s$ ($W_{\text{mm}} = s W_{\text{px}}$, $L_{\text{mm}} = s L_{\text{px}}$), the scale factor cancels identically in the dimensionless ratio $W/L$. Consequently, scale calibration uncertainty does **not** propagate into the calculated impact angle, although it is rigorously tracked for reported physical dimensions ($mm$).
4. **Empirical Simulation vs. Probability Distinction:** A Monte Carlo 95% simulation interval reflects parameter dispersion under chosen distributional assumptions; it is **never** presented as an absolute probability that a specific incident deposition event occurred.
5. **Non-Destructive Revision Tracking:** Historical analyses are preserved under immutable revision IDs (`UNC-YYYY-NNNN-R<rev>`) with certified examiner review sign-offs.

---

## 2. Implementation Details

### A. Modified & Created Files
1. **Created:** [`uncertainty_analysis.py`](file:///c:/Users/ANKIT/.gemini/antigravity/scratch/bloodstain-pattern-measurement-calculator/uncertainty_analysis.py) (875 lines)
   - Core mathematical engine containing input validation, exact partial derivatives, first-order Taylor expansion variance calculation, vectorized Monte Carlo sampling, OAT sensitivity perturbation, and base64 chart rendering.
2. **Modified:** [`database.py`](file:///c:/Users/ANKIT/.gemini/antigravity/scratch/bloodstain-pattern-measurement-calculator/database.py)
   - Created `uncertainty_analyses` SQLite table with indexes on `case_id`, `stain_id`, and `analysis_id`.
   - Added CRUD helper functions: `create_uncertainty_analysis_record`, `update_uncertainty_analysis_review`, `get_uncertainty_analysis_by_id`, `get_uncertainty_analyses_for_case`, `get_uncertainty_analyses_for_stain`, and `get_next_uncertainty_analysis_revision`.
   - Wired audit event logging for `UNCERTAINTY_ANALYSIS_EXECUTED` and `UNCERTAINTY_REVIEW_SAVED`.
3. **Modified:** [`app.py`](file:///c:/Users/ANKIT/.gemini/antigravity/scratch/bloodstain-pattern-measurement-calculator/app.py)
   - Added routes:
     - `GET /uncertainty-analysis` (Redirect to default or first active case)
     - `GET /uncertainty-analysis/<case_id>` (Primary interactive workspace)
     - `POST /uncertainty-analysis/<case_id>/analyze` (Pipeline execution)
     - `POST /uncertainty-analysis/<case_id>/review/<analysis_id>` (Examiner review sign-off with cross-case 403 authorization)
     - `GET /api/uncertainty-analysis/<analysis_id>` (Direct REST API)
4. **Created:** [`templates/uncertainty_analysis.html`](file:///c:/Users/ANKIT/.gemini/antigravity/scratch/bloodstain-pattern-measurement-calculator/templates/uncertainty_analysis.html) (479 lines)
   - Dark-slate forensic interface with 5 key sections: Measurement Summary (A), Analytical Propagation (B), Monte Carlo Simulation (C), Sensitivity & Tornado Chart (D), and Examiner Governance (E).
5. **Modified:** [`templates/base.html`](file:///c:/Users/ANKIT/.gemini/antigravity/scratch/bloodstain-pattern-measurement-calculator/templates/base.html) & [`templates/case_detail.html`](file:///c:/Users/ANKIT/.gemini/antigravity/scratch/bloodstain-pattern-measurement-calculator/templates/case_detail.html)
   - Added sidebar navigation link `Uncertainty & Sensitivity` and Case Detail action button `Uncertainty Lab`.
6. **Created:** [`tests/test_phase5_3_uncertainty_analysis.py`](file:///c:/Users/ANKIT/.gemini/antigravity/scratch/bloodstain-pattern-measurement-calculator/tests/test_phase5_3_uncertainty_analysis.py) (280 lines, 13 automated tests).
7. **Created:** [`scripts/validate_phase5_3_browser.py`](file:///c:/Users/ANKIT/.gemini/antigravity/scratch/bloodstain-pattern-measurement-calculator/scripts/validate_phase5_3_browser.py) (215 lines, live browser automation harness).
8. **Modified:** [`README.md`](file:///c:/Users/ANKIT/.gemini/antigravity/scratch/bloodstain-pattern-measurement-calculator/README.md) (Updated milestone checklist and scientific methodology documentation).

### B. Database Schema Additions
```sql
CREATE TABLE IF NOT EXISTS uncertainty_analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id TEXT UNIQUE NOT NULL,      -- UNC-YYYY-NNNN-R<rev>
    case_id TEXT NOT NULL,
    stain_id INTEGER,                     -- FK to measurements.id
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
CREATE INDEX IF NOT EXISTS idx_unc_case_id ON uncertainty_analyses(case_id);
CREATE INDEX IF NOT EXISTS idx_unc_stain_id ON uncertainty_analyses(stain_id);
CREATE INDEX IF NOT EXISTS idx_unc_analysis_id ON uncertainty_analyses(analysis_id);
```

---

## 3. Scientific Validation

### A. Mathematical Equations & Partial Derivatives
Given the right-triangle impact angle relationship:
$$\theta = \arcsin\left(\frac{W}{L}\right)$$

The exact first-order partial derivatives are:
$$\frac{\partial \theta}{\partial W} = \frac{1}{\sqrt{1 - (W/L)^2}} \cdot \frac{1}{L} = \frac{1}{\sqrt{L^2 - W^2}}$$
$$\frac{\partial \theta}{\partial L} = \frac{1}{\sqrt{1 - (W/L)^2}} \cdot \left(-\frac{W}{L^2}\right) = -\frac{W}{L \sqrt{L^2 - W^2}}$$

Analytical propagated variance including covariance $\operatorname{cov}(W, L) = \rho_{WL} \sigma_W \sigma_L$:
$$\sigma_\theta^2 = \left(\frac{\partial \theta}{\partial W}\right)^2 \sigma_W^2 + \left(\frac{\partial \theta}{\partial L}\right)^2 \sigma_L^2 + 2 \left(\frac{\partial \theta}{\partial W}\right) \left(\frac{\partial \theta}{\partial L}\right) \rho_{WL} \sigma_W \sigma_L$$
$$\sigma_{\theta,\text{deg}} = \sqrt{\sigma_\theta^2} \cdot \frac{180^\circ}{\pi}$$

### B. Normalized Sensitivity Indices (Elasticity Symmetry)
Normalized sensitivity index for input parameter $x$:
$$S_x = \frac{\partial \theta}{\partial x} \cdot \frac{x}{\theta}$$
$$S_W = \frac{W}{\theta \sqrt{L^2 - W^2}}, \quad S_L = -\frac{W}{\theta \sqrt{L^2 - W^2}} = -S_W$$
This demonstrates exact mathematical elasticity symmetry: a $+1\%$ change in Stain Width produces the exact same percentage increase in impact angle as a $-1\%$ change in Stain Length.

### C. Example Numerical Verification ($30.0^\circ$ Reference Case)
- **Inputs:** $W = 5.0\text{ mm}$, $L = 10.0\text{ mm}$, $\sigma_W = 0.10\text{ mm}$, $\sigma_L = 0.20\text{ mm}$, $\rho = 0.0$
- **Baseline Angle:** $\theta_0 = \arcsin(0.5) = 30.00^\circ$ ($0.5236\text{ rad}$)
- **Partial Derivatives:**
  $$\frac{\partial \theta}{\partial W} = \frac{1}{\sqrt{100 - 25}} = \frac{1}{\sqrt{75}} \approx 0.11547\text{ rad/mm}$$
  $$\frac{\partial \theta}{\partial L} = -\frac{5}{10 \sqrt{75}} \approx -0.057735\text{ rad/mm}$$
- **Propagated Standard Uncertainty:**
  $$\sigma_\theta^2 = (0.11547 \times 0.10)^2 + (-0.057735 \times 0.20)^2 = 1.333 \times 10^{-4} + 1.333 \times 10^{-4} = 2.667 \times 10^{-4}\text{ rad}^2$$
  $$\sigma_\theta = 0.01633\text{ rad} \approx 0.9356^\circ \approx 0.94^\circ$$
- **Analytical Bounds:**
  - $1\sigma$ (68.3%): $[29.06^\circ, 30.94^\circ]$
  - $2\sigma$ (95.4%): $[28.17^\circ, 31.83^\circ]$
- **Monte Carlo Empirical Simulation ($N = 10,000$ draws, seed $42$):**
  - Mean: $30.01^\circ$ (error $< 0.01^\circ$ from theoretical)
  - Standard Deviation: $0.96^\circ$
  - 95% Interval ($P_{2.5}$ to $P_{97.5}$): $[28.18^\circ, 31.95^\circ]$
  - Invalid draws: $0$ ($0.0\%$)

---

## 4. Automated Testing

The complete test suite was executed in `.venv314`:
- **Phase 5.3 Test Suite:** 13 / 13 passed in 14.30s
- **Full Project Regression Suite:** **122 / 122 passed in 150.87s (100% passing)**

```
============================= test session starts =============================
platform win32 -- Python 3.14.10, pytest-9.2.0, pluggy-1.7.0
rootdir: C:\Users\ANKIT\.gemini\antigravity\scratch\bloodstain-pattern-measurement-calculator
collected 122 items

tests/test_calculator.py .................                               [ 13%]
tests/test_database.py .................                                  [ 27%]
tests/test_image_processing.py .................                          [ 41%]
tests/test_phase4_final_validation.py ..................................  [ 69%]
tests/test_phase5_1_pattern_analysis.py .............                     [ 80%]
tests/test_phase5_2_overlap_analysis.py .............                     [ 90%]
tests/test_phase5_3_uncertainty_analysis.py .............                 [100%]

============================ 122 passed in 150.87s ============================
```

### Breakdown of Test Results:
- **Total Tests:** 122
- **Passed Tests:** 122 (100%)
- **Failed Tests:** 0
- **Skipped Tests:** 0
- **Regressions Detected:** None

---

## 5. Live Website Browser Validation

- **Application URL:** `http://127.0.0.1:5000`
- **Browser Executable:** `C:\Program Files\Google\Chrome\Application\chrome.exe` (Headless mode)
- **Validation Harness:** [`scripts/validate_phase5_3_browser.py`](file:///c:/Users/ANKIT/.gemini/antigravity/scratch/bloodstain-pattern-measurement-calculator/scripts/validate_phase5_3_browser.py)
- **Console & Backend Errors:** 0 JavaScript errors; 0 HTTP 500 errors in Flask logs.

### Step-by-Step Validation Summary:
1. **Dashboard & Navigation:** GET `/` returned HTTP 200; `Uncertainty & Sensitivity` sidebar link verified.
2. **Case Detail Action Button:** GET `/cases/BPA-2026-0101` verified `Uncertainty Lab` button.
3. **Primary Workspace:** GET `/uncertainty-analysis/BPA-2026-0101` verified dark-slate theme and governance notice.
4. **Execution Pipeline:** POST `/uncertainty-analysis/BPA-2026-0101/analyze` computed baseline angle $30.0^\circ \pm 0.94^\circ$, $10,000$ Monte Carlo iterations, and parameter sensitivity.
5. **Invalid Input Enforcement:** POST with $W=12.0, L=5.0$ triggered forensic error: *"Width (12.0) cannot exceed length (5.0)"*.
6. **Examiner Review Sign-off:** POST review recorded *"Examiner Approved"* by *"Senior Examiner J. Miller, CBPA"*.
7. **Regression Compatibility:** Verified Phase 5.2 `/overlap-analysis/BPA-2026-0101` loaded and operated without regression.
8. **Responsive Viewports:** Desktop ($1600 \times 1050$) and mobile ($400 \times 850$) layouts rendered responsively.

### Captured Validation Screenshots (`validation/phase5_3/`):
| Screenshot Filename | Target Viewport | File Size | Verified Content |
| :--- | :--- | :--- | :--- |
| [`01_uncertainty_dashboard.png`](file:///c:/Users/ANKIT/.gemini/antigravity/scratch/bloodstain-pattern-measurement-calculator/validation/phase5_3/01_uncertainty_dashboard.png) | 1600x1050 | 229,165 B | Full Uncertainty & Sensitivity Laboratory workspace with methodology notice |
| [`02_measurement_uncertainty_input.png`](file:///c:/Users/ANKIT/.gemini/antigravity/scratch/bloodstain-pattern-measurement-calculator/validation/phase5_3/02_measurement_uncertainty_input.png) | 1600x750 | 142,124 B | Measurement inputs, presets (Glancing, Oblique, Steep), and uncertainty source dropdown |
| [`03_impact_angle_uncertainty_results.png`](file:///c:/Users/ANKIT/.gemini/antigravity/scratch/bloodstain-pattern-measurement-calculator/validation/phase5_3/03_impact_angle_uncertainty_results.png) | 1600x950 | 189,599 B | Section A & B: Analytical Error Propagation ($30.0^\circ \pm 0.94^\circ$), 1σ and 2σ bounds, derivatives |
| [`04_monte_carlo_results.png`](file:///c:/Users/ANKIT/.gemini/antigravity/scratch/bloodstain-pattern-measurement-calculator/validation/phase5_3/04_monte_carlo_results.png) | 1600x1350 | 288,656 B | Section C: Monte Carlo statistics ($N=10,000$, mean $30.01^\circ$, $95\%$ interval $[28.18^\circ, 31.95^\circ]$) |
| [`05_monte_carlo_histogram.png`](file:///c:/Users/ANKIT/.gemini/antigravity/scratch/bloodstain-pattern-measurement-calculator/validation/phase5_3/05_monte_carlo_histogram.png) | 1600x1200 | 269,250 B | Section C: Rendered Matplotlib distribution histogram with mean and $95\%$ percentile lines |
| [`06_sensitivity_tornado_chart.png`](file:///c:/Users/ANKIT/.gemini/antigravity/scratch/bloodstain-pattern-measurement-calculator/validation/phase5_3/06_sensitivity_tornado_chart.png) | 1600x1500 | 302,354 B | Section D: Rendered Tornado Sensitivity Chart ($\pm 5\%$ perturbation) and elasticity rankings |
| [`07_invalid_input_handling.png`](file:///c:/Users/ANKIT/.gemini/antigravity/scratch/bloodstain-pattern-measurement-calculator/validation/phase5_3/07_invalid_input_handling.png) | 1600x900 | 169,240 B | Error alert: *"Measurement Uncertainty Validation Error: Width (12.0) cannot exceed length (5.0)"* |
| [`08_successful_saved_analysis.png`](file:///c:/Users/ANKIT/.gemini/antigravity/scratch/bloodstain-pattern-measurement-calculator/validation/phase5_3/08_successful_saved_analysis.png) | 1600x1100 | 248,898 B | Section E: Persistent examiner review sign-off with green status badge |
| [`09_previous_phase_compatibility.png`](file:///c:/Users/ANKIT/.gemini/antigravity/scratch/bloodstain-pattern-measurement-calculator/validation/phase5_3/09_previous_phase_compatibility.png) | 1600x1050 | 241,589 B | Phase 5.2 Overlap Analysis workspace demonstrating zero regression |
| [`10_mobile_responsive_view.png`](file:///c:/Users/ANKIT/.gemini/antigravity/scratch/bloodstain-pattern-measurement-calculator/validation/phase5_3/10_mobile_responsive_view.png) | 400x850 | 67,182 B | Mobile viewport validation confirming responsive single-column card layout |

---

## 6. Issues Discovered and Resolved

| Issue Identified | Root Cause | Resolution Implemented |
| :--- | :--- | :--- |
| `UncertaintyValidationError` exception caught by `except (ValueError, TypeError)` in `validate_uncertainty_inputs` | `UncertaintyValidationError` inherits from `ValueError`, causing inner negative-check exceptions to be intercepted by string conversion catch block. | Separated float type-casting into dedicated `try/except` block, and performed negative boundary checks on the resulting float value outside the catch block. |
| `sqlite3.OperationalError: no such column: c.case_title` in database retrieval | SQL query joined `cases c` referencing `c.case_title`, but the `cases` table column is named `title`. | Updated queries to `SELECT ua.*, c.title AS case_title ...` in `get_uncertainty_analysis_by_id` and related functions. |
| In HTML template assertion test, `&` entity caused substring mismatch | Jinja renders `&` in titles as `&amp;` for HTML safety. | Updated test assertion to match `&amp;` in raw HTML response. |

---

## 7. Final Assessment

- **Completed Functionality:**
  - Measurement uncertainty validation (length, width, scale, graticule/caliper source).
  - Exact first-order analytical error propagation for impact angle ($\theta = \arcsin(W/L)$).
  - Covariance and axis correlation ($\rho_{WL}$) handling.
  - Scale invariance principle mathematically and visually demonstrated.
  - High-speed vectorized Monte Carlo simulation with empirical 95% simulation intervals and invalid-draw tracking.
  - One-At-A-Time sensitivity perturbation, normalized elasticity index symmetry ($S_W = -S_L$), and tornado chart rendering.
  - Database schema, non-destructive revisioning, and audit logging.
  - Dark-slate responsive frontend dashboard and navigation integration.
  - 122/122 automated tests passing.
  - Live website headless browser validation with 10 real screenshots captured.
- **Scientific Limitations:**
  - Analytical Taylor series approximation degrades near $\theta \to 90^\circ$ ($W \to L$), where the derivative approaches infinity. In this region, Monte Carlo simulation provides the mathematically sound non-linear representation.
  - Straight-line stringing continues to neglect gravitational drop and aerodynamic drag, meaning impact angles determine an approximate area of origin rather than an exact focal point.
- **Acceptance Criteria:** **Phase 5.3 fully meets all functional, mathematical, and forensic acceptance criteria.**

---

## 8. Git Status

- **Current Branch:** `main`
- **Release Commit:** Ready for commit
- **Staged / Modified Files:**
  - `README.md`
  - `app.py`
  - `database.py`
  - `templates/base.html`
  - `templates/case_detail.html`
  - `uncertainty_analysis.py`
  - `templates/uncertainty_analysis.html`
  - `tests/test_phase5_3_uncertainty_analysis.py`
  - `scripts/validate_phase5_3_browser.py`
  - `validation/phase5_3/*`
  - `PHASE_5_3_COMPLETION_AND_VALIDATION_REPORT.md`
