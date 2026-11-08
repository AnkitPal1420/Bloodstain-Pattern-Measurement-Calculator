# Bloodstain Pattern Measurement Calculator 🩸

An educational, open-source forensic science application to calculate and visualize the approximate angle of impact of bloodstains using idealized elliptical geometry.

---

## 📌 Project Overview

In forensic Bloodstain Pattern Analysis (BPA), estimating the impact angle of individual spatter stains is a foundational step in reconstructing 3D points of origin (the position of the blood source in three-dimensional space).

When a spherical blood droplet strikes a surface obliquely, momentum elongates the stain into an ellipse along its path of travel:
- **Width ($W$)**: Minor axis of the ellipse (corresponds to the diameter of the droplet upon initial contact).
- **Length ($L$)**: Major axis of the ellipse (corresponds to the elongated distance travelled along the surface, excluding tails or spines).

Using basic right-triangle trigonometry, the impact angle ($\alpha$) is calculated as:

$$\sin(\alpha) = \frac{\text{Width}}{\text{Length}} \implies \alpha = \arcsin\left(\frac{\text{Width}}{\text{Length}}\right)$$

Where:
- $\alpha$ is the impact angle in degrees ($0^\circ < \alpha \le 90^\circ$).
- A circular stain ($W = L$) represents a perpendicular $90^\circ$ drop.
- As $\alpha$ approaches $0^\circ$, the stain becomes increasingly narrow and elongated.

---

## ⚠️ Educational & Forensic Disclaimer

> **Important**: This application is strictly an **educational tool** for students, educators, and introductory forensic science training. 
> 
> The calculations rely on idealized geometric models on smooth, non-porous surfaces. Real-world physical factors—such as impact velocity, droplet volume, surface tension, substrate porosity/texture, and secondary spatter spines—can alter stain morphology. **This software must not be used as evidence in forensic casework or legal proceedings without certified laboratory validation.**

---

## 📂 Project Structure

```text
bloodstain-pattern-measurement-calculator/
├── app.py                   # Flask interactive laboratory web application
├── calculator.py            # Core trigonometric calculation engine & validation
├── database.py              # SQLite data persistence layer (cases, evidence, measurements)
├── image_processing.py      # OpenCV computer vision module (Milestone 3)
├── report_generator.py      # ReportLab PDF report generation (Milestone 4)
├── database/                # SQLite persistent storage
│   └── forensic.db
├── templates/               # Flask Jinja2 HTML templates
│   ├── base.html            # Master layout with responsive sidebar navigation
│   ├── index.html           # Main Laboratory Dashboard
│   ├── cases.html           # Case Registry with search & status filters
│   ├── case_form.html       # Initiate New Forensic Case form
│   ├── case_detail.html     # Case Inspection & linked evidence view
│   ├── evidence.html        # Physical Evidence Vault
│   ├── evidence_form.html   # Log New Evidence Item form
│   ├── measurement.html     # Measurement Workspace (Optical staging & manual controls)
│   └── results.html         # Results & Trajectory Dashboard
├── static/                  # Static web assets
│   ├── css/style.css        # Dark navy/charcoal forensic laboratory stylesheet
│   ├── js/main.js           # Interactive client presets script & mobile menu
│   └── images/              # Static images directory
├── sample_images/           # Educational sample bloodstain images
│   └── README.md
├── assets/                  # Diagrams, figures, and UI assets
│   └── README.md
├── tests/                   # Automated test suite
│   ├── __init__.py
│   ├── test_calculator.py   # 15 Unit tests covering validation and math
│   ├── test_app.py          # 7 Smoke and integration tests for Flask app
│   ├── test_phase2.py       # 10 Tests for cases, evidence, and workspaces
│   └── test_phase3.py       # 11 Tests for OpenCV computer vision & spatial calibration
├── requirements.txt         # Project dependencies
├── README.md                # Comprehensive project documentation
├── .gitignore               # Ignored files (virtual environments, databases, caches)
└── LICENSE                  # MIT Open-Source License
```

---

## 🛠️ Technology Stack

- **Python 3.14+**: Core programming language.
- **Flask**: Lightweight WSGI web application framework.
- **NumPy & SciPy**: Scientific computing, numerical arrays & trigonometric calculations.
- **OpenCV (`opencv-python`) & Pillow**: Image preprocessing and ellipse contour detection (Milestone 2).
- **Matplotlib & Pandas**: Visualization, plotting, and structured data analysis.
- **ReportLab**: Structured PDF export (Milestone 3).
- **pytest / unittest**: Automated test coverage.

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have **Python 3.10 or higher** installed on your system.

### 2. Clone or Navigate to the Repository
```bash
cd bloodstain-pattern-measurement-calculator
```

### 3. Create and Activate a Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**On macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🧪 Running the Unit Tests

The test suite validates input sanitization, zero/negative handling, geometric bounds ($W \le L$), and angle conversions.

Run tests using Python's built-in `unittest` (no extra packages needed):
```powershell
python -m unittest discover -s tests
```

Or using `pytest`:
```powershell
pytest -v
```

---

## 🌐 Launching the Flask Web Application

Start the local web server:
```powershell
python app.py
```

Once started, open your web browser to `http://127.0.0.1:5000`.

---

## 🧭 Project Milestones

- [x] **Milestone 1: Core Engine & Validation**
  - Implement $\alpha = \arcsin(W/L)$ with degree conversion.
  - Strict input validation (empty values, negative numbers, zero values, $W > L$, non-numeric input).
  - Flask user interface with presets, responsive metrics, and scientific explanations.
  - Automated unit test suite with 100% core pass rate.
  - Clean repository structure, `.gitignore`, and MIT License.

- [x] **Milestone 2: Frontend, Dashboard & Case Management**
  - Dark navy/charcoal laboratory design system with subtle crimson accents.
  - SQLite persistent data layer (`database.py`) for cases, evidence items, and stain calculations.
  - Main Dashboard with real-time laboratory metrics, recent case activities, and quick actions.
  - Case Registry with unique Case ID generation (`BPA-YYYY-NNNN`), search, and status tracking.
  - Physical Evidence Vault for logging patterns (Impact Spatter, Cast-off, Transfer, etc.).
  - Measurement Workspace Interface with Phase 3 optical reticle staging and manual calculation mode.
  - Results Dashboard with verified measurement history and trajectory placeholders.
  - 32/32 automated tests passing with 100% pass rate.

- [x] **Milestone 3: Computer Vision Optical Analysis (OpenCV)**
  - Secure digital bloodstain photographic image acquisition (`uploads/original/`, `uploads/processed/`) and SQLite metadata tracking (`evidence_images`).
  - Grayscale conversion, Gaussian blur, Otsu global vs Adaptive Gaussian thresholding comparison, and morphological opening/closing cleanup.
  - Contour segmentation, candidate stain selection, and direct least-squares ellipse fitting (`cv2.fitEllipse`) with enforced $W \le L$.
  - Interactive 2-point spatial pixel-to-millimeter calibration and multi-layer visual inspection overlays.
  - 43/43 automated regression and computer vision tests passing with 100% pass rate.

- [ ] **Milestone 4: Trajectory Projection & PDF Reporting**
  - 2D Area of Convergence vector projection.
  - 3D Area of Origin spatial height ($Z$) triangulation.
  - Printable forensic case summary PDF reports via ReportLab.

---

## 🛡️ Security & Ethics
- Never commit private case files, credentials, or sensitive data.
- Only upload openly licensed or synthetic educational images.

## 📄 License
This project is licensed under the [MIT License](LICENSE).
