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
├── app.py                   # Streamlit interactive web application
├── calculator.py            # Core trigonometric calculation engine & validation
├── image_processing.py      # OpenCV computer vision module (Milestone 2)
├── report_generator.py      # ReportLab PDF report generation (Milestone 3)
├── sample_images/           # Educational sample bloodstain images
│   └── README.md
├── assets/                  # Diagrams, figures, and UI assets
│   └── README.md
├── tests/                   # Automated test suite
│   ├── __init__.py
│   └── test_calculator.py   # Unit tests covering validation and math
├── requirements.txt         # Project dependencies
├── README.md                # Comprehensive project documentation
├── .gitignore               # Ignored files (virtual environments, caches)
└── LICENSE                  # MIT Open-Source License
```

---

## 🛠️ Technology Stack

- **Python 3.10+**: Core programming language.
- **Streamlit**: Interactive web dashboard.
- **NumPy**: Numerical arrays and scientific math.
- **OpenCV (`opencv-python-headless`)**: Image preprocessing and ellipse contour detection (Milestone 2).
- **Matplotlib**: Ellipse fitting visualization and spatter graphs.
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

## 🌐 Launching the Streamlit Application

Start the local web server:
```powershell
streamlit run app.py
```

Once started, open your web browser to `http://localhost:8501`.

---

## 🧭 Project Milestones

- [x] **Milestone 1: Core Engine & Validation**
  - Implement $\alpha = \arcsin(W/L)$ with degree conversion.
  - Strict input validation (empty values, negative numbers, zero values, $W > L$, non-numeric input).
  - Streamlit user interface with presets, metrics, and scientific explanations.
  - Automated unit test suite with 100% core pass rate.
  - Clean repository structure, `.gitignore`, and MIT License.

- [ ] **Milestone 2: OpenCV Image Processing**
  - Digital bloodstain image upload.
  - Automatic binarization and morphological filtering.
  - Direct least-squares ellipse fitting (`cv2.fitEllipse`).
  - Automated extraction of minor and major axes.

- [ ] **Milestone 3: Visualization & PDF Reporting**
  - Matplotlib visual overlays of detected ellipse on the original stain.
  - Printable educational PDF report with case details, metrics, and disclaimers.

---

## 🛡️ Security & Ethics
- Never commit private case files, credentials, or sensitive data.
- Only upload openly licensed or synthetic educational images.

## 📄 License
This project is licensed under the [MIT License](LICENSE).
