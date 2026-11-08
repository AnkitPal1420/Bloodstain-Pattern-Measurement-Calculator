"""
test_phase4_visualization.py
============================
Automated test suite for Phase 4.2:
3D Trajectory Reconstruction & 2D Area of Convergence Visualization Engine.

Covers:
1. 2D Area of Convergence plot generation & PNG header validation
2. 3D Trajectory scene generation & PNG header validation
3. Correct coordinate plotting in surface and 3D laboratory frames
4. Multiple receiving surface orientations (Horizontal Floor, Vertical Wall, Inclined Surface)
5. Missing directionality rejection
6. Empty and invalid dataset handling
7. Uncertainty & residual telemetry verification
8. Flask /trajectory route integration (GET presets, POST multi-stain reconstruction, error handling)
"""

import math
import unittest

import numpy as np

from app import app
import database
from trajectory import StainTrajectoryInput, TrajectoryError
import trajectory_visualization
from trajectory_visualization import (
    build_trajectory_scene_data,
    get_surface_basis,
    render_2d_convergence_plot,
    render_3d_trajectory_plot,
)

PNG_MAGIC_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class TestPhase4VisualizationSuite(unittest.TestCase):
    """Test suite verifying Phase 4.2 2D/3D visualization outputs and Flask integration."""

    def setUp(self):
        """Configure Flask test client and sample synthetic radial stain dataset."""
        app.config["TESTING"] = True
        app.config["WTF_CSRF_ENABLED"] = False
        self.client = app.test_client()
        database.init_db()

        # Synthetic 3-stain pattern originating from true 3D source (X=100, Y=150, Z=300) mm
        self.sample_stains = [
            StainTrajectoryInput(
                stain_id="Stain-01",
                width=7.0710678,
                length=10.0,
                surface_x_mm=400.0,
                surface_y_mm=150.0,
                gamma_deg=0.0,
                sigma_gamma_deg=2.0,
            ),
            StainTrajectoryInput(
                stain_id="Stain-02",
                width=7.0710678,
                length=10.0,
                surface_x_mm=100.0,
                surface_y_mm=450.0,
                gamma_deg=90.0,
                sigma_gamma_deg=2.0,
            ),
            StainTrajectoryInput(
                stain_id="Stain-03",
                width=10.0 * math.sin(math.atan2(300.0, math.hypot(200.0, 200.0))),
                length=10.0,
                surface_x_mm=300.0,
                surface_y_mm=350.0,
                gamma_deg=45.0,
                sigma_gamma_deg=2.0,
            ),
        ]

    # -----------------------------------------------------------------------
    # 1. 2D Convergence Plot Generation & Output Validity
    # -----------------------------------------------------------------------

    def test_2d_convergence_plot_generation_and_png_validity(self):
        """Verify render_2d_convergence_plot produces valid PNG bytes, base64 URI, and scene metrics."""
        res = render_2d_convergence_plot(self.sample_stains, surface_type="Horizontal Floor")
        self.assertIn("png_bytes", res)
        self.assertIn("base64_png", res)
        self.assertIn("scene", res)

        self.assertTrue(res["png_bytes"].startswith(PNG_MAGIC_SIGNATURE))
        self.assertGreater(len(res["png_bytes"]), 5000)
        self.assertTrue(res["base64_png"].startswith("data:image/png;base64,"))

        conv = res["scene"]["convergence_2d"]
        self.assertAlmostEqual(conv["x_surface_mm"], 100.0, places=1)
        self.assertAlmostEqual(conv["y_surface_mm"], 150.0, places=1)
        self.assertAlmostEqual(conv["rms_residual_mm"], 0.0, places=2)
        self.assertGreater(conv["convergence_radius_mm"], 0.0)

    # -----------------------------------------------------------------------
    # 2. 3D Trajectory Scene Generation & Output Validity
    # -----------------------------------------------------------------------

    def test_3d_trajectory_plot_generation_and_png_validity(self):
        """Verify render_3d_trajectory_plot produces valid 3D PNG figure and origin coordinates."""
        res = render_3d_trajectory_plot(self.sample_stains, surface_type="Horizontal Floor")
        self.assertTrue(res["png_bytes"].startswith(PNG_MAGIC_SIGNATURE))
        self.assertGreater(len(res["png_bytes"]), 5000)
        self.assertTrue(res["base64_png"].startswith("data:image/png;base64,"))

        scene = res["scene"]
        self.assertEqual(scene["stain_count"], 3)
        self.assertAlmostEqual(scene["tangent_origin_3d"]["z_mean_surface_mm"], 300.0, places=1)
        self.assertAlmostEqual(scene["cpa_origin_3d"]["z_surface_mm"], 300.0, places=1)
        self.assertGreater(scene["tangent_origin_3d"]["z_combined_uncertainty_mm"], 0.0)

    # -----------------------------------------------------------------------
    # 3. Multiple Surface Orientations & Coordinate Transformations
    # -----------------------------------------------------------------------

    def test_multiple_surface_orientations_floor_wall_and_inclined(self):
        """Verify scene coordinates transform accurately across Floor, Vertical Wall, and Inclined planes."""
        # 1. Horizontal Floor: surface (X_s=100, Y_s=150, Z_s=300) -> Lab (100, 150, 300)
        floor_scene = build_trajectory_scene_data(self.sample_stains, surface_type="Horizontal Floor")
        np.testing.assert_allclose(
            floor_scene["cpa_origin_3d"]["point_lab_mm"],
            [100.0, 150.0, 300.0],
            atol=0.5,
        )

        # 2. Vertical Wall: axis_x=[1,0,0], axis_y=[0,0,-1], normal=[0,+1,0]
        # Surface (X_s=100, Y_s=150, Z_s=300) -> Lab X=100, Lab Y=+300 (normal), Lab Z=-150
        wall_scene = build_trajectory_scene_data(self.sample_stains, surface_type="Vertical Wall")
        np.testing.assert_allclose(
            wall_scene["cpa_origin_3d"]["point_lab_mm"],
            [100.0, 300.0, -150.0],
            atol=0.5,
        )

        # 3. Inclined Surface (45 degrees)
        inc_scene = build_trajectory_scene_data(
            self.sample_stains,
            surface_type="Inclined Surface (45°)",
            incline_deg=45.0,
        )
        self.assertIn("Inclined Surface", inc_scene["surface_basis"]["surface_type"])
        # Verify Lab coordinates differ from flat floor due to 45° rotation
        self.assertNotAlmostEqual(
            inc_scene["cpa_origin_3d"]["point_lab_mm"][2],
            floor_scene["cpa_origin_3d"]["point_lab_mm"][2],
            places=1,
        )

    # -----------------------------------------------------------------------
    # 4. Missing Directionality & Empty/Invalid Dataset Handling
    # -----------------------------------------------------------------------

    def test_missing_directionality_and_invalid_datasets_rejected(self):
        """Verify missing gamma_deg, empty lists, single stain, and invalid surface types raise TrajectoryError."""
        # Empty dataset
        with self.assertRaises(TrajectoryError):
            build_trajectory_scene_data([])

        # Single stain
        with self.assertRaises(TrajectoryError):
            build_trajectory_scene_data([self.sample_stains[0]])

        # Missing directionality (gamma_deg=None on oblique stain)
        no_dir_stains = [
            self.sample_stains[0],
            StainTrajectoryInput("S-NoDir", width=5.0, length=10.0, surface_x_mm=200.0, surface_y_mm=250.0, gamma_deg=None),
        ]
        with self.assertRaises(TrajectoryError):
            render_2d_convergence_plot(no_dir_stains)

        # Invalid surface type
        with self.assertRaises(TrajectoryError):
            get_surface_basis("Spherical Ceiling")

    # -----------------------------------------------------------------------
    # 5. Flask /trajectory Route Integration Tests
    # -----------------------------------------------------------------------

    def test_flask_trajectory_lab_get_and_presets(self):
        """Verify GET /trajectory renders 3D canvas, Matplotlib plots, and surface presets."""
        for preset in ("floor_radial", "vertical_wall", "inclined_45"):
            res = self.client.get(f"/trajectory?preset={preset}")
            self.assertEqual(res.status_code, 200)
            html = res.get_data(as_text=True)
            self.assertIn("3D Spatial Trajectory Reconstruction &amp; Convergence Laboratory", html)
            self.assertIn("trajectory3DCanvas", html)
            self.assertIn("data:image/png;base64,", html)
            self.assertIn("Straight-Line Stringing Approximation", html)

    def test_flask_trajectory_lab_post_valid_and_invalid_inputs(self):
        """Verify POST /trajectory reconstructs custom multi-stain inputs and handles parallel ray errors."""
        # Valid POST
        post_data = {
            "surface_type": "Horizontal Floor",
            "incline_deg": "",
            "stain_id": ["Test-S1", "Test-S2"],
            "width": ["5.0", "5.0"],
            "length": ["10.0", "10.0"],
            "surface_x_mm": ["300.0", "100.0"],
            "surface_y_mm": ["100.0", "300.0"],
            "gamma_deg": ["0.0", "90.0"],
            "sigma_gamma_deg": ["2.0", "2.0"],
        }
        res = self.client.post("/trajectory", data=post_data)
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)
        self.assertIn("Test-S1", html)
        self.assertIn("(100.0, 100.0) mm", html)

        # Invalid POST: parallel rays (both gamma = 0.0)
        parallel_post = dict(post_data)
        parallel_post["gamma_deg"] = ["0.0", "0.0"]
        err_res = self.client.post("/trajectory", data=parallel_post)
        self.assertEqual(err_res.status_code, 200)
        err_html = err_res.get_data(as_text=True)
        self.assertIn("Trajectory Reconstruction Validation Error", err_html)
        self.assertIn("parallel", err_html.lower())


if __name__ == "__main__":
    unittest.main()

