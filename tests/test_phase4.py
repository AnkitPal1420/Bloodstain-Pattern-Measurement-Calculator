"""
test_phase4.py
==============
Comprehensive automated unit and numerical precision tests for Phase 4.1:
Trajectory Reconstruction Engine (trajectory.py).

Covers:
1. Impact angle & horizontal/vertical trajectory component decomposition
2. 3D direction vector generation & unit normalization
3. Coordinate transformations (Image <-> 2D Surface <-> 3D Laboratory Frame)
4. Single-stain trajectory reconstruction & uncertainty propagation
5. 2D Area of Convergence & 3D Area of Origin reconstruction
6. Parallel, collinear, and diverging trajectory handling
7. Invalid input handling, missing directionality safeguards, and numerical edge cases
"""

import math
import sys
import unittest

import numpy as np

import database
import trajectory
from trajectory import (
    StainTrajectoryInput,
    TrajectoryError,
    compute_direction_vector_3d,
    compute_trajectory_components,
    estimate_area_of_convergence_2d,
    estimate_area_of_origin_3d,
    image_to_surface_coordinates,
    intersect_two_2d_rays,
    normalize_angle_360,
    normalize_vector,
    propagate_uncertainty,
    reconstruct_single_stain_trajectory,
    resolve_opencv_ellipse_azimuth,
    surface_to_image_coordinates,
    surface_to_lab_coordinates,
)


class TestPhase4TrajectoryEngine(unittest.TestCase):
    """Test suite verifying Phase 4.1 mathematical, geometric, and forensic requirements."""

    def test_module_independence_from_flask(self):
        """Ensure trajectory.py is strictly decoupled from Flask."""
        with open(trajectory.__file__, "r", encoding="utf-8") as f:
            source = f.read()
        self.assertNotIn("import flask", source.lower())
        self.assertNotIn("from flask", source.lower())

    # -----------------------------------------------------------------------
    # 1. Impact Angle & Trajectory Components
    # -----------------------------------------------------------------------

    def test_trajectory_components_30_45_90_degrees(self):
        """Verify horizontal cos(alpha), vertical sin(alpha), and tangent height D*tan(alpha)."""
        # 30 degrees: W=5, L=10 -> sin(alpha)=0.5, cos(alpha)=sqrt(3)/2 ~ 0.866025
        comp30 = compute_trajectory_components(5.0, 10.0, planar_distance_mm=100.0)
        self.assertAlmostEqual(comp30["angle_degrees"], 30.0, places=2)
        self.assertAlmostEqual(comp30["vertical_component"], 0.5, places=5)
        self.assertAlmostEqual(comp30["horizontal_component"], math.sqrt(3) / 2.0, places=5)
        self.assertAlmostEqual(comp30["estimated_height_mm"], 100.0 * math.tan(math.radians(30.0)), places=3)
        self.assertFalse(comp30["is_perpendicular"])

        # 45 degrees: W = L / sqrt(2) -> tan(alpha) = 1.0 -> Z == D
        w45 = 10.0 * math.sin(math.radians(45.0))
        comp45 = compute_trajectory_components(w45, 10.0, planar_distance_mm=250.0)
        self.assertAlmostEqual(comp45["angle_degrees"], 45.0, places=2)
        self.assertAlmostEqual(comp45["tangent_slope"], 1.0, places=4)
        self.assertAlmostEqual(comp45["estimated_height_mm"], 250.0, places=2)

        # 90 degrees perpendicular: W=10, L=10 -> horiz=0, vert=1, tangent_slope=None
        comp90 = compute_trajectory_components(10.0, 10.0)
        self.assertTrue(comp90["is_perpendicular"])
        self.assertAlmostEqual(comp90["horizontal_component"], 0.0, places=6)
        self.assertAlmostEqual(comp90["vertical_component"], 1.0, places=6)
        self.assertIsNone(comp90["tangent_slope"])

        # Requesting tangent height at 90 degrees must raise TrajectoryError
        with self.assertRaises(TrajectoryError):
            compute_trajectory_components(10.0, 10.0, planar_distance_mm=100.0)

    # -----------------------------------------------------------------------
    # 2. Direction Vector Generation & Normalization
    # -----------------------------------------------------------------------

    def test_direction_vector_normalization_and_sense(self):
        """Verify 3D unit vector norm == 1.0 and forward vs back-projected components."""
        # Stain moving along +X (gamma = 0 deg) with 30 deg impact angle:
        # Back-projected source lies in -X direction (beta = 180 deg) and +Z (upward)
        back_vec = compute_direction_vector_3d(angle_degrees=30.0, gamma_deg=0.0, back_project=True)
        self.assertAlmostEqual(back_vec["norm"], 1.0, places=6)
        self.assertAlmostEqual(back_vec["reverse_azimuth_deg"], 180.0, places=4)
        self.assertAlmostEqual(back_vec["vx"], -math.cos(math.radians(30.0)), places=5)
        self.assertAlmostEqual(back_vec["vy"], 0.0, places=5)
        self.assertAlmostEqual(back_vec["vz"], 0.5, places=5)

        # Forward impact vector: +X and -Z
        fwd_vec = compute_direction_vector_3d(angle_degrees=30.0, gamma_deg=0.0, back_project=False)
        self.assertAlmostEqual(fwd_vec["norm"], 1.0, places=6)
        self.assertAlmostEqual(fwd_vec["vx"], math.cos(math.radians(30.0)), places=5)
        self.assertAlmostEqual(fwd_vec["vz"], -0.5, places=5)

    def test_missing_directionality_rejected_for_oblique_stain(self):
        """An oblique ellipse without resolved gamma_deg must raise TrajectoryError."""
        with self.assertRaises(TrajectoryError):
            compute_direction_vector_3d(angle_degrees=45.0, gamma_deg=None)

        # 90-degree circular stain does not require gamma_deg for its vertical normal vector
        perp_vec = compute_direction_vector_3d(angle_degrees=90.0, gamma_deg=None)
        self.assertAlmostEqual(perp_vec["vz"], 1.0, places=6)
        self.assertIsNone(perp_vec["gamma_deg"])

    def test_normalize_vector_validation(self):
        """Verify vector normalization handles 2D/3D vectors and rejects zero or NaN vectors."""
        v2 = normalize_vector([3.0, 4.0])
        self.assertAlmostEqual(float(np.linalg.norm(v2)), 1.0, places=7)
        self.assertAlmostEqual(v2[0], 0.6, places=6)
        self.assertAlmostEqual(v2[1], 0.8, places=6)

        with self.assertRaises(TrajectoryError):
            normalize_vector([0.0, 0.0, 0.0])

        with self.assertRaises(TrajectoryError):
            normalize_vector([1.0, float("nan"), 0.0])

    # -----------------------------------------------------------------------
    # 3. Coordinate System Transformations
    # -----------------------------------------------------------------------

    def test_image_and_surface_coordinate_roundtrip(self):
        """Verify invertible transformation between image pixels (u, v) and surface (X_s, Y_s) mm."""
        surf = image_to_surface_coordinates(
            u_px=200.0,
            v_px=100.0,
            mm_per_pixel=0.25,
            origin_u_px=0.0,
            origin_v_px=400.0,
            invert_y=True,
        )
        self.assertAlmostEqual(surf["surface_x_mm"], 50.0, places=4)
        self.assertAlmostEqual(surf["surface_y_mm"], 75.0, places=4)

        img_back = surface_to_image_coordinates(
            surface_x_mm=surf["surface_x_mm"],
            surface_y_mm=surf["surface_y_mm"],
            mm_per_pixel=0.25,
            origin_u_px=0.0,
            origin_v_px=400.0,
            invert_y=True,
        )
        self.assertAlmostEqual(img_back["u_px"], 200.0, places=4)
        self.assertAlmostEqual(img_back["v_px"], 100.0, places=4)

    def test_surface_to_lab_coordinates_floor_and_vertical_wall(self):
        """Verify 3D rigid-body transformation from local surface frame to global room frame."""
        # Vertical Wall at Y_lab = 1000 mm, with +X_s along +X_lab and +Y_s upward along +Z_lab
        # Normal e_z = e_x x e_y = [1,0,0] x [0,0,1] = [0, -1, 0] (pointing into room toward -Y_lab)
        res = surface_to_lab_coordinates(
            point_surface_mm=[200.0, 300.0, 50.0],
            vector_surface=[0.0, 0.0, 1.0],
            surface_origin_lab_mm=[0.0, 1000.0, 0.0],
            surface_axis_x_lab=[1.0, 0.0, 0.0],
            surface_axis_y_lab=[0.0, 0.0, 1.0],
        )
        pt_lab = res["point_lab_mm"]
        self.assertAlmostEqual(pt_lab[0], 200.0, places=4)
        self.assertAlmostEqual(pt_lab[1], 950.0, places=4)  # 1000 - 50 along normal
        self.assertAlmostEqual(pt_lab[2], 300.0, places=4)
        np.testing.assert_allclose(res["vector_lab"], [0.0, -1.0, 0.0], atol=1e-6)

        # Collinear basis axes must raise TrajectoryError
        with self.assertRaises(TrajectoryError):
            surface_to_lab_coordinates(
                point_surface_mm=[0.0, 0.0, 0.0],
                surface_axis_x_lab=[1.0, 0.0, 0.0],
                surface_axis_y_lab=[2.0, 0.0, 0.0],
            )

    def test_opencv_ellipse_azimuth_resolution(self):
        """Verify OpenCV angle conversion with explicit 180-degree sense toggle."""
        az_base = resolve_opencv_ellipse_azimuth(30.0, invert_y_axis=True, flip_sense_180=False)
        az_flip = resolve_opencv_ellipse_azimuth(30.0, invert_y_axis=True, flip_sense_180=True)
        self.assertAlmostEqual(az_base, 330.0, places=4)
        self.assertAlmostEqual(az_flip, 150.0, places=4)

    # -----------------------------------------------------------------------
    # 4. Trajectory Reconstruction & Synthetic 3D Origin Verification
    # -----------------------------------------------------------------------

    def test_synthetic_multi_stain_2d_convergence_and_3d_origin(self):
        """
        Reconstruct a known 3D blood source at (X=100, Y=150, Z=300) mm from 3 synthetic
        radial spatter stains on a horizontal surface (Z=0).
        """
        true_origin = np.array([100.0, 150.0, 300.0])
        stain_positions = [
            (400.0, 150.0),  # Stain 1: East of source (D = 300 mm, gamma = 0 deg)
            (100.0, 450.0),  # Stain 2: North of source (D = 300 mm, gamma = 90 deg)
            (300.0, 350.0),  # Stain 3: NE of source (dx = 200, dy = 200, gamma = 45 deg)
        ]

        stains = []
        for idx, (sx, sy) in enumerate(stain_positions, start=1):
            dx = sx - true_origin[0]
            dy = sy - true_origin[1]
            planar_dist = math.hypot(dx, dy)
            gamma_deg = normalize_angle_360(math.degrees(math.atan2(dy, dx)))
            alpha_rad = math.atan2(true_origin[2], planar_dist)
            length_mm = 10.0
            width_mm = length_mm * math.sin(alpha_rad)

            stains.append(
                StainTrajectoryInput(
                    stain_id=f"Stain-{idx:02d}",
                    width=width_mm,
                    length=length_mm,
                    surface_x_mm=sx,
                    surface_y_mm=sy,
                    gamma_deg=gamma_deg,
                )
            )

        origin_res = estimate_area_of_origin_3d(stains)
        conv_2d = origin_res["convergence_2d"]
        tan_origin = origin_res["tangent_method_origin"]
        cpa_origin = origin_res["skew_ray_cpa_origin"]

        self.assertAlmostEqual(conv_2d["convergence_x_mm"], 100.0, places=2)
        self.assertAlmostEqual(conv_2d["convergence_y_mm"], 150.0, places=2)
        self.assertAlmostEqual(conv_2d["rms_residual_mm"], 0.0, places=3)

        self.assertAlmostEqual(tan_origin["z_mean_mm"], 300.0, places=2)
        self.assertAlmostEqual(cpa_origin["x_mm"], 100.0, places=2)
        self.assertAlmostEqual(cpa_origin["y_mm"], 150.0, places=2)
        self.assertAlmostEqual(cpa_origin["z_mm"], 300.0, places=2)
        self.assertAlmostEqual(cpa_origin["rms_3d_miss_mm"], 0.0, places=3)

    # -----------------------------------------------------------------------
    # 5. Parallel, Diverging, and Invalid Geometries
    # -----------------------------------------------------------------------

    def test_parallel_and_diverging_trajectories_rejected(self):
        """Parallel or diverging trajectories must raise TrajectoryError."""
        # Parallel rays (both traveling along gamma = 0 deg)
        parallel_stains = [
            StainTrajectoryInput("S1", width=5.0, length=10.0, surface_x_mm=100.0, surface_y_mm=50.0, gamma_deg=0.0),
            StainTrajectoryInput("S2", width=5.0, length=10.0, surface_x_mm=100.0, surface_y_mm=150.0, gamma_deg=0.0),
        ]
        with self.assertRaises(TrajectoryError):
            estimate_area_of_convergence_2d(parallel_stains)

        # Pairwise parallel check
        with self.assertRaises(TrajectoryError):
            intersect_two_2d_rays([0.0, 0.0], [1.0, 0.0], [0.0, 10.0], [1.0, 0.0])

        # Diverging rays (back-projections point away from each other)
        diverging_stains = [
            StainTrajectoryInput("D1", width=5.0, length=10.0, surface_x_mm=0.0, surface_y_mm=0.0, gamma_deg=45.0),
            StainTrajectoryInput("D2", width=5.0, length=10.0, surface_x_mm=100.0, surface_y_mm=0.0, gamma_deg=135.0),
        ]
        with self.assertRaises(TrajectoryError):
            estimate_area_of_convergence_2d(diverging_stains, require_forward_rays=True)

    def test_uncertainty_propagation_and_edge_cases(self):
        """Verify uncertainty propagation for normal oblique stains and near-90° singularity."""
        unc30 = propagate_uncertainty(width=5.0, length=10.0, sigma_width=0.1, sigma_length=0.1, planar_distance_mm=200.0)
        self.assertGreater(unc30["sigma_angle_degrees"], 0.0)
        self.assertGreater(unc30["sigma_height_mm"], 0.0)
        self.assertFalse(unc30["near_perpendicular_singularity"])

        # Near-perpendicular stain (W == L) must not divide by zero
        unc90 = propagate_uncertainty(width=10.0, length=10.0, sigma_width=0.1, sigma_length=0.1)
        self.assertTrue(unc90["near_perpendicular_singularity"])
        self.assertTrue(math.isfinite(unc90["sigma_angle_degrees"]))

    def test_invalid_inputs_and_uncalibrated_rejection(self):
        """Verify invalid inputs (single stain, uncalibrated stain, 90° stain in 2D convergence) raise TrajectoryError."""
        s1 = StainTrajectoryInput("S1", width=5.0, length=10.0, surface_x_mm=100.0, surface_y_mm=50.0, gamma_deg=0.0)
        with self.assertRaises(TrajectoryError):
            estimate_area_of_convergence_2d([s1])

        uncal = StainTrajectoryInput(
            "S2", width=5.0, length=10.0, surface_x_mm=100.0, surface_y_mm=150.0, gamma_deg=45.0, is_calibrated=False
        )
        with self.assertRaises(TrajectoryError):
            estimate_area_of_convergence_2d([s1, uncal], require_calibrated=True)

        perp = StainTrajectoryInput(
            "S90", width=10.0, length=10.0, surface_x_mm=100.0, surface_y_mm=150.0, gamma_deg=45.0
        )
        with self.assertRaises(TrajectoryError):
            estimate_area_of_convergence_2d([s1, perp])

    def test_database_phase4_columns_backward_compatible(self):
        """Verify database.create_measurement stores optional Phase 4 directional/surface columns cleanly."""
        database.init_db()
        row_id = database.create_measurement(
            case_id=None,
            evidence_id=None,
            stain_label="Traj-Stain-01",
            width=5.0,
            length=10.0,
            ratio=0.5,
            angle_degrees=30.0,
            angle_radians=0.5236,
            is_perpendicular=False,
            interpretation="Oblique",
            direction_angle=135.0,
            surface_x_mm=240.0,
            surface_y_mm=180.0,
            surface_type="Vertical Wall",
        )
        self.assertIsNotNone(row_id)
        recent = database.get_recent_measurements(limit=5)
        matched = [m for m in recent if m["id"] == row_id]
        self.assertEqual(len(matched), 1)
        self.assertAlmostEqual(matched[0]["direction_angle"], 135.0, places=2)
        self.assertAlmostEqual(matched[0]["surface_x_mm"], 240.0, places=2)
        self.assertEqual(matched[0]["surface_type"], "Vertical Wall")


if __name__ == "__main__":
    unittest.main()

