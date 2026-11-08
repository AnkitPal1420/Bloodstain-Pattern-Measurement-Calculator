"""
test_phase3.py
==============
Comprehensive automated unit and integration tests for Phase 3:
OpenCV Computer Vision & Optical Image Processing Pipeline.

Covers:
- Subphase 3.1: Secure image acquisition, extension & content validation, path traversal prevention
- Subphase 3.2: Grayscale conversion, Gaussian blur, Otsu vs Adaptive thresholding
- Subphase 3.3: Morphological filtering, contour segmentation, noise filtering, candidate selection
- Subphase 3.4: Direct least-squares ellipse fitting (cv2.fitEllipse), minor <= major axis guarantee
- Subphase 3.5: Spatial pixel-to-millimeter calibration and physical unit conversion
- Subphase 3.6: End-to-end integration with Measurement Lab, Calculator Engine, and SQLite database
"""

import io
import os
import unittest

import cv2
import numpy as np

import database
import image_processing
from app import app
from image_processing import ImageProcessingError


class TestPhase3ComputerVisionSuite(unittest.TestCase):
    """Test suite verifying Phase 3 computer vision functions and Flask integration."""

    def setUp(self):
        """Initialize Flask test client and database before each test."""
        app.config["TESTING"] = True
        app.config["WTF_CSRF_ENABLED"] = False
        self.client = app.test_client()
        database.init_db(seed_demo=True)
        self.sample_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_images")

    # -----------------------------------------------------------------------
    # Subphase 3.1: Image Acquisition & Validation Unit Tests
    # -----------------------------------------------------------------------

    def test_allowed_file_extensions(self):
        """Verify allowed forensic image extensions (.jpg, .jpeg, .png)."""
        self.assertTrue(image_processing.is_allowed_file("stain_photo.png"))
        self.assertTrue(image_processing.is_allowed_file("SCENE_01.JPG"))
        self.assertTrue(image_processing.is_allowed_file("evidence.jpeg"))
        self.assertFalse(image_processing.is_allowed_file("malware.exe"))
        self.assertFalse(image_processing.is_allowed_file("script.py"))
        self.assertFalse(image_processing.is_allowed_file("no_extension"))
        self.assertFalse(image_processing.is_allowed_file(""))

    def test_load_and_validate_corrupted_or_empty_image(self):
        """Verify corrupted or empty byte streams raise ImageProcessingError."""
        with self.assertRaises(ImageProcessingError):
            image_processing.load_and_validate_image(b"")

        with self.assertRaises(ImageProcessingError):
            image_processing.load_and_validate_image(b"not-a-valid-image-header-12345")

        with self.assertRaises(ImageProcessingError):
            image_processing.load_and_validate_image("/nonexistent/path/missing.png")

    # -----------------------------------------------------------------------
    # Subphase 3.2: Preprocessing (Grayscale, Blur, Otsu & Adaptive)
    # -----------------------------------------------------------------------

    def test_preprocessing_grayscale_blur_and_thresholding(self):
        """Verify grayscale conversion, Gaussian blur, Otsu, and Adaptive thresholding."""
        sample_path = os.path.join(self.sample_dir, "sample_30deg_oblique.png")
        bgr = image_processing.load_and_validate_image(sample_path)
        self.assertEqual(len(bgr.shape), 3)

        gray = image_processing.convert_to_grayscale(bgr)
        self.assertEqual(len(gray.shape), 2)

        # Even kernel size should automatically normalize to odd
        blurred = image_processing.apply_gaussian_blur(gray, kernel_size=4)
        self.assertEqual(blurred.shape, gray.shape)

        otsu_mask = image_processing.apply_thresholding(blurred, method="otsu", invert=True)
        adaptive_mask = image_processing.apply_thresholding(blurred, method="adaptive", invert=True)

        self.assertEqual(otsu_mask.shape, gray.shape)
        self.assertEqual(adaptive_mask.shape, gray.shape)
        self.assertTrue(np.any(otsu_mask == 255))
        self.assertTrue(np.any(adaptive_mask == 255))

        with self.assertRaises(ImageProcessingError):
            image_processing.apply_thresholding(blurred, method="invalid_method")

    # -----------------------------------------------------------------------
    # Subphase 3.3 & 3.4: Segmentation, Contour Detection & Ellipse Fitting
    # -----------------------------------------------------------------------

    def test_contour_detection_and_ellipse_fitting_30deg(self):
        """Verify 30° synthetic stain yields W/L ratio ~ 0.50 and minor <= major axis."""
        sample_path = os.path.join(self.sample_dir, "sample_30deg_oblique.png")
        pipeline = image_processing.process_bloodstain_pipeline(
            image_source=sample_path,
            threshold_method="otsu",
            invert_binary=True,
        )

        self.assertGreaterEqual(len(pipeline["candidates"]), 1)
        ellipse = pipeline["ellipse"]
        self.assertLessEqual(ellipse["minor_axis_px"], ellipse["major_axis_px"])
        self.assertAlmostEqual(ellipse["ratio"], 0.50, delta=0.03)

    def test_contour_detection_and_ellipse_fitting_90deg(self):
        """Verify 90° circular stain yields W/L ratio ~ 1.0."""
        sample_path = os.path.join(self.sample_dir, "sample_90deg_circular.png")
        pipeline = image_processing.process_bloodstain_pipeline(
            image_source=sample_path,
            threshold_method="otsu",
            invert_binary=True,
        )

        ellipse = pipeline["ellipse"]
        self.assertLessEqual(ellipse["minor_axis_px"], ellipse["major_axis_px"])
        self.assertAlmostEqual(ellipse["ratio"], 1.0, delta=0.03)

    def test_ellipse_fitting_insufficient_points_error(self):
        """Verify contours with fewer than 5 points raise ImageProcessingError."""
        tiny_contour = np.array([[[10, 10]], [[12, 10]], [[12, 12]], [[10, 12]]], dtype=np.int32)
        with self.assertRaises(ImageProcessingError):
            image_processing.fit_stain_ellipse(tiny_contour)

    # -----------------------------------------------------------------------
    # Subphase 3.5: Spatial Pixel-to-Millimeter Calibration
    # -----------------------------------------------------------------------

    def test_spatial_calibration_math_and_validation(self):
        """Verify pixel-to-millimeter scale factor calculation and physical axis conversion."""
        cal = image_processing.calculate_scale_factor(
            point1=(60.0, 340.0),
            point2=(100.0, 340.0),
            known_length_mm=10.0,
        )
        self.assertAlmostEqual(cal["pixel_distance"], 40.0, places=2)
        self.assertAlmostEqual(cal["mm_per_pixel"], 0.25, places=4)
        self.assertAlmostEqual(cal["pixels_per_mm"], 4.0, places=4)
        self.assertIsNone(cal["warning"])

        converted = image_processing.apply_calibration(
            minor_axis_px=100.0,
            major_axis_px=200.0,
            mm_per_pixel=cal["mm_per_pixel"],
        )
        self.assertAlmostEqual(converted["width_mm"], 25.0, places=2)
        self.assertAlmostEqual(converted["length_mm"], 50.0, places=2)

        # Identical points must raise ImageProcessingError
        with self.assertRaises(ImageProcessingError):
            image_processing.calculate_scale_factor((10.0, 10.0), (10.0, 10.0), 10.0)

        # Non-positive known length must raise ImageProcessingError
        with self.assertRaises(ImageProcessingError):
            image_processing.calculate_scale_factor((10.0, 10.0), (50.0, 10.0), 0.0)

    # -----------------------------------------------------------------------
    # Subphase 3.6: Flask End-to-End Integration Tests
    # -----------------------------------------------------------------------

    def test_flask_upload_valid_image_and_vision_pipeline(self):
        """Upload a valid PNG image, run OpenCV vision pipeline, calibrate, and save measurement."""
        sample_path = os.path.join(self.sample_dir, "sample_45deg_calibrated.png")
        with open(sample_path, "rb") as f:
            img_bytes = f.read()

        data = {
            "case_id": "BPA-2026-0101",
            "evidence_id": "EVD-0101-01",
            "image_file": (io.BytesIO(img_bytes), "test_stain_45.png"),
        }
        res = self.client.post(
            "/measurement/upload",
            data=data,
            content_type="multipart/form-data",
            follow_redirects=True,
        )
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)
        self.assertIn("OpenCV Ellipse Fitted", html)
        self.assertIn("IMG-", html)

        # Retrieve the newly created image_id from database
        images = database.get_images_for_case("BPA-2026-0101")
        self.assertGreaterEqual(len(images), 1)
        image_id = images[0]["image_id"]

        # Test calibrating and saving the optical measurement
        save_res = self.client.post(
            "/measurement",
            data={
                "image_id": image_id,
                "case_id": "BPA-2026-0101",
                "evidence_id": "EVD-0101-01",
                "stain_label": "Optical-Stain-45",
                "threshold_method": "otsu",
                "blur_kernel": "5",
                "invert_binary": "true",
                "min_area": "30",
                "selected_candidate_id": "1",
                "cal_x1": "60",
                "cal_y1": "340",
                "cal_x2": "100",
                "cal_y2": "340",
                "known_length_mm": "10.0",
                "form_action": "save_vision",
                "save_record": "true",
            },
            follow_redirects=True,
        )
        self.assertEqual(save_res.status_code, 200)
        save_html = save_res.get_data(as_text=True)
        self.assertIn("saved to history", save_html)
        self.assertIn("Calibrated Optical Scale", save_html)

        # Verify record persisted in database with image_id and calibration status
        measurements = database.get_recent_measurements(limit=5, case_id="BPA-2026-0101")
        optical_records = [m for m in measurements if m["image_id"] == image_id]
        self.assertEqual(len(optical_records), 1)
        rec = optical_records[0]
        self.assertEqual(rec["image_id"], image_id)
        self.assertIn("Calibrated", rec["calibration_status"])
        self.assertAlmostEqual(rec["angle_degrees"], 45.0, delta=2.0)

    def test_flask_upload_rejects_invalid_extension_and_corrupted_bytes(self):
        """Verify upload route rejects unsupported extensions and corrupted image content."""
        # 1. Invalid extension
        bad_ext_res = self.client.post(
            "/measurement/upload",
            data={"image_file": (io.BytesIO(b"hello"), "exploit.txt")},
            content_type="multipart/form-data",
            headers={"Accept": "application/json"},
        )
        self.assertEqual(bad_ext_res.status_code, 400)

        # 2. Valid extension but corrupted image bytes
        corrupt_res = self.client.post(
            "/measurement/upload",
            data={"image_file": (io.BytesIO(b"corrupted-pixel-stream"), "fake.png")},
            content_type="multipart/form-data",
            headers={"Accept": "application/json"},
        )
        self.assertEqual(corrupt_res.status_code, 400)

    def test_flask_sample_load_and_adaptive_thresholding(self):
        """Verify loading built-in sample and switching to Adaptive Gaussian thresholding."""
        res = self.client.post(
            "/measurement/upload",
            data={"sample_name": "sample_30deg_oblique.png"},
            headers={"Accept": "application/json"},
        )
        self.assertEqual(res.status_code, 201)
        payload = res.get_json()
        self.assertEqual(payload["status"], "ok")
        img_id = payload["image_id"]

        # Request Measurement Lab with adaptive thresholding
        adaptive_res = self.client.get(
            f"/measurement?image_id={img_id}&threshold_method=adaptive&blur_kernel=5"
        )
        self.assertEqual(adaptive_res.status_code, 200)
        html = adaptive_res.get_data(as_text=True)
        self.assertIn("ADAPTIVE Binary", html)
        self.assertIn("OpenCV Ellipse Fitted", html)

    def test_path_traversal_blocked_on_upload_routes(self):
        """Ensure path traversal attempts on /uploads/ endpoints return 404."""
        res1 = self.client.get("/uploads/original/../app.py")
        self.assertEqual(res1.status_code, 404)
        res2 = self.client.get("/uploads/processed/../../database/forensic.db")
        self.assertEqual(res2.status_code, 404)



if __name__ == "__main__":
    unittest.main()
