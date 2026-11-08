"""
test_app.py
===========
Integration and smoke tests for the Flask web application interface.
Compatible with pytest and unittest.
"""

import sys
import unittest

from app import app


class TestFlaskApplication(unittest.TestCase):
    """Tests for Flask routes, template rendering, and error handling."""

    def setUp(self):
        """Set up the Flask test client."""
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_streamlit_not_imported(self):
        """Ensure Streamlit is completely removed from app module dependencies."""
        self.assertNotIn("streamlit", sys.modules)

    def test_homepage_get_status_200(self):
        """GET / should return HTTP 200 and render the homepage."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("Bloodstain Pattern Measurement Calculator", html)
        self.assertIn('name="width"', html)
        self.assertIn('name="length"', html)
        self.assertIn("Scientific Principles & Geometric Foundation", html)

    def test_calculation_post_valid_30_deg(self):
        """POST / with width=5.0, length=10.0 should compute 30.00 degrees."""
        response = self.client.post("/", data={"width": "5.0", "length": "10.0"})
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("30.00°", html)
        self.assertIn("0.5000", html)
        self.assertIn("0.5236", html)
        self.assertIn("Oblique Strike", html)

    def test_calculation_post_perpendicular_90_deg(self):
        """POST / with width=10.0, length=10.0 should compute 90.00 degrees (circular)."""
        response = self.client.post("/", data={"width": "10.0", "length": "10.0"})
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("90.00°", html)
        self.assertIn("1.0000", html)
        self.assertIn("Circular Stain", html)

    def test_calculation_post_invalid_width_exceeds_length(self):
        """POST / with width > length should render validation error."""
        response = self.client.post("/", data={"width": "15.0", "length": "10.0"})
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("Validation Error", html)
        self.assertIn("cannot be greater than length", html)

    def test_calculation_post_invalid_negative(self):
        """POST / with negative value should render validation error."""
        response = self.client.post("/", data={"width": "-5.0", "length": "10.0"})
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("Validation Error", html)
        self.assertIn("must be positive numbers", html)

    def test_calculation_post_invalid_non_numeric(self):
        """POST / with non-numeric text should handle error safely."""
        response = self.client.post("/", data={"width": "abc", "length": "10.0"})
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("Validation Error", html)


if __name__ == "__main__":
    unittest.main()
