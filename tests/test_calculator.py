"""
test_calculator.py
==================
Unit tests for the Bloodstain Pattern Measurement Calculator core engine.
Compatible with both pytest and Python's built-in unittest runner.
"""

import math
import unittest

from calculator import (
    InvalidMeasurementError,
    calculate_impact_angle,
    validate_measurements,
)


class TestValidation(unittest.TestCase):
    """Tests for measurement input validation."""

    def test_valid_numeric_inputs(self):
        """Valid positive floats where width <= length should pass."""
        w, l = validate_measurements(5.0, 10.0)
        self.assertEqual(w, 5.0)
        self.assertEqual(l, 10.0)

    def test_valid_string_numeric_inputs(self):
        """String numbers should be cleanly cast to floats."""
        w, l = validate_measurements("5.5", "11.0")
        self.assertEqual(w, 5.5)
        self.assertEqual(l, 11.0)

    def test_valid_equal_measurements(self):
        """Width equal to length (circular stain) should pass."""
        w, l = validate_measurements(10, 10)
        self.assertEqual(w, 10.0)
        self.assertEqual(l, 10.0)

    def test_empty_none_inputs(self):
        """None values should raise InvalidMeasurementError."""
        with self.assertRaises(InvalidMeasurementError) as ctx:
            validate_measurements(None, 10.0)
        self.assertIn("cannot be empty", str(ctx.exception))

        with self.assertRaises(InvalidMeasurementError) as ctx:
            validate_measurements(5.0, None)
        self.assertIn("cannot be empty", str(ctx.exception))

    def test_blank_string_inputs(self):
        """Empty or whitespace-only strings should raise InvalidMeasurementError."""
        with self.assertRaises(InvalidMeasurementError) as ctx:
            validate_measurements("", 10.0)
        self.assertIn("cannot be blank", str(ctx.exception))

        with self.assertRaises(InvalidMeasurementError) as ctx:
            validate_measurements("   ", 10.0)
        self.assertIn("cannot be blank", str(ctx.exception))

        with self.assertRaises(InvalidMeasurementError) as ctx:
            validate_measurements(5.0, "")
        self.assertIn("cannot be blank", str(ctx.exception))

    def test_non_numeric_strings(self):
        """Text strings or strings with units should raise InvalidMeasurementError."""
        with self.assertRaises(InvalidMeasurementError) as ctx:
            validate_measurements("five", 10.0)
        self.assertIn("Invalid numerical input", str(ctx.exception))

        with self.assertRaises(InvalidMeasurementError) as ctx:
            validate_measurements(5.0, "10mm")
        self.assertIn("Invalid numerical input", str(ctx.exception))

    def test_zero_measurements(self):
        """Zero values should raise InvalidMeasurementError."""
        with self.assertRaises(InvalidMeasurementError) as ctx:
            validate_measurements(0.0, 10.0)
        self.assertIn("Width cannot be zero", str(ctx.exception))

        with self.assertRaises(InvalidMeasurementError) as ctx:
            validate_measurements(5.0, 0.0)
        self.assertIn("Length cannot be zero", str(ctx.exception))

    def test_negative_measurements(self):
        """Negative values should raise InvalidMeasurementError."""
        with self.assertRaises(InvalidMeasurementError) as ctx:
            validate_measurements(-5.0, 10.0)
        self.assertIn("must be positive", str(ctx.exception))

        with self.assertRaises(InvalidMeasurementError) as ctx:
            validate_measurements(5.0, -10.0)
        self.assertIn("must be positive", str(ctx.exception))

    def test_width_greater_than_length(self):
        """Width greater than length violates ellipse geometry."""
        with self.assertRaises(InvalidMeasurementError) as ctx:
            validate_measurements(12.0, 10.0)
        self.assertIn("cannot be greater than length", str(ctx.exception))

    def test_nan_and_infinity(self):
        """NaN and inf values should raise InvalidMeasurementError."""
        with self.assertRaises(InvalidMeasurementError) as ctx:
            validate_measurements(float("nan"), 10.0)
        self.assertIn("cannot be NaN", str(ctx.exception))

        with self.assertRaises(InvalidMeasurementError) as ctx:
            validate_measurements(5.0, float("inf"))
        self.assertIn("cannot be infinite", str(ctx.exception))


class TestCalculation(unittest.TestCase):
    """Tests for angle calculation accuracy and output structure."""

    def test_thirty_degree_angle(self):
        """Width/Length = 0.5 corresponds to arcsin(0.5) = 30 degrees."""
        result = calculate_impact_angle(5.0, 10.0)
        self.assertEqual(result["width"], 5.0)
        self.assertEqual(result["length"], 10.0)
        self.assertEqual(result["ratio"], 0.5)
        self.assertEqual(result["angle_degrees"], 30.0)
        self.assertTrue(math.isclose(result["angle_radians"], math.pi / 6, rel_tol=1e-4))
        self.assertFalse(result["is_perpendicular"])

    def test_ninety_degree_perpendicular(self):
        """Width == Length corresponds to arcsin(1.0) = 90 degrees (circular stain)."""
        result = calculate_impact_angle(10.0, 10.0)
        self.assertEqual(result["ratio"], 1.0)
        self.assertEqual(result["angle_degrees"], 90.0)
        self.assertTrue(math.isclose(result["angle_radians"], math.pi / 2, rel_tol=1e-4))
        self.assertTrue(result["is_perpendicular"])

    def test_forty_five_degree_angle(self):
        """Width/Length = sqrt(2)/2 corresponds to arcsin(1/sqrt(2)) = 45 degrees."""
        w = math.sqrt(2)
        l = 2.0
        result = calculate_impact_angle(w, l)
        self.assertTrue(math.isclose(result["angle_degrees"], 45.0, abs_tol=0.01))

    def test_sixty_degree_angle(self):
        """Width/Length = sqrt(3)/2 corresponds to 60 degrees."""
        w = math.sqrt(3)
        l = 2.0
        result = calculate_impact_angle(w, l)
        self.assertTrue(math.isclose(result["angle_degrees"], 60.0, abs_tol=0.01))

    def test_result_dictionary_keys(self):
        """Result must contain all expected informative keys."""
        result = calculate_impact_angle(4.0, 8.0)
        expected_keys = {
            "width",
            "length",
            "ratio",
            "angle_degrees",
            "angle_radians",
            "is_perpendicular",
            "interpretation",
            "disclaimer",
        }
        self.assertTrue(expected_keys.issubset(result.keys()))
        self.assertIn("Educational Estimate", result["disclaimer"])


if __name__ == "__main__":
    unittest.main()
