"""
Tests for the Model module.

This module contains comprehensive tests for the ElectricityConsumptionModel class
to ensure proper model training, evaluation, prediction, and persistence.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from src.model import ElectricityConsumptionModel
from src.data_generator import DataGenerator


class TestElectricityConsumptionModel:
    """Test cases for ElectricityConsumptionModel class."""

    def setup_method(self):
        """Set up test data for each test method."""
        self.generator = DataGenerator(seed=42)
        self.data = self.generator.generate_data(n_samples=1000)
        self.train_data, self.val_data, self.test_data = self.generator.split_data(
            self.data
        )

        self.model = ElectricityConsumptionModel()

    def test_initialization(self):
        """Test model initialization."""
        model = ElectricityConsumptionModel()

        assert model.model is None
        assert model.preprocessor is None
        assert model.feature_names is None
        assert not model.is_trained

    def test_prepare_features_valid_data(self):
        """Test feature preparation with valid data."""
        # Test with valid data
        valid_data = pd.DataFrame(
            {
                "temperature": [25.0, 30.0],
                "day_of_week": ["Monday", "Saturday"],
                "major_event": [0, 1],
            }
        )

        prepared_data = self.model.prepare_features(valid_data)

        assert isinstance(prepared_data, pd.DataFrame)
        assert list(prepared_data.columns) == [
            "temperature",
            "day_of_week",
            "major_event",
        ]
        assert len(prepared_data) == 2

    def test_prepare_features_missing_columns(self):
        """Test feature preparation with missing columns."""
        invalid_data = pd.DataFrame(
            {
                "temperature": [25.0],
                "day_of_week": ["Monday"],
                # Missing major_event column
            }
        )

        with pytest.raises(ValueError, match="Missing required columns"):
            self.model.prepare_features(invalid_data)

    def test_prepare_features_invalid_temperature(self):
        """Test feature preparation with invalid temperature values."""
        invalid_data = pd.DataFrame(
            {
                "temperature": [10.0, 40.0],  # Outside valid range
                "day_of_week": ["Monday", "Tuesday"],
                "major_event": [0, 0],
            }
        )

        with pytest.raises(ValueError, match="Temperature must be between 15 and 35"):
            self.model.prepare_features(invalid_data)

    def test_prepare_features_invalid_day_of_week(self):
        """Test feature preparation with invalid day of week values."""
        invalid_data = pd.DataFrame(
            {"temperature": [25.0], "day_of_week": ["InvalidDay"], "major_event": [0]}
        )

        with pytest.raises(ValueError, match="Day of week must be one of"):
            self.model.prepare_features(invalid_data)

    def test_prepare_features_invalid_major_event(self):
        """Test feature preparation with invalid major event values."""
        invalid_data = pd.DataFrame(
            {
                "temperature": [25.0],
                "day_of_week": ["Monday"],
                "major_event": [2],  # Invalid value
            }
        )

        with pytest.raises(ValueError, match="Major event must be 0 or 1"):
            self.model.prepare_features(invalid_data)

    def test_train_model(self):
        """Test model training."""
        X_train = self.train_data.drop("consumption_kwh", axis=1)
        y_train = self.train_data[["consumption_kwh"]]

        metrics = self.model.train(X_train, y_train)

        # Check that model is trained
        assert self.model.is_trained
        assert self.model.model is not None
        assert self.model.feature_names is not None

        # Check metrics structure
        expected_metrics = ["train_mse", "train_rmse", "train_mae", "train_r2"]
        assert all(metric in metrics for metric in expected_metrics)

        # Check metric values are reasonable
        assert metrics["train_mse"] > 0
        assert metrics["train_rmse"] > 0
        assert metrics["train_mae"] > 0
        assert 0 <= metrics["train_r2"] <= 1

    def test_evaluate_model_not_trained(self):
        """Test evaluation when model is not trained."""
        X_test = self.test_data.drop("consumption_kwh", axis=1)
        y_test = self.test_data[["consumption_kwh"]]

        with pytest.raises(ValueError, match="Model must be trained before evaluation"):
            self.model.evaluate(X_test, y_test)

    def test_evaluate_model(self):
        """Test model evaluation."""
        # Train model first
        X_train = self.train_data.drop("consumption_kwh", axis=1)
        y_train = self.train_data[["consumption_kwh"]]
        self.model.train(X_train, y_train)

        # Evaluate model
        X_test = self.test_data.drop("consumption_kwh", axis=1)
        y_test = self.test_data[["consumption_kwh"]]

        metrics = self.model.evaluate(X_test, y_test)

        # Check metrics structure
        expected_metrics = ["test_mse", "test_rmse", "test_mae", "test_r2"]
        assert all(metric in metrics for metric in expected_metrics)

        # Check metric values are reasonable
        assert metrics["test_mse"] > 0
        assert metrics["test_rmse"] > 0
        assert metrics["test_mae"] > 0
        assert 0 <= metrics["test_r2"] <= 1

    def test_predict_not_trained(self):
        """Test prediction when model is not trained."""
        with pytest.raises(
            ValueError, match="Model must be trained before making predictions"
        ):
            self.model.predict(25.0, "Monday", 0)

    def test_predict_valid_inputs(self):
        """Test prediction with valid inputs."""
        # Train model first
        X_train = self.train_data.drop("consumption_kwh", axis=1)
        y_train = self.train_data[["consumption_kwh"]]
        self.model.train(X_train, y_train)

        # Test prediction
        prediction = self.model.predict(25.0, "Monday", 0)

        assert isinstance(prediction, float)
        assert prediction >= 0  # Should be non-negative

    def test_predict_different_inputs(self):
        """Test prediction with different input combinations."""
        # Train model first
        X_train = self.train_data.drop("consumption_kwh", axis=1)
        y_train = self.train_data[["consumption_kwh"]]
        self.model.train(X_train, y_train)

        # Test different temperature values
        pred1 = self.model.predict(20.0, "Monday", 0)
        pred2 = self.model.predict(30.0, "Monday", 0)

        # Higher temperature should generally lead to higher consumption
        assert pred2 > pred1

        # Test different days
        pred3 = self.model.predict(25.0, "Saturday", 0)
        pred4 = self.model.predict(25.0, "Monday", 0)

        # Should be different (though not necessarily higher/lower due to randomness)
        assert pred3 != pred4

        # Test with and without major event
        pred5 = self.model.predict(25.0, "Monday", 1)
        pred6 = self.model.predict(25.0, "Monday", 0)

        # Major event should increase consumption
        assert pred5 > pred6

    def test_get_model_coefficients_not_trained(self):
        """Test getting coefficients when model is not trained."""
        with pytest.raises(
            ValueError, match="Model must be trained before accessing coefficients"
        ):
            self.model.get_model_coefficients()

    def test_get_model_coefficients(self):
        """Test getting model coefficients."""
        # Train model first
        X_train = self.train_data.drop("consumption_kwh", axis=1)
        y_train = self.train_data[["consumption_kwh"]]
        self.model.train(X_train, y_train)

        coefficients = self.model.get_model_coefficients()

        # Check structure
        assert "feature_names" in coefficients
        assert "coefficients" in coefficients
        assert "intercept" in coefficients

        # Check types
        assert isinstance(coefficients["feature_names"], list)
        assert isinstance(coefficients["coefficients"], list)
        assert isinstance(coefficients["intercept"], float)

        # Check lengths
        assert len(coefficients["feature_names"]) == len(coefficients["coefficients"])
        assert len(coefficients["feature_names"]) > 0

    def test_save_model_not_trained(self):
        """Test saving model when not trained."""
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp_file:
            filepath = tmp_file.name

        try:
            with pytest.raises(ValueError, match="Model must be trained before saving"):
                self.model.save_model(filepath)
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_save_and_load_model(self):
        """Test saving and loading model."""
        # Train model first
        X_train = self.train_data.drop("consumption_kwh", axis=1)
        y_train = self.train_data[["consumption_kwh"]]
        self.model.train(X_train, y_train)

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp_file:
            filepath = tmp_file.name

        try:
            # Save model
            self.model.save_model(filepath)
            assert os.path.exists(filepath)

            # Create new model and load
            new_model = ElectricityConsumptionModel()
            new_model.load_model(filepath)

            # Check that model is trained
            assert new_model.is_trained
            assert new_model.model is not None

            # Test prediction with loaded model
            original_pred = self.model.predict(25.0, "Monday", 0)
            loaded_pred = new_model.predict(25.0, "Monday", 0)

            # Predictions should be identical
            assert abs(original_pred - loaded_pred) < 1e-10

        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_load_model_file_not_found(self):
        """Test loading model from non-existent file."""
        with pytest.raises(FileNotFoundError):
            self.model.load_model("non_existent_file.joblib")

    def test_model_performance_reasonable(self):
        """Test that model performance is reasonable."""
        # Train model
        X_train = self.train_data.drop("consumption_kwh", axis=1)
        y_train = self.train_data[["consumption_kwh"]]
        train_metrics = self.model.train(X_train, y_train)

        # Evaluate model
        X_test = self.test_data.drop("consumption_kwh", axis=1)
        y_test = self.test_data[["consumption_kwh"]]
        test_metrics = self.model.evaluate(X_test, y_test)

        # R-squared should be reasonable (not too low, not perfect)
        assert 0.3 <= train_metrics["train_r2"] <= 0.995
        assert 0.3 <= test_metrics["test_r2"] <= 0.995

        # Test R-squared should not be much worse than train R-squared
        assert test_metrics["test_r2"] >= train_metrics["train_r2"] - 0.2

    def test_model_consistency(self):
        """Test that model predictions are consistent."""
        # Train model
        X_train = self.train_data.drop("consumption_kwh", axis=1)
        y_train = self.train_data[["consumption_kwh"]]
        self.model.train(X_train, y_train)

        # Make same prediction multiple times
        pred1 = self.model.predict(25.0, "Monday", 0)
        pred2 = self.model.predict(25.0, "Monday", 0)
        pred3 = self.model.predict(25.0, "Monday", 0)

        # All predictions should be identical
        assert abs(pred1 - pred2) < 1e-10
        assert abs(pred2 - pred3) < 1e-10

    def test_model_feature_importance(self):
        """Test that model captures feature importance correctly."""
        # Train model
        X_train = self.train_data.drop("consumption_kwh", axis=1)
        y_train = self.train_data[["consumption_kwh"]]
        self.model.train(X_train, y_train)

        coefficients = self.model.get_model_coefficients()

        # Temperature coefficient should be positive (higher temp = higher consumption)
        temp_idx = coefficients["feature_names"].index("temperature")
        assert coefficients["coefficients"][temp_idx] > 0

        # Major event coefficient should be positive (events increase consumption)
        event_idx = coefficients["feature_names"].index("major_event")
        assert coefficients["coefficients"][event_idx] > 0

    def test_model_with_extreme_values(self):
        """Test model behavior with extreme input values."""
        # Train model
        X_train = self.train_data.drop("consumption_kwh", axis=1)
        y_train = self.train_data[["consumption_kwh"]]
        self.model.train(X_train, y_train)

        # Test with minimum temperature
        min_pred = self.model.predict(15.0, "Monday", 0)
        assert min_pred >= 0

        # Test with maximum temperature
        max_pred = self.model.predict(35.0, "Monday", 0)
        assert max_pred >= 0

        # Test with major event
        event_pred = self.model.predict(25.0, "Monday", 1)
        assert event_pred >= 0
