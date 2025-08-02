"""
Tests for the Gradio Application module.

This module contains tests for the ElectricityPredictorApp class to ensure
the web interface functions correctly.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from src.app import ElectricityPredictorApp


class TestElectricityPredictorApp:
    """Test cases for ElectricityPredictorApp class."""

    def setup_method(self):
        """Set up test app for each test method."""
        self.app = ElectricityPredictorApp()

    def test_initialization(self):
        """Test app initialization."""
        assert self.app.data_generator is not None
        assert self.app.model is not None
        assert not self.app.is_model_trained

    def test_generate_and_train_success(self):
        """Test successful data generation and training."""
        # Mock the data generator methods
        with patch.object(
            self.app.data_generator, "generate_data"
        ) as mock_generate, patch.object(
            self.app.data_generator, "split_data"
        ) as mock_split, patch.object(
            self.app.model, "train"
        ) as mock_train, patch.object(
            self.app.model, "evaluate"
        ) as mock_evaluate:

            # Create mock data
            mock_data = pd.DataFrame(
                {
                    "temperature": [25.0, 30.0],
                    "day_of_week": ["Monday", "Tuesday"],
                    "major_event": [0, 1],
                    "consumption_kwh": [15.0, 18.0],
                }
            )
            mock_generate.return_value = mock_data

            # Create mock split data
            train_data = mock_data.iloc[:1]
            val_data = mock_data.iloc[1:2]
            test_data = mock_data.iloc[1:2]
            mock_split.return_value = (train_data, val_data, test_data)

            mock_train.return_value = {
                "train_mse": 2.5,
                "train_rmse": 1.58,
                "train_mae": 1.2,
                "train_r2": 0.85,
            }
            mock_evaluate.return_value = {
                "test_mse": 2.8,
                "test_rmse": 1.67,
                "test_mae": 1.3,
                "test_r2": 0.82,
            }

            # Call the method
            data_info, training_metrics, evaluation_metrics = (
                self.app.generate_and_train(
                    n_samples=1000,
                    noise_level=0.1,
                    train_size=0.7,
                    val_size=0.15,
                    test_size=0.15,
                )
            )

            # Check that methods were called
            mock_generate.assert_called_once_with(1000, 0.1)
            mock_split.assert_called_once_with(mock_data, 0.7, 0.15, 0.15)
            mock_train.assert_called_once()
            mock_evaluate.assert_called_once()

            # Check that app state was updated
            assert self.app.is_model_trained
            assert hasattr(self.app, "train_data")
            assert hasattr(self.app, "val_data")
            assert hasattr(self.app, "test_data")

            # Check output strings contain expected information
            assert "Data Generated Successfully!" in data_info
            assert "Training Metrics:" in training_metrics
            assert "Test Set Evaluation:" in evaluation_metrics
            assert "2.5000" in training_metrics  # MSE value
            assert "0.8500" in training_metrics  # R² value

    def test_generate_and_train_error(self):
        """Test error handling in data generation and training."""
        # Mock the data generator to raise an exception
        with patch.object(
            self.app.data_generator,
            "generate_data",
            side_effect=Exception("Test error"),
        ):
            data_info, training_metrics, evaluation_metrics = (
                self.app.generate_and_train(
                    n_samples=1000,
                    noise_level=0.1,
                    train_size=0.7,
                    val_size=0.15,
                    test_size=0.15,
                )
            )

            assert "Error during data generation and training" in data_info
            assert training_metrics == ""
            assert evaluation_metrics == ""

    def test_predict_consumption_not_trained(self):
        """Test prediction when model is not trained."""
        result = self.app.predict_consumption(25.0, "Monday", False)

        assert "Model must be trained first" in result

    def test_predict_consumption_success(self):
        """Test successful prediction."""
        # Set up the app as if it's trained
        self.app.is_model_trained = True

        # Mock the model prediction
        with patch.object(
            self.app.model, "predict", return_value=16.5
        ) as mock_predict, patch.object(
            self.app.model, "get_model_coefficients"
        ) as mock_coeffs:

            mock_coeffs.return_value = {
                "feature_names": ["temperature", "major_event"],
                "coefficients": [0.3, 2.0],
                "intercept": 10.0,
            }

            result = self.app.predict_consumption(25.0, "Monday", True)

            # Check that prediction was called
            mock_predict.assert_called_once_with(25.0, "Monday", 1)

            # Check output contains expected information
            assert "Estimated Daily Electricity Consumption: 16.5 kWh" in result
            assert "Temperature: 25.0°C" in result
            assert "Day of Week: Monday" in result
            assert "Major Event: Yes" in result
            assert "Model Type: Linear Regression" in result

    def test_predict_consumption_error(self):
        """Test error handling in prediction."""
        # Set up the app as if it's trained
        self.app.is_model_trained = True

        # Mock the model to raise an exception
        with patch.object(
            self.app.model, "predict", side_effect=Exception("Prediction error")
        ):
            result = self.app.predict_consumption(25.0, "Monday", False)

            assert "Error during prediction" in result

    def test_get_model_info_not_trained(self):
        """Test getting model info when model is not trained."""
        result = self.app.get_model_info()

        assert "Model must be trained first" in result

    def test_get_model_info_success(self):
        """Test successful model info retrieval."""
        # Set up the app as if it's trained
        self.app.is_model_trained = True

        # Mock the model coefficients
        with patch.object(self.app.model, "get_model_coefficients") as mock_coeffs:
            mock_coeffs.return_value = {
                "feature_names": ["temperature", "day_tuesday", "major_event"],
                "coefficients": [0.3, 0.5, 2.0],
                "intercept": 10.0,
            }

            result = self.app.get_model_info()

            # Check output contains expected information
            assert "**Model Information:**" in result
            assert "**Model Type:** Linear Regression" in result
            assert "**Intercept:** 10.0000" in result
            assert "**Feature Coefficients:**" in result
            assert "temperature" in result
            assert "major_event" in result
            assert "**Interpretation:**" in result

    def test_get_model_info_error(self):
        """Test error handling in model info retrieval."""
        # Set up the app as if it's trained
        self.app.is_model_trained = True

        # Mock the model to raise an exception
        with patch.object(
            self.app.model,
            "get_model_coefficients",
            side_effect=Exception("Info error"),
        ):
            result = self.app.get_model_info()

            assert "Error getting model info" in result

    def test_boolean_conversion_in_prediction(self):
        """Test that boolean values are correctly converted to integers."""
        # Set up the app as if it's trained
        self.app.is_model_trained = True

        # Mock the model prediction
        with patch.object(self.app.model, "predict") as mock_predict, patch.object(
            self.app.model, "get_model_coefficients"
        ) as mock_coeffs:

            mock_predict.return_value = 15.0
            mock_coeffs.return_value = {
                "feature_names": ["temperature", "major_event"],
                "coefficients": [0.3, 2.0],
                "intercept": 10.0,
            }

            # Test with True
            self.app.predict_consumption(25.0, "Monday", True)
            mock_predict.assert_called_with(25.0, "Monday", 1)

            # Test with False
            self.app.predict_consumption(25.0, "Monday", False)
            mock_predict.assert_called_with(25.0, "Monday", 0)

    def test_data_storage_after_training(self):
        """Test that data is properly stored after training."""
        # Mock the data generator
        with patch.object(
            self.app.data_generator, "generate_data"
        ) as mock_generate, patch.object(
            self.app.data_generator, "split_data"
        ) as mock_split, patch.object(
            self.app.model, "train"
        ) as mock_train, patch.object(
            self.app.model, "evaluate"
        ) as mock_evaluate:

            # Create mock data
            mock_data = pd.DataFrame(
                {
                    "temperature": [25.0, 30.0],
                    "day_of_week": ["Monday", "Tuesday"],
                    "major_event": [0, 1],
                    "consumption_kwh": [15.0, 18.0],
                }
            )
            mock_generate.return_value = mock_data

            train_data = mock_data.iloc[:1]
            val_data = mock_data.iloc[1:2]
            test_data = mock_data.iloc[1:2]
            mock_split.return_value = (train_data, val_data, test_data)

            mock_train.return_value = {
                "train_mse": 2.5,
                "train_rmse": 1.58,
                "train_mae": 1.2,
                "train_r2": 0.85,
            }
            mock_evaluate.return_value = {
                "test_mse": 2.8,
                "test_rmse": 1.67,
                "test_mae": 1.3,
                "test_r2": 0.82,
            }

            # Call the method
            self.app.generate_and_train(1000, 0.1, 0.7, 0.15, 0.15)

            # Check that data is stored
            assert hasattr(self.app, "train_data")
            assert hasattr(self.app, "val_data")
            assert hasattr(self.app, "test_data")
            assert len(self.app.train_data) == 1
            assert len(self.app.val_data) == 1
            assert len(self.app.test_data) == 1

    def test_interface_creation(self):
        """Test that the Gradio interface can be created."""
        # This test verifies that the interface creation doesn't raise exceptions
        try:
            interface = self.app.create_interface()
            assert interface is not None
        except Exception as e:
            pytest.fail(f"Interface creation failed: {e}")

    def test_prediction_output_format(self):
        """Test that prediction output is properly formatted."""
        # Set up the app as if it's trained
        self.app.is_model_trained = True

        # Mock the model
        with patch.object(
            self.app.model, "predict", return_value=16.5
        ) as mock_predict, patch.object(
            self.app.model, "get_model_coefficients"
        ) as mock_coeffs:

            mock_coeffs.return_value = {
                "feature_names": ["temperature", "major_event"],
                "coefficients": [0.3, 2.0],
                "intercept": 10.0,
            }

            result = self.app.predict_consumption(25.0, "Monday", False)

            # Check formatting
            assert "**Prediction Result:**" in result
            assert "**Input Parameters:**" in result
            assert "**Model Information:**" in result
            assert "Estimated Daily Electricity Consumption: 16.5 kWh" in result
            assert "Temperature: 25.0°C" in result
            assert "Day of Week: Monday" in result
            assert "Major Event: No" in result

    def test_model_info_output_format(self):
        """Test that model info output is properly formatted."""
        # Set up the app as if it's trained
        self.app.is_model_trained = True

        # Mock the model coefficients
        with patch.object(self.app.model, "get_model_coefficients") as mock_coeffs:
            mock_coeffs.return_value = {
                "feature_names": ["temperature", "day_tuesday", "major_event"],
                "coefficients": [0.3, 0.5, 2.0],
                "intercept": 10.0,
            }

            result = self.app.get_model_info()

            # Check formatting
            assert "**Model Information:**" in result
            assert "**Model Type:**" in result
            assert "**Intercept:**" in result
            assert "**Feature Coefficients:**" in result
            assert "| Feature | Coefficient |" in result
            assert "**Interpretation:**" in result
            assert "Positive coefficients increase predicted consumption" in result
