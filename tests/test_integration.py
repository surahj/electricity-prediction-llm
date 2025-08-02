"""
Integration tests for the Daily Household Electricity Consumption Predictor.

This module contains integration tests that test the complete workflow
from data generation through model training to prediction.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from src.data_generator import DataGenerator
from src.model import ElectricityConsumptionModel
from src.app import ElectricityPredictorApp


class TestIntegration:
    """Integration tests for the complete system."""

    def setup_method(self):
        """Set up test environment for each test method."""
        self.generator = DataGenerator(seed=42)
        self.model = ElectricityConsumptionModel()
        self.app = ElectricityPredictorApp()

    def test_complete_workflow(self):
        """Test the complete workflow from data generation to prediction."""
        # Step 1: Generate data
        data = self.generator.generate_data(n_samples=1000, noise_level=0.1)
        assert len(data) == 1000
        assert all(
            col in data.columns
            for col in ["temperature", "day_of_week", "major_event", "consumption_kwh"]
        )

        # Step 2: Split data
        train_data, val_data, test_data = self.generator.split_data(data)
        assert len(train_data) + len(val_data) + len(test_data) == len(data)

        # Step 3: Train model
        X_train = train_data.drop("consumption_kwh", axis=1)
        y_train = train_data[["consumption_kwh"]]
        train_metrics = self.model.train(X_train, y_train)

        assert self.model.is_trained
        assert "train_r2" in train_metrics
        assert train_metrics["train_r2"] > 0.3  # Reasonable performance

        # Step 4: Evaluate model
        X_test = test_data.drop("consumption_kwh", axis=1)
        y_test = test_data[["consumption_kwh"]]
        test_metrics = self.model.evaluate(X_test, y_test)

        assert "test_r2" in test_metrics
        assert test_metrics["test_r2"] > 0.3  # Reasonable performance

        # Step 5: Make predictions
        prediction1 = self.model.predict(25.0, "Monday", 0)
        prediction2 = self.model.predict(30.0, "Saturday", 1)

        assert prediction1 > 0
        assert prediction2 > 0
        assert (
            prediction2 > prediction1
        )  # Higher temp + weekend + event should increase consumption

    def test_app_integration(self):
        """Test the complete app workflow."""
        # Test data generation and training through the app
        data_info, training_metrics, evaluation_metrics = self.app.generate_and_train(
            n_samples=500,
            noise_level=0.1,
            train_size=0.7,
            val_size=0.15,
            test_size=0.15,
        )

        assert self.app.is_model_trained
        assert "Data Generated Successfully!" in data_info
        assert "Training Metrics:" in training_metrics
        assert "Test Set Evaluation:" in evaluation_metrics

        # Test prediction through the app
        prediction_result = self.app.predict_consumption(25.0, "Monday", False)
        assert "Estimated Daily Electricity Consumption:" in prediction_result
        assert "Temperature: 25.0°C" in prediction_result

        # Test model info through the app
        model_info = self.app.get_model_info()
        assert "Model Information:" in model_info
        assert "Feature Coefficients:" in model_info

    def test_model_persistence(self):
        """Test model saving and loading."""
        # Generate data and train model
        data = self.generator.generate_data(n_samples=500)
        train_data, _, _ = self.generator.split_data(data)

        X_train = train_data.drop("consumption_kwh", axis=1)
        y_train = train_data[["consumption_kwh"]]
        self.model.train(X_train, y_train)

        # Save model
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp_file:
            model_path = tmp_file.name

        try:
            self.model.save_model(model_path)
            assert os.path.exists(model_path)

            # Load model in new instance
            new_model = ElectricityConsumptionModel()
            new_model.load_model(model_path)

            assert new_model.is_trained

            # Test predictions are identical
            pred1 = self.model.predict(25.0, "Monday", 0)
            pred2 = new_model.predict(25.0, "Monday", 0)

            assert abs(pred1 - pred2) < 1e-10

        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)

    def test_data_persistence(self):
        """Test data saving and loading."""
        # Generate data
        data = self.generator.generate_data(n_samples=100)

        # Save data
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_file:
            data_path = tmp_file.name

        try:
            self.generator.save_data(data, data_path)
            assert os.path.exists(data_path)

            # Load data
            loaded_data = self.generator.load_data(data_path)

            # Check data is identical
            pd.testing.assert_frame_equal(data, loaded_data)

        finally:
            if os.path.exists(data_path):
                os.unlink(data_path)

    def test_model_performance_consistency(self):
        """Test that model performance is consistent across runs."""
        # Generate data
        data = self.generator.generate_data(n_samples=1000, noise_level=0.1)
        train_data, _, test_data = self.generator.split_data(data)

        # Train model multiple times with same data
        X_train = train_data.drop("consumption_kwh", axis=1)
        y_train = train_data[["consumption_kwh"]]
        X_test = test_data.drop("consumption_kwh", axis=1)
        y_test = test_data[["consumption_kwh"]]

        r2_scores = []
        for _ in range(3):
            model = ElectricityConsumptionModel()
            model.train(X_train, y_train)
            metrics = model.evaluate(X_test, y_test)
            r2_scores.append(metrics["test_r2"])

        # R² scores should be very similar (within 0.01)
        assert max(r2_scores) - min(r2_scores) < 0.01

    def test_feature_importance_consistency(self):
        """Test that feature importance is consistent with domain knowledge."""
        # Generate data and train model
        data = self.generator.generate_data(n_samples=1000)
        train_data, _, _ = self.generator.split_data(data)

        X_train = train_data.drop("consumption_kwh", axis=1)
        y_train = train_data[["consumption_kwh"]]
        self.model.train(X_train, y_train)

        # Get coefficients
        coefficients = self.model.get_model_coefficients()

        # Find temperature coefficient
        temp_idx = coefficients["feature_names"].index("temperature")
        temp_coef = coefficients["coefficients"][temp_idx]

        # Find major event coefficient
        event_idx = coefficients["feature_names"].index("major_event")
        event_coef = coefficients["coefficients"][event_idx]

        # Temperature should have positive effect (higher temp = higher consumption)
        assert temp_coef > 0

        # Major event should have positive effect (events increase consumption)
        assert event_coef > 0

    def test_prediction_bounds(self):
        """Test that predictions are within reasonable bounds."""
        # Generate data and train model
        data = self.generator.generate_data(n_samples=1000)
        train_data, _, _ = self.generator.split_data(data)

        X_train = train_data.drop("consumption_kwh", axis=1)
        y_train = train_data[["consumption_kwh"]]
        self.model.train(X_train, y_train)

        # Test predictions across different inputs
        predictions = []

        for temp in [15, 20, 25, 30, 35]:
            for day in [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ]:
                for event in [0, 1]:
                    pred = self.model.predict(temp, day, event)
                    predictions.append(pred)

        # All predictions should be positive
        assert all(p > 0 for p in predictions)

        # Predictions should be within reasonable range (5-50 kWh)
        assert all(5 <= p <= 50 for p in predictions)

    def test_data_quality_checks(self):
        """Test that generated data meets quality requirements."""
        # Generate data
        data = self.generator.generate_data(n_samples=1000)

        # Check for missing values
        assert not data.isnull().any().any()

        # Check data types
        assert data["temperature"].dtype in [np.float64, np.float32]
        assert data["day_of_week"].dtype == "object"
        assert data["major_event"].dtype in [np.int64, np.int32]
        assert data["consumption_kwh"].dtype in [np.float64, np.float32]

        # Check value ranges
        assert data["temperature"].min() >= 15
        assert data["temperature"].max() <= 35
        assert all(data["major_event"].isin([0, 1]))
        assert all(data["consumption_kwh"] > 0)

        # Check day of week values
        valid_days = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        assert all(day in valid_days for day in data["day_of_week"].unique())

        # Check correlations make sense
        temp_consumption_corr = data["temperature"].corr(data["consumption_kwh"])
        assert temp_consumption_corr > 0  # Positive correlation

    def test_error_handling(self):
        """Test error handling in the complete workflow."""
        # Test with invalid temperature
        with pytest.raises(ValueError):
            self.model.predict(10.0, "Monday", 0)  # Temperature too low

        with pytest.raises(ValueError):
            self.model.predict(40.0, "Monday", 0)  # Temperature too high

        # Test with invalid day
        with pytest.raises(ValueError):
            self.model.predict(25.0, "InvalidDay", 0)

        # Test with invalid major event
        with pytest.raises(ValueError):
            self.model.predict(25.0, "Monday", 2)  # Invalid value

        # Test prediction without training
        untrained_model = ElectricityConsumptionModel()
        with pytest.raises(ValueError):
            untrained_model.predict(25.0, "Monday", 0)

    def test_app_state_management(self):
        """Test that app state is properly managed."""
        # Initially not trained
        assert not self.app.is_model_trained

        # After training
        self.app.generate_and_train(500, 0.1, 0.7, 0.15, 0.15)
        assert self.app.is_model_trained

        # Check that data is stored
        assert hasattr(self.app, "train_data")
        assert hasattr(self.app, "val_data")
        assert hasattr(self.app, "test_data")

        # Check data sizes
        assert len(self.app.train_data) > 0
        assert len(self.app.val_data) > 0
        assert len(self.app.test_data) > 0
