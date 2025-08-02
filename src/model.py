"""
Model Module for Daily Household Electricity Consumption Predictor

This module handles data preprocessing, model training, evaluation, and prediction
for the electricity consumption prediction model.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
from typing import Tuple, Dict, Any, Optional
import os


class ElectricityConsumptionModel:
    """Linear regression model for predicting daily electricity consumption."""

    def __init__(self):
        """Initialize the model with preprocessing pipeline."""
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.is_trained = False

    def _create_preprocessor(self) -> ColumnTransformer:
        """
        Create preprocessing pipeline for the features.

        Returns:
            ColumnTransformer with preprocessing steps
        """
        # Numerical features (temperature)
        numerical_features = ["temperature"]
        numerical_transformer = StandardScaler()

        # Categorical features (day_of_week)
        categorical_features = ["day_of_week"]
        categorical_transformer = OneHotEncoder(drop="first", sparse=False)

        # Boolean features (major_event) - no transformation needed
        boolean_features = ["major_event"]
        boolean_transformer = "passthrough"

        # Combine all transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, numerical_features),
                ("cat", categorical_transformer, categorical_features),
                ("bool", boolean_transformer, boolean_features),
            ],
            remainder="drop",
        )

        return preprocessor

    def _create_pipeline(self) -> Pipeline:
        """
        Create the complete model pipeline.

        Returns:
            Pipeline with preprocessing and model
        """
        preprocessor = self._create_preprocessor()
        model = LinearRegression()

        pipeline = Pipeline([("preprocessor", preprocessor), ("regressor", model)])

        return pipeline

    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for training/prediction.

        Args:
            data: Input DataFrame with raw features

        Returns:
            DataFrame with prepared features
        """
        required_columns = ["temperature", "day_of_week", "major_event"]

        # Validate input data
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Validate data types and ranges
        if not all(data["temperature"].between(15, 35)):
            raise ValueError("Temperature must be between 15 and 35 degrees Celsius")

        valid_days = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        if not all(day in valid_days for day in data["day_of_week"].unique()):
            raise ValueError(f"Day of week must be one of: {valid_days}")

        if not all(data["major_event"].isin([0, 1])):
            raise ValueError("Major event must be 0 or 1")

        return data[required_columns].copy()

    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame) -> Dict[str, float]:
        """
        Train the model on the provided data.

        Args:
            X_train: Training features
            y_train: Training targets

        Returns:
            Dictionary with training metrics
        """
        # Prepare features
        X_prepared = self.prepare_features(X_train)

        # Create and train pipeline
        self.model = self._create_pipeline()
        self.model.fit(X_prepared, y_train["consumption_kwh"])

        # Store feature names for later use
        self.feature_names = X_prepared.columns.tolist()
        self.is_trained = True

        # Calculate training metrics
        y_pred = self.model.predict(X_prepared)
        metrics = {
            "train_mse": mean_squared_error(y_train["consumption_kwh"], y_pred),
            "train_rmse": np.sqrt(
                mean_squared_error(y_train["consumption_kwh"], y_pred)
            ),
            "train_mae": mean_absolute_error(y_train["consumption_kwh"], y_pred),
            "train_r2": r2_score(y_train["consumption_kwh"], y_pred),
        }

        return metrics

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate the model on test data.

        Args:
            X_test: Test features
            y_test: Test targets

        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")

        # Prepare features
        X_prepared = self.prepare_features(X_test)

        # Make predictions
        y_pred = self.model.predict(X_prepared)

        # Calculate metrics
        metrics = {
            "test_mse": mean_squared_error(y_test["consumption_kwh"], y_pred),
            "test_rmse": np.sqrt(mean_squared_error(y_test["consumption_kwh"], y_pred)),
            "test_mae": mean_absolute_error(y_test["consumption_kwh"], y_pred),
            "test_r2": r2_score(y_test["consumption_kwh"], y_pred),
        }

        return metrics

    def predict(self, temperature: float, day_of_week: str, major_event: int) -> float:
        """
        Make a single prediction.

        Args:
            temperature: Average daily temperature in Celsius
            day_of_week: Day of the week
            major_event: Whether there's a major event (0 or 1)

        Returns:
            Predicted electricity consumption in kWh
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Create input DataFrame
        input_data = pd.DataFrame(
            {
                "temperature": [temperature],
                "day_of_week": [day_of_week],
                "major_event": [major_event],
            }
        )

        # Prepare features
        X_prepared = self.prepare_features(input_data)

        # Make prediction
        prediction = self.model.predict(X_prepared)[0]

        return max(0, prediction)  # Ensure non-negative prediction

    def get_model_coefficients(self) -> Dict[str, Any]:
        """
        Get model coefficients and feature names.

        Returns:
            Dictionary with model coefficients and feature information
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before accessing coefficients")

        # Get feature names from preprocessor
        preprocessor = self.model.named_steps["preprocessor"]
        feature_names = []

        # Numerical features
        feature_names.extend(["temperature"])

        # Categorical features (one-hot encoded)
        cat_transformer = preprocessor.named_transformers_["cat"]
        day_names = [
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]  # Monday is dropped
        feature_names.extend([f"day_{day.lower()}" for day in day_names])

        # Boolean features
        feature_names.extend(["major_event"])

        # Get coefficients
        coefficients = self.model.named_steps["regressor"].coef_
        intercept = self.model.named_steps["regressor"].intercept_

        return {
            "feature_names": feature_names,
            "coefficients": coefficients.tolist(),
            "intercept": float(intercept),
        }

    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.

        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Save model
        joblib.dump(self.model, filepath)

    def load_model(self, filepath: str) -> None:
        """
        Load a trained model from disk.

        Args:
            filepath: Path to the saved model
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        self.model = joblib.load(filepath)
        self.is_trained = True

        # Extract feature names from the loaded model
        preprocessor = self.model.named_steps["preprocessor"]
        self.feature_names = ["temperature", "day_of_week", "major_event"]
