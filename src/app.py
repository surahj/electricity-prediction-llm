"""
Gradio Web Application for Daily Household Electricity Consumption Predictor

This module provides a user-friendly web interface for the electricity consumption
prediction model using Gradio.
"""

import gradio as gr
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.data_generator import DataGenerator
from src.model import ElectricityConsumptionModel


class ElectricityPredictorApp:
    """Gradio application for electricity consumption prediction."""

    def __init__(self):
        """Initialize the application with model and data generator."""
        self.data_generator = DataGenerator(seed=42)
        self.model = ElectricityConsumptionModel()
        self.is_model_trained = False

    def generate_and_train(
        self,
        n_samples: int,
        noise_level: float,
        train_size: float,
        val_size: float,
        test_size: float,
    ) -> Tuple[str, str, str]:
        """
        Generate synthetic data and train the model.

        Args:
            n_samples: Number of data points to generate
            noise_level: Level of noise in the data
            train_size: Proportion for training set
            val_size: Proportion for validation set
            test_size: Proportion for test set

        Returns:
            Tuple of (data_info, training_metrics, evaluation_metrics)
        """
        try:
            # Generate data
            data = self.data_generator.generate_data(n_samples, noise_level)

            # Split data
            train_data, val_data, test_data = self.data_generator.split_data(
                data, train_size, val_size, test_size
            )

            # Store data for later use
            self.train_data = train_data
            self.val_data = val_data
            self.test_data = test_data

            # Train model
            X_train = train_data.drop("consumption_kwh", axis=1)
            y_train = train_data[["consumption_kwh"]]

            training_metrics = self.model.train(X_train, y_train)

            # Evaluate model
            X_test = test_data.drop("consumption_kwh", axis=1)
            y_test = test_data[["consumption_kwh"]]

            evaluation_metrics = self.model.evaluate(X_test, y_test)

            self.is_model_trained = True

            # Format output strings
            data_info = f"""
            **Data Generated Successfully!**
            
            - Total samples: {len(data)}
            - Training samples: {len(train_data)}
            - Validation samples: {len(val_data)}
            - Test samples: {len(test_data)}
            
            **Data Statistics:**
            - Temperature range: {data['temperature'].min():.1f}°C - {data['temperature'].max():.1f}°C
            - Consumption range: {data['consumption_kwh'].min():.1f} - {data['consumption_kwh'].max():.1f} kWh
            - Average consumption: {data['consumption_kwh'].mean():.1f} kWh
            """

            training_metrics_str = f"""
            **Training Metrics:**
            - Mean Squared Error (MSE): {training_metrics['train_mse']:.4f}
            - Root Mean Squared Error (RMSE): {training_metrics['train_rmse']:.4f}
            - Mean Absolute Error (MAE): {training_metrics['train_mae']:.4f}
            - R-squared (R²): {training_metrics['train_r2']:.4f}
            """

            evaluation_metrics_str = f"""
            **Test Set Evaluation:**
            - Mean Squared Error (MSE): {evaluation_metrics['test_mse']:.4f}
            - Root Mean Squared Error (RMSE): {evaluation_metrics['test_rmse']:.4f}
            - Mean Absolute Error (MAE): {evaluation_metrics['test_mae']:.4f}
            - R-squared (R²): {evaluation_metrics['test_r2']:.4f}
            """

            return data_info, training_metrics_str, evaluation_metrics_str

        except Exception as e:
            error_msg = f"Error during data generation and training: {str(e)}"
            return error_msg, "", ""

    def predict_consumption(
        self, temperature: float, day_of_week: str, major_event: bool
    ) -> str:
        """
        Make a prediction for electricity consumption.

        Args:
            temperature: Average daily temperature in Celsius
            day_of_week: Day of the week
            major_event: Whether there's a major event

        Returns:
            Formatted prediction result
        """
        if not self.is_model_trained:
            return "**Error:** Model must be trained first. Please generate data and train the model."

        try:
            # Convert boolean to int
            major_event_int = 1 if major_event else 0

            # Make prediction
            prediction = self.model.predict(temperature, day_of_week, major_event_int)

            # Get model coefficients for explanation
            coefficients = self.model.get_model_coefficients()

            # Format result
            result = f"""
            **Prediction Result:**
            
            **Estimated Daily Electricity Consumption: {prediction:.1f} kWh**
            
            **Input Parameters:**
            - Temperature: {temperature}°C
            - Day of Week: {day_of_week}
            - Major Event: {'Yes' if major_event else 'No'}
            
            **Model Information:**
            - Model Type: Linear Regression
            - Intercept: {coefficients['intercept']:.4f}
            - Number of Features: {len(coefficients['feature_names'])}
            """

            return result

        except Exception as e:
            return f"**Error during prediction:** {str(e)}"

    def get_model_info(self) -> str:
        """
        Get detailed information about the trained model.

        Returns:
            Formatted model information
        """
        if not self.is_model_trained:
            return "**Error:** Model must be trained first."

        try:
            coefficients = self.model.get_model_coefficients()
            print(coefficients)

            # Create feature importance table
            feature_importance = []
            for i, (feature, coef) in enumerate(
                zip(coefficients["feature_names"], coefficients["coefficients"])
            ):
                feature_importance.append(f"| {feature} | {coef:.4f} |")

            feature_table = "\n".join(feature_importance)

            info = f"""
            **Model Information:**
            
            **Model Type:** Linear Regression
            
            **Intercept:** {coefficients['intercept']:.4f}
            
            **Feature Coefficients:**
            | Feature | Coefficient |
            |---------|-------------|
            {feature_table}
            
            **Interpretation:**
            - Positive coefficients increase predicted consumption
            - Negative coefficients decrease predicted consumption
            - Temperature coefficient shows how much consumption changes per degree Celsius
            - Day coefficients show consumption differences compared to Monday (baseline)
            - Major event coefficient shows additional consumption during events
            """

            return info

        except Exception as e:
            return f"**Error getting model info:** {str(e)}"

    def create_interface(self) -> gr.Interface:
        """
        Create the Gradio interface.

        Returns:
            Gradio Interface object
        """
        with gr.Blocks(
            title="Daily Household Electricity Consumption Predictor"
        ) as interface:
            gr.Markdown(
                """
            # ⚡ Daily Household Electricity Consumption Predictor
            
            This application helps Nigerian households estimate their daily electricity consumption 
            based on temperature, day of the week, and major events.
            
            ## How to Use:
            1. **Generate Data & Train Model**: Click the button to generate synthetic data and train the model
            2. **Make Predictions**: Enter your parameters and get consumption estimates
            3. **View Model Info**: See how the model works and feature importance
            """
            )

            with gr.Tab("Data Generation & Training"):
                gr.Markdown("### Step 1: Generate Synthetic Data and Train Model")

                with gr.Row():
                    with gr.Column():
                        n_samples = gr.Slider(
                            minimum=100,
                            maximum=5000,
                            value=1000,
                            step=100,
                            label="Number of Data Points",
                        )
                        noise_level = gr.Slider(
                            minimum=0.01,
                            maximum=0.5,
                            value=0.1,
                            step=0.01,
                            label="Noise Level",
                        )

                    with gr.Column():
                        train_size = gr.Slider(
                            minimum=0.5,
                            maximum=0.9,
                            value=0.7,
                            step=0.05,
                            label="Training Set Proportion",
                        )
                        val_size = gr.Slider(
                            minimum=0.05,
                            maximum=0.3,
                            value=0.15,
                            step=0.05,
                            label="Validation Set Proportion",
                        )
                        test_size = gr.Slider(
                            minimum=0.05,
                            maximum=0.3,
                            value=0.15,
                            step=0.05,
                            label="Test Set Proportion",
                        )

                train_button = gr.Button(
                    "Generate Data & Train Model", variant="primary"
                )

                with gr.Row():
                    data_info = gr.Markdown("**Data information will appear here...**")

                with gr.Row():
                    training_metrics = gr.Markdown(
                        "**Training metrics will appear here...**"
                    )
                    evaluation_metrics = gr.Markdown(
                        "**Evaluation metrics will appear here...**"
                    )

                train_button.click(
                    fn=self.generate_and_train,
                    inputs=[n_samples, noise_level, train_size, val_size, test_size],
                    outputs=[data_info, training_metrics, evaluation_metrics],
                )

            with gr.Tab("Prediction"):
                gr.Markdown("### Step 2: Predict Electricity Consumption")

                with gr.Row():
                    with gr.Column():
                        temperature = gr.Slider(
                            minimum=15,
                            maximum=35,
                            value=25,
                            step=0.5,
                            label="Average Daily Temperature (°C)",
                        )
                        day_of_week = gr.Dropdown(
                            choices=[
                                "Monday",
                                "Tuesday",
                                "Wednesday",
                                "Thursday",
                                "Friday",
                                "Saturday",
                                "Sunday",
                            ],
                            value="Monday",
                            label="Day of the Week",
                        )
                        major_event = gr.Checkbox(
                            label="Major Event (Holiday, Power Outage, etc.)",
                            value=False,
                        )

                    with gr.Column():
                        predict_button = gr.Button(
                            "Predict Consumption", variant="primary"
                        )
                        prediction_result = gr.Markdown(
                            "**Prediction result will appear here...**"
                        )

                predict_button.click(
                    fn=self.predict_consumption,
                    inputs=[temperature, day_of_week, major_event],
                    outputs=prediction_result,
                )

            with gr.Tab("Model Information"):
                gr.Markdown("### Step 3: Understand the Model")

                info_button = gr.Button("Show Model Information", variant="secondary")
                model_info = gr.Markdown("**Model information will appear here...**")

                info_button.click(fn=self.get_model_info, inputs=[], outputs=model_info)

            gr.Markdown(
                """
            ---
            **Note:** This application uses synthetic data for demonstration purposes. 
            In a real-world scenario, you would use actual historical consumption data.
            """
            )

        return interface


def main():
    """Main function to launch the application."""
    app = ElectricityPredictorApp()
    interface = app.create_interface()

    # Get port from environment variable (for Render deployment)
    port = int(os.environ.get("PORT", 7860))

    # Launch the interface
    interface.launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=port,  # Use environment port
        share=False,  # Don't create public link
        debug=False,  # Disable debug mode for production
    )


if __name__ == "__main__":
    main()
