"""
Data Generator Module for Daily Household Electricity Consumption Predictor

This module generates synthetic data for training and testing the electricity consumption
prediction model. It creates realistic patterns based on temperature, day of week, and events.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import random


class DataGenerator:
    """Generates synthetic electricity consumption data for training and testing."""

    def __init__(self, seed: Optional[int] = 42):
        """
        Initialize the data generator.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def generate_data(
        self, n_samples: int = 1000, noise_level: float = 0.1
    ) -> pd.DataFrame:
        """
        Generate synthetic electricity consumption data.

        Args:
            n_samples: Number of data points to generate
            noise_level: Level of noise to add to the data (0-1)

        Returns:
            DataFrame with features and target variable
        """
        # Generate features
        temperatures = np.random.normal(25, 8, n_samples)  # Mean 25°C, std 8°C
        temperatures = np.clip(temperatures, 15, 35)  # Clip to realistic range

        days_of_week = np.random.choice(
            [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ],
            n_samples,
        )

        major_events = np.random.choice(
            [0, 1], n_samples, p=[0.9, 0.1]
        )  # 10% chance of event

        # Create base consumption pattern
        base_consumption = 15.0  # Base consumption in kWh

        # Temperature effect (higher temp = higher consumption due to AC/fans)
        temp_effect = 0.3 * (temperatures - 25)

        # Day of week effect (weekends typically higher consumption)
        day_effects = {
            "Monday": 0.5,
            "Tuesday": 0.3,
            "Wednesday": 0.2,
            "Thursday": 0.1,
            "Friday": 0.8,
            "Saturday": 1.5,
            "Sunday": 1.2,
        }
        day_effect = np.array([day_effects[day] for day in days_of_week])

        # Major event effect (events typically increase consumption)
        event_effect = major_events * 2.0

        # Calculate consumption
        consumption = base_consumption + temp_effect + day_effect + event_effect

        # Add noise
        noise = np.random.normal(0, noise_level * np.std(consumption), n_samples)
        consumption += noise

        # Ensure positive values
        consumption = np.maximum(consumption, 5.0)

        # Create DataFrame
        data = pd.DataFrame(
            {
                "temperature": temperatures,
                "day_of_week": days_of_week,
                "major_event": major_events,
                "consumption_kwh": consumption,
            }
        )

        return data

    def split_data(
        self,
        data: pd.DataFrame,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into training, validation, and test sets.

        Args:
            data: Input DataFrame
            train_size: Proportion for training set
            val_size: Proportion for validation set
            test_size: Proportion for test set

        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        assert (
            abs(train_size + val_size + test_size - 1.0) < 1e-6
        ), "Split proportions must sum to 1"

        # Shuffle data
        data_shuffled = data.sample(frac=1, random_state=self.seed).reset_index(
            drop=True
        )

        n_samples = len(data_shuffled)
        train_end = int(n_samples * train_size)
        val_end = train_end + int(n_samples * val_size)

        train_data = data_shuffled[:train_end]
        val_data = data_shuffled[train_end:val_end]
        test_data = data_shuffled[val_end:]

        return train_data, val_data, test_data

    def save_data(self, data: pd.DataFrame, filepath: str) -> None:
        """
        Save data to CSV file.

        Args:
            data: DataFrame to save
            filepath: Path to save the file
        """
        data.to_csv(filepath, index=False)

    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load data from CSV file.

        Args:
            filepath: Path to the file

        Returns:
            Loaded DataFrame
        """
        return pd.read_csv(filepath)
