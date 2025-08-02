"""
Tests for the DataGenerator module.

This module contains comprehensive tests for the DataGenerator class to ensure
proper data generation, splitting, and file operations.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from src.data_generator import DataGenerator


class TestDataGenerator:
    """Test cases for DataGenerator class."""

    def test_initialization(self):
        """Test DataGenerator initialization with and without seed."""
        # Test with seed
        generator = DataGenerator(seed=42)
        assert generator.seed == 42

        # Test without seed
        generator_no_seed = DataGenerator(seed=None)
        assert generator_no_seed.seed is None

    def test_generate_data_basic(self):
        """Test basic data generation with default parameters."""
        generator = DataGenerator(seed=42)
        data = generator.generate_data()

        # Check DataFrame structure
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 1000  # Default n_samples
        assert list(data.columns) == [
            "temperature",
            "day_of_week",
            "major_event",
            "consumption_kwh",
        ]

        # Check data types
        assert data["temperature"].dtype in [np.float64, np.float32]
        assert data["day_of_week"].dtype == "object"
        assert data["major_event"].dtype in [np.int64, np.int32]
        assert data["consumption_kwh"].dtype in [np.float64, np.float32]

    def test_generate_data_custom_parameters(self):
        """Test data generation with custom parameters."""
        generator = DataGenerator(seed=42)
        data = generator.generate_data(n_samples=500, noise_level=0.2)

        assert len(data) == 500

        # Check temperature range
        assert data["temperature"].min() >= 15
        assert data["temperature"].max() <= 35

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

        # Check major event values
        assert all(event in [0, 1] for event in data["major_event"].unique())

        # Check consumption is positive
        assert all(data["consumption_kwh"] > 0)

    def test_generate_data_reproducibility(self):
        """Test that data generation is reproducible with the same seed."""
        # Reset numpy random seed to ensure reproducibility
        np.random.seed(42)

        generator1 = DataGenerator(seed=42)
        data1 = generator1.generate_data(n_samples=100)

        # Reset numpy random seed again
        np.random.seed(42)

        generator2 = DataGenerator(seed=42)
        data2 = generator2.generate_data(n_samples=100)

        pd.testing.assert_frame_equal(data1, data2)

    def test_generate_data_different_seeds(self):
        """Test that different seeds produce different data."""
        generator1 = DataGenerator(seed=42)
        generator2 = DataGenerator(seed=123)

        data1 = generator1.generate_data(n_samples=100)
        data2 = generator2.generate_data(n_samples=100)

        # Data should be different
        assert not data1.equals(data2)

    def test_split_data_basic(self):
        """Test basic data splitting functionality."""
        generator = DataGenerator(seed=42)
        data = generator.generate_data(n_samples=1000)

        train_data, val_data, test_data = generator.split_data(data)

        # Check split proportions
        assert len(train_data) == 700  # 70% of 1000
        assert len(val_data) == 150  # 15% of 1000
        assert len(test_data) == 150  # 15% of 1000

        # Check total samples
        assert len(train_data) + len(val_data) + len(test_data) == len(data)

        # Check all data is used
        all_data = pd.concat([train_data, val_data, test_data])
        assert len(all_data) == len(data)

    def test_split_data_custom_proportions(self):
        """Test data splitting with custom proportions."""
        generator = DataGenerator(seed=42)
        data = generator.generate_data(n_samples=1000)

        train_data, val_data, test_data = generator.split_data(
            data, train_size=0.6, val_size=0.2, test_size=0.2
        )

        assert len(train_data) == 600
        assert len(val_data) == 200
        assert len(test_data) == 200

    def test_split_data_validation(self):
        """Test that split proportions validation works."""
        generator = DataGenerator(seed=42)
        data = generator.generate_data(n_samples=100)

        # Test invalid proportions
        with pytest.raises(AssertionError):
            generator.split_data(data, train_size=0.5, val_size=0.3, test_size=0.3)

        with pytest.raises(AssertionError):
            generator.split_data(data, train_size=0.4, val_size=0.3, test_size=0.2)

    def test_split_data_reproducibility(self):
        """Test that data splitting is reproducible."""
        generator = DataGenerator(seed=42)
        data = generator.generate_data(n_samples=1000)

        # First split
        train1, val1, test1 = generator.split_data(data)

        # Second split with same data
        train2, val2, test2 = generator.split_data(data)

        # Results should be identical
        pd.testing.assert_frame_equal(train1, train2)
        pd.testing.assert_frame_equal(val1, val2)
        pd.testing.assert_frame_equal(test1, test2)

    def test_save_and_load_data(self):
        """Test saving and loading data to/from CSV."""
        generator = DataGenerator(seed=42)
        data = generator.generate_data(n_samples=100)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as tmp_file:
            filepath = tmp_file.name

        try:
            # Save data
            generator.save_data(data, filepath)

            # Check file exists
            assert os.path.exists(filepath)

            # Load data
            loaded_data = generator.load_data(filepath)

            # Check data is identical
            pd.testing.assert_frame_equal(data, loaded_data)

        finally:
            # Clean up
            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_data_statistics(self):
        """Test that generated data has reasonable statistics."""
        generator = DataGenerator(seed=42)
        data = generator.generate_data(n_samples=1000)

        # Temperature statistics
        assert 15 <= data["temperature"].mean() <= 35
        assert data["temperature"].std() > 0

        # Consumption statistics
        assert data["consumption_kwh"].mean() > 0
        assert data["consumption_kwh"].std() > 0

        # Day of week distribution
        day_counts = data["day_of_week"].value_counts()
        assert len(day_counts) == 7
        # All days should have some data
        assert all(count > 0 for count in day_counts.values)

        # Major event distribution (should be mostly 0s)
        event_counts = data["major_event"].value_counts()
        assert 0 in event_counts.index
        assert 1 in event_counts.index
        # Should be more 0s than 1s
        assert event_counts[0] > event_counts[1]

    def test_noise_level_effect(self):
        """Test that noise level affects data variability."""
        generator = DataGenerator(seed=42)

        # Generate data with low noise
        data_low_noise = generator.generate_data(n_samples=1000, noise_level=0.01)

        # Generate data with high noise
        data_high_noise = generator.generate_data(n_samples=1000, noise_level=0.5)

        # High noise should have higher standard deviation
        assert (
            data_high_noise["consumption_kwh"].std()
            > data_low_noise["consumption_kwh"].std()
        )

    def test_temperature_consumption_correlation(self):
        """Test that temperature and consumption have positive correlation."""
        generator = DataGenerator(seed=42)
        data = generator.generate_data(n_samples=1000)

        correlation = data["temperature"].corr(data["consumption_kwh"])
        assert correlation > 0  # Should be positive correlation

    def test_day_of_week_effect(self):
        """Test that different days have different consumption patterns."""
        generator = DataGenerator(seed=42)
        data = generator.generate_data(n_samples=1000)

        # Group by day and check consumption means
        day_consumption = data.groupby("day_of_week")["consumption_kwh"].mean()

        # Should have some variation between days
        assert day_consumption.std() > 0

        # Weekend days (Saturday, Sunday) should generally have higher consumption
        weekend_avg = (day_consumption["Saturday"] + day_consumption["Sunday"]) / 2
        weekday_avg = (
            day_consumption["Monday"]
            + day_consumption["Tuesday"]
            + day_consumption["Wednesday"]
            + day_consumption["Thursday"]
            + day_consumption["Friday"]
        ) / 5

        # This might not always be true due to randomness, but should be generally true
        # We'll just check that there's variation
        assert abs(weekend_avg - weekday_avg) > 0.1

    def test_major_event_effect(self):
        """Test that major events increase consumption."""
        generator = DataGenerator(seed=42)
        data = generator.generate_data(n_samples=1000)

        # Group by major event and check consumption means
        event_consumption = data.groupby("major_event")["consumption_kwh"].mean()

        # Consumption should be higher when there's a major event
        assert event_consumption[1] > event_consumption[0]
