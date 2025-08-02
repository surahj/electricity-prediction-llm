"""
Electricity Consumption Predictor Package

This package contains the main components for the electricity consumption prediction application:
- DataGenerator: Synthetic data generation
- ElectricityConsumptionModel: ML model implementation
- ElectricityPredictorApp: Gradio web interface
"""

from .data_generator import DataGenerator
from .model import ElectricityConsumptionModel
from .app import ElectricityPredictorApp

__version__ = "1.0.0"
__author__ = "Opeyemi Odebode"
__email__ = "odebodeopeyemi@gmail.com"

__all__ = ["DataGenerator", "ElectricityConsumptionModel", "ElectricityPredictorApp"]
