# Daily Household Electricity Consumption Predictor

A web-based application designed to help Nigerian households estimate their daily electricity usage in Kilowatt-hours (kWh). This project serves as a practical learning vehicle for Machine Learning Operations (MLOps), covering the full lifecycle from data preparation and model training to deployment, monitoring, and continuous improvement.

## 🎯 Project Goals

### Business Goals

- **Empower Households**: Provide users with a simple, accessible tool to understand and predict their daily electricity consumption
- **Promote Energy Awareness**: Help users identify factors influencing their electricity usage, encouraging more efficient energy habits
- **Inform Budgeting**: Enable users to better estimate their electricity bills, reducing financial surprises
- **Foundational MLOps Learning**: Serve as a concrete project to apply and understand core MLOps principles

### Machine Learning & Technical Goals

- **Accurate Prediction**: Develop a regression model capable of predicting daily kWh consumption with acceptable accuracy
- **User-Friendly Interface**: Create an intuitive web interface that allows easy input of features and clear display of predictions
- **Deployable Application**: Build a self-contained application that can be deployed to a public platform
- **MLOps Readiness**: Design the application with modularity and best practices that facilitate future MLOps implementation

## 🏗️ Project Structure

```
lin-re-model/
├── src/
│   ├── __init__.py
│   ├── data_generator.py      # Synthetic data generation
│   ├── model.py              # ML model training and prediction
│   └── app.py                # Gradio web interface
├── tests/
│   ├── __init__.py
│   ├── test_data_generator.py # Data generator tests
│   ├── test_model.py         # Model tests
│   ├── test_app.py           # Application tests
│   └── test_integration.py   # Integration tests
├── requirements.txt          # Python dependencies
├── pytest.ini              # Pytest configuration
├── run_tests.py            # Test runner script
└── README.md               # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. **Clone the repository** (if not already done):

   ```bash
   git clone <repository-url>
   cd lin-re-model
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:

   ```bash
   python src/app.py
   ```

4. **Open your browser** and navigate to `http://localhost:7860`

## 🧪 Testing

This project includes comprehensive tests to ensure code quality and functionality. The test suite covers:

- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end workflow testing
- **Data Quality Tests**: Validation of synthetic data generation
- **Model Performance Tests**: Verification of model accuracy and consistency

### Running Tests

#### Option 1: Using the test runner script

```bash
# Run all tests with coverage
python run_tests.py

# Run only unit tests
python run_tests.py unit

# Run only integration tests
python run_tests.py integration
```

#### Option 2: Using pytest directly

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_model.py

# Run specific test class
pytest tests/test_model.py::TestElectricityConsumptionModel

# Run specific test method
pytest tests/test_model.py::TestElectricityConsumptionModel::test_train_model
```

### Test Coverage

The test suite provides comprehensive coverage including:

- **Data Generator Tests**:

  - Data generation with different parameters
  - Data splitting functionality
  - Data persistence (save/load)
  - Data quality validation
  - Reproducibility checks

- **Model Tests**:

  - Model initialization and training
  - Feature preparation and validation
  - Prediction functionality
  - Model evaluation metrics
  - Model persistence (save/load)
  - Error handling

- **Application Tests**:

  - Web interface functionality
  - User interaction flows
  - Error handling in UI
  - State management

- **Integration Tests**:
  - Complete workflow testing
  - End-to-end functionality
  - Performance consistency
  - Data quality across components

### Expected Test Results

When all tests pass, you should see output similar to:

```
🧪 Running Daily Household Electricity Consumption Predictor Tests
======================================================================
============================= test session starts ==============================
platform linux -- Python 3.8.x, pytest-7.4.0, pluggy-1.0.0
rootdir: /path/to/lin-re-model
plugins: cov-4.1.0
collected 45 tests

tests/test_app.py ...................                              [ 42%]
tests/test_data_generator.py ...................                  [ 78%]
tests/test_integration.py ..........                              [100%]

---------- coverage: platform linux, python 3.8.x-final-0 -----------
Name                           Stmts   Miss  Cover   Missing
------------------------------------------------------------
src/__init__.py                    1      0   100%
src/app.py                       180      5    97%   180-185
src/data_generator.py             95      2    98%   95-97
src/model.py                     180      8    96%   180-188
------------------------------------------------------------
TOTAL                           456     15    97%

============================== 45 passed in 5.23s ==============================

✅ All tests passed!
```

## 📊 Model Features

The electricity consumption prediction model uses the following features:

1. **Average Daily Temperature** (°C): Numerical input (15-35°C range)
2. **Day of the Week**: Categorical input (Monday through Sunday)
3. **Major Event**: Boolean input (Holiday, Power Outage, etc.)

### Model Algorithm

- **Algorithm**: Linear Regression
- **Preprocessing**: StandardScaler for numerical features, OneHotEncoder for categorical features
- **Evaluation Metrics**: MSE, RMSE, MAE, R²

## 🎮 Using the Application

### Step 1: Generate Data & Train Model

1. Navigate to the "Data Generation & Training" tab
2. Adjust parameters as desired:
   - Number of Data Points (100-5000)
   - Noise Level (0.01-0.5)
   - Training/Validation/Test Set Proportions
3. Click "Generate Data & Train Model"
4. Review the training metrics and evaluation results

### Step 2: Make Predictions

1. Navigate to the "Prediction" tab
2. Enter your parameters:
   - Average Daily Temperature (15-35°C)
   - Day of the Week
   - Major Event (checkbox)
3. Click "Predict Consumption"
4. View your estimated daily electricity consumption

### Step 3: Understand the Model

1. Navigate to the "Model Information" tab
2. Click "Show Model Information"
3. Review feature coefficients and model interpretation

## 🔧 Development

### Adding New Tests

To add new tests:

1. **Unit Tests**: Add to appropriate test file in `tests/`
2. **Integration Tests**: Add to `tests/test_integration.py`
3. **Follow naming convention**: `test_<functionality>`
4. **Use descriptive docstrings**: Explain what the test validates

### Test Best Practices

- **Isolation**: Each test should be independent
- **Descriptive names**: Test names should clearly indicate what they test
- **Assertions**: Use specific assertions with meaningful messages
- **Coverage**: Aim for high test coverage (>95%)
- **Performance**: Tests should run quickly (<10 seconds total)

### Running Tests in Development

During development, you can run tests in different ways:

```bash
# Quick test run (no coverage)
pytest -x  # Stop on first failure

# Run tests in parallel (if pytest-xdist installed)
pytest -n auto

# Run tests with detailed output
pytest -v -s

# Run tests and watch for changes
pytest-watch  # Requires pytest-watch package
```

## 🚀 Deployment

### Local Deployment

```bash
python src/app.py
```

### Hugging Face Spaces Deployment

1. Create a new Space on Hugging Face
2. Upload the project files
3. Configure the Space to run `python src/app.py`
4. The application will be available at your Space URL

## 📈 Future Enhancements

### MLOps Features (Future Phases)

- **Data Versioning**: Implement DVC for data version control
- **Experiment Tracking**: Integrate MLflow or Weights & Biases
- **Model Registry**: Use MLflow Model Registry for model lifecycle management
- **Containerization**: Create Dockerfile for reproducible environments
- **CI/CD**: Set up GitHub Actions for automated testing and deployment
- **Model Monitoring**: Implement monitoring for data drift and performance degradation
- **Continuous Training**: Define triggers for automated retraining

### Model Improvements

- **Feature Engineering**: Add more complex features (historical averages, time of day, etc.)
- **Advanced Models**: Experiment with Random Forest, Gradient Boosting, etc.
- **Hyperparameter Tuning**: Implement automated hyperparameter optimization
- **Ensemble Methods**: Combine multiple models for better predictions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Gradio team for the excellent web interface framework
- Scikit-learn team for the machine learning library
- The MLOps community for best practices and guidance
