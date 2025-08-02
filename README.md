# ⚡ Electricity Consumption Predictor

A machine learning application that predicts daily electricity consumption based on various factors like temperature, day of the week, and special events.

## 🚀 Live Demo

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/YOUR_USERNAME/electricity-consumption-predictor)

## 📊 Features

- **Temperature-based predictions**: Considers how temperature affects electricity usage
- **Day of week analysis**: Accounts for different consumption patterns on weekdays vs weekends
- **Special events**: Factors in holidays and major events
- **Interactive interface**: User-friendly Gradio web interface
- **Model insights**: Detailed explanation of prediction factors

## 🛠️ Technology Stack

- **Machine Learning**: scikit-learn (Linear Regression)
- **Data Processing**: pandas, numpy
- **Web Interface**: Gradio
- **Model Persistence**: joblib

## 🏗️ Project Structure

```
├── src/
│   ├── app.py              # Main Gradio application
│   ├── model.py            # ML model implementation
│   └── data_generator.py   # Synthetic data generation
├── tests/
│   ├── test_model.py       # Model unit tests
│   ├── test_app.py         # App unit tests
│   └── test_integration.py # Integration tests
├── app.py                  # Hugging Face Spaces entry point
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🧪 Usage

### Local Development

1. **Clone the repository**:

   ```bash
   git clone https://github.com/YOUR_USERNAME/electricity-consumption-predictor.git
   cd electricity-consumption-predictor
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:

   ```bash
   python app.py
   ```

4. **Run tests**:
   ```bash
   pytest tests/
   ```

### Hugging Face Spaces

The app is automatically deployed on Hugging Face Spaces. Simply visit the live demo link above to use the application.

## 📈 How It Works

1. **Data Generation**: Creates synthetic electricity consumption data with realistic patterns
2. **Model Training**: Trains a linear regression model on historical data
3. **Feature Engineering**: Extracts relevant features (temperature, day of week, events)
4. **Prediction**: Uses the trained model to predict consumption for new scenarios
5. **Interpretation**: Provides detailed breakdown of prediction factors

## 🎯 Model Features

- **Temperature Effect**: Higher temperatures increase AC usage
- **Day of Week**: Weekends typically have different consumption patterns
- **Base Consumption**: Minimum daily electricity usage
- **Event Impact**: Special events can significantly affect consumption

## 📊 Example Predictions

| Temperature | Day       | Event   | Predicted Consumption |
| ----------- | --------- | ------- | --------------------- |
| 25°C        | Monday    | None    | 16.5 kWh              |
| 35°C        | Saturday  | Holiday | 22.3 kWh              |
| 15°C        | Wednesday | None    | 14.1 kWh              |

## 🔧 Configuration

The model can be customized by modifying parameters in `src/model.py`:

- Training data size
- Feature weights
- Model hyperparameters

## 🧪 Testing

Run the test suite to ensure everything works correctly:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src

# Run specific test file
pytest tests/test_model.py
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📞 Support

If you encounter any issues or have questions:

- Open an issue on GitHub
- Check the Hugging Face Spaces discussion
- Review the test files for usage examples

---

**Built with ❤️ using Gradio and scikit-learn**
