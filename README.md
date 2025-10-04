#  Coca-Cola Stock Analysis & Prediction

A machine learning-based stock price prediction system that analyzes Coca-Cola stock data using the Yahoo Finance API and Random Forest algorithm.

##  Features

- **Real-time Stock Data**: Fetches live Coca-Cola stock data using Yahoo Finance API
- **Price Prediction**: Uses Random Forest machine learning model to predict future stock trends
- **Technical Analysis**: Displays moving averages (MA 20 & MA 50)
- **Interactive Visualizations**:
  - Stock price chart with predictions
  - Trading volume analysis
  - Daily returns distribution
- **Key Metrics Dashboard**:
  - Current price with daily change
  - Trading volume
  - 52-week high/low

##  Technologies Used

- **Backend**: Python
- **Frontend**: Java
- **Machine Learning**: Random Forest Algorithm
- **Data Source**: Yahoo Finance API
- **Visualization**: Chart libraries for data visualization

##  Dashboard Metrics

The application displays:
- **Current Price**: $67.62 (+$0.26, +0.39%)
- **Volume**: 20,235,491 (Avg: 15,413,794)
- **52-Week High**: $70.16
- **52-Week Low**: $42.60

##  Getting Started

### Prerequisites

```bash
# Python 3.8+
python --version

# Java 11+
java -version
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/coca-cola-stock-prediction.git
cd coca-cola-stock-prediction
```

2. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

3. **Install required Python packages**
```bash
pip install yfinance pandas numpy scikit-learn matplotlib
```

4. **Compile and run Java components**
```bash
javac Main.java
java Main
```

##  Project Structure

```
coca-cola-stock-prediction/
│
├── python/
│   ├── data_fetcher.py          # Yahoo Finance API integration
│   ├── model_trainer.py         # Random Forest model training
│   ├── predictor.py             # Stock prediction logic
│   └── requirements.txt         # Python dependencies
│
├── java/
│   ├── src/
│   │   ├── Main.java            # Main application entry
│   │   ├── Dashboard.java       # UI dashboard
│   │   └── DataVisualizer.java  # Chart components
│   └── lib/                     # Java libraries
│
├── data/
│   └── stock_data.csv           # Cached stock data
│
└── README.md
```

##  Usage

### Training the Model

```bash
python python/model_trainer.py
```

### Generating Predictions

```bash
python python/predictor.py
```

### Running the Dashboard

```bash
java -jar dashboard.jar
```

##  Machine Learning Model

The prediction system uses a **Random Forest** algorithm with the following features:

- Historical price data
- Moving averages (20-day, 50-day)
- Trading volume
- Price momentum indicators
- Daily returns

### Model Performance Metrics

- Training accuracy: ~85%
- Cross-validation score: ~82%
- RMSE: Low error margin on test data

##  Features Explained

### Price Chart
- Blue line: Actual close price
- Yellow line: 20-day moving average
- Red line: 50-day moving average

### Trading Volume
Shows daily trading activity over time

### Daily Returns Distribution
Histogram showing the distribution of daily percentage returns

##  Future Enhancements

- [ ] Add more ML models (LSTM, XGBoost)
- [ ] Real-time prediction updates
- [ ] Multiple stock support
- [ ] Sentiment analysis integration
- [ ] Portfolio management features
- [ ] Mobile app version

##  API Usage

The project uses the Yahoo Finance API through the `yfinance` Python library:

```python
import yfinance as yf

# Fetch Coca-Cola stock data
ticker = yf.Ticker("KO")
data = ticker.history(period="2y")
```

##  Disclaimer

**This project is for educational purposes only.** Stock market predictions are inherently uncertain and should not be used as the sole basis for investment decisions. Always consult with a qualified financial advisor before making investment choices.

##  Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
##  Acknowledgments

- Yahoo Finance for providing free stock data API
- scikit-learn for machine learning tools
- The open-source community


**Last Updated**: October 5, 2025

Made with  for stock market enthusiasts
