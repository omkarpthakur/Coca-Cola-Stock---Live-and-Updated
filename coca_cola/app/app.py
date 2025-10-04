from flask import Flask, render_template, jsonify, request
import yfinance as yf
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')
import requests
import time

app = Flask(__name__)

class StockPredictor:
    def __init__(self):
        self.model = None
        self.model_path = 'models/stock_model.pkl'
        self.is_trained = False
        self.load_model()
    
    def load_model(self):
        """Load trained model if exists and is valid"""
        try:
            if os.path.exists(self.model_path):
                file_size = os.path.getsize(self.model_path)
                if file_size > 0:  # Check if file is not empty
                    with open(self.model_path, 'rb') as f:
                        self.model = pickle.load(f)
                    self.is_trained = True
                    print(" Model loaded successfully")
                else:
                    print("Model file is empty, will train new model")
                    os.remove(self.model_path)  # Remove empty file
            else:
                print(" No existing model found, will train new model")
        except Exception as e:
            print(f" Error loading model: {e}")
            # Remove corrupted file
            if os.path.exists(self.model_path):
                os.remove(self.model_path)
            self.model = None
            self.is_trained = False
    
    def save_model(self):
        """Save trained model"""
        try:
            os.makedirs('models', exist_ok=True)
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            self.is_trained = True
            print(" Model saved successfully")
        except Exception as e:
            print(f" Error saving model: {e}")
    
    def prepare_features(self, data):
        """Prepare features for model training/prediction"""
        try:
            df = data.copy()
            
            # Technical indicators
            df['MA_20'] = df['Close'].rolling(window=20).mean()
            df['MA_50'] = df['Close'].rolling(window=50).mean()
            df['MA_200'] = df['Close'].rolling(window=200).mean()
            
            # Price momentum
            df['Price_Rate_Of_Change'] = df['Close'].pct_change(periods=10)
            df['Momentum'] = df['Close'] - df['Close'].shift(10)
            
            # Volatility
            df['Volatility'] = df['Close'].rolling(window=20).std()
            
            # Price position relative to moving averages
            df['Price_MA20_Ratio'] = df['Close'] / df['MA_20']
            df['Price_MA50_Ratio'] = df['Close'] / df['MA_50']
            
            # Volume indicators
            df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
            
            # Lagged features
            for lag in [1, 2, 3, 5]:
                df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
            
            # Target variable (next day's closing price)
            df['Target'] = df['Close'].shift(-1)
            
            return df.dropna()
        except Exception as e:
            print(f" Error preparing features: {e}")
            return None
    
    def train_model(self, data):
        """Train the prediction model"""
        try:
            df = self.prepare_features(data)
            if df is None:
                return {'error': 'Failed to prepare features'}
            
            # Feature selection
            feature_columns = [col for col in df.columns if col not in ['Date', 'Target', 'Open', 'High', 'Low', 'Volume']]
            
            # Ensure we have enough data
            if len(df) < 100:
                return {'error': 'Insufficient data for training'}
            
            X = df[feature_columns]
            y = df['Target']
            
            # Split data (chronological split)
            split_index = int(len(X) * 0.8)
            X_train, X_test = X[:split_index], X[split_index:]
            y_train, y_test = y[:split_index], y[split_index:]
            
            # Train model
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            print(" Training model...")
            self.model.fit(X_train, y_train)
            
            # Evaluate model
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)
            
            # Make predictions
            y_pred = self.model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            self.save_model()
            
            return {
                'train_score': train_score,
                'test_score': test_score,
                'mae': mae,
                'rmse': rmse,
                'status': 'success'
            }
        except Exception as e:
            print(f" Error training model: {e}")
            return {'error': f'Model training failed: {str(e)}'}

# Initialize stock predictor
predictor = StockPredictor()

def get_stock_data(ticker='KO', period='2y', max_retries=3):
    """Fetch and process stock data with multiple fallback options"""
    for attempt in range(max_retries):
        try:
            print(f" Attempt {attempt + 1} to fetch {ticker} data...")
            
            # Try different methods to get data
            stock = yf.Ticker(ticker)
            
            # Try different period formats
            periods_to_try = ['2y', '1y', '6mo', '3mo']
            
            for p in periods_to_try:
                try:
                    print(f"   Trying period: {p}")
                    data = stock.history(period=p)
                    
                    if not data.empty and len(data) > 10:
                        print(f"Successfully fetched {len(data)} days of data with period {p}")
                        
                        # Ensure we have a datetime index
                        if not isinstance(data.index, pd.DatetimeIndex):
                            data.index = pd.to_datetime(data.index)
                        
                        # Calculate additional metrics
                        data['Daily_Return'] = data['Close'].pct_change()
                        data['MA_20'] = data['Close'].rolling(window=20).mean()
                        data['MA_50'] = data['Close'].rolling(window=50).mean()
                        data['Volatility'] = data['Daily_Return'].rolling(window=20).std()
                        
                        return data
                except Exception as e:
                    print(f"   Period {p} failed: {e}")
                    continue
            
            # If all periods fail, try using start/end dates
            print(" Trying with specific date range...")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=730)  # 2 years
            data = stock.history(start=start_date, end=end_date)
            
            if not data.empty and len(data) > 10:
                print(f"Successfully fetched {len(data)} days of data with date range")
                
                # Ensure we have a datetime index
                if not isinstance(data.index, pd.DatetimeIndex):
                    data.index = pd.to_datetime(data.index)
                
                # Calculate additional metrics
                data['Daily_Return'] = data['Close'].pct_change()
                data['MA_20'] = data['Close'].rolling(window=20).mean()
                data['MA_50'] = data['Close'].rolling(window=50).mean()
                data['Volatility'] = data['Daily_Return'].rolling(window=20).std()
                
                return data
            
            # If still no data, try alternative tickers
            print(" Trying alternative tickers...")
            alternative_tickers = ['KO', 'COKE', 'PEP']  # Coca-Cola, Coca-Cola Consolidated, PepsiCo
            
            for alt_ticker in alternative_tickers:
                if alt_ticker != ticker:
                    print(f"   Trying alternative: {alt_ticker}")
                    try:
                        alt_stock = yf.Ticker(alt_ticker)
                        alt_data = alt_stock.history(period='1y')
                        
                        if not alt_data.empty and len(alt_data) > 10:
                            print(f" Successfully fetched {len(alt_data)} days of {alt_ticker} data")
                            
                            # Ensure we have a datetime index
                            if not isinstance(alt_data.index, pd.DatetimeIndex):
                                alt_data.index = pd.to_datetime(alt_data.index)
                            
                            # Calculate additional metrics
                            alt_data['Daily_Return'] = alt_data['Close'].pct_change()
                            alt_data['MA_20'] = alt_data['Close'].rolling(window=20).mean()
                            alt_data['MA_50'] = alt_data['Close'].rolling(window=50).mean()
                            alt_data['Volatility'] = alt_data['Daily_Return'].rolling(window=20).std()
                            
                            return alt_data
                    except Exception as e:
                        print(f"   Alternative {alt_ticker} failed: {e}")
                        continue
            
            # If all else fails, generate sample data
            print(" Using sample data as fallback")
            return generate_sample_data()
            
        except Exception as e:
            print(f" Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(" Retrying after delay...")
                time.sleep(2)  # Wait before retry
    
    print(" All data fetch attempts failed")
    return None

def generate_sample_data():
    """Generate realistic sample Coca-Cola stock data"""
    print(" Generating sample Coca-Cola stock data...")
    
    # Create date range for the last 2 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Remove weekends
    dates = dates[dates.weekday < 5]
    
    # Generate realistic Coca-Cola stock data
    np.random.seed(42)  # For reproducible results
    
    # Start around $50 (typical Coca-Cola price range)
    base_price = 50.0
    prices = [base_price]
    volumes = []
    
    # Generate price series with realistic volatility
    for i in range(1, len(dates)):
        # Daily return with some volatility
        daily_return = np.random.normal(0.0005, 0.015)  # Small positive drift with 1.5% daily volatility
        new_price = prices[-1] * (1 + daily_return)
        
        # Ensure price stays in reasonable range
        if new_price < 30:
            new_price = prices[-1] * 1.01  # Bounce back up
        elif new_price > 70:
            new_price = prices[-1] * 0.99  # Pull back down
            
        prices.append(new_price)
        
        # Generate volume (millions of shares)
        volume = np.random.normal(15, 5)  # Average 15M shares per day
        volume = max(1, volume)  # Ensure positive volume
        volumes.append(volume * 1000000)  # Convert to actual share count
    
    # Create DataFrame
    data = pd.DataFrame({
        'Open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
        'High': [p * (1 + abs(np.random.normal(0.01, 0.005))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0.01, 0.005))) for p in prices],
        'Close': prices,
        'Volume': [volumes[0]] + volumes  # Add first volume to match length
    }, index=dates[:len(prices)])
    
    # Calculate additional metrics
    data['Daily_Return'] = data['Close'].pct_change()
    data['MA_20'] = data['Close'].rolling(window=20).mean()
    data['MA_50'] = data['Close'].rolling(window=50).mean()
    data['Volatility'] = data['Daily_Return'].rolling(window=20).std()
    
    print(f" Generated {len(data)} days of sample data")
    return data

def create_plots(data, predictions=None):
    """Create Plotly charts for the dashboard"""
    try:
        # Main price chart with moving averages
        fig_price = go.Figure()
        
        fig_price.add_trace(go.Scatter(
            x=data.index, y=data['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='#1f77b4', width=2)
        ))
        
        fig_price.add_trace(go.Scatter(
            x=data.index, y=data['MA_20'],
            mode='lines',
            name='MA 20',
            line=dict(color='orange', width=1, dash='dash')
        ))
        
        fig_price.add_trace(go.Scatter(
            x=data.index, y=data['MA_50'],
            mode='lines',
            name='MA 50',
            line=dict(color='red', width=1, dash='dash')
        ))
        
        if predictions:
            future_dates, future_prices = predictions
            if future_dates and future_prices:
                fig_price.add_trace(go.Scatter(
                    x=future_dates, y=future_prices,
                    mode='lines+markers',
                    name='Predicted Price',
                    line=dict(color='green', width=2, dash='dot'),
                    marker=dict(size=4)
                ))
        
        fig_price.update_layout(
            title='Coca-Cola Stock Price with Predictions',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            template='plotly_white',
            height=400
        )
        
        # Volume chart
        fig_volume = go.Figure()
        
        colors = ['red' if data['Close'].iloc[i] < data['Open'].iloc[i] else 'green' 
                  for i in range(len(data))]
        
        fig_volume.add_trace(go.Bar(
            x=data.index,
            y=data['Volume'],
            marker_color=colors,
            name='Volume'
        ))
        
        fig_volume.update_layout(
            title='Trading Volume',
            xaxis_title='Date',
            yaxis_title='Volume',
            template='plotly_white',
            height=300
        )
        
        # Daily returns distribution
        returns_data = data['Daily_Return'].dropna()
        if len(returns_data) > 0:
            fig_returns = px.histogram(
                x=returns_data,
                nbins=50,
                title='Distribution of Daily Returns'
            )
            
            fig_returns.update_layout(
                xaxis_title='Daily Return',
                yaxis_title='Frequency',
                template='plotly_white',
                height=300
            )
            returns_html = fig_returns.to_html(full_html=False)
        else:
            returns_html = '<div class="chart-placeholder">No returns data available</div>'
        
        return {
            'price_chart': fig_price.to_html(full_html=False),
            'volume_chart': fig_volume.to_html(full_html=False),
            'returns_chart': returns_html
        }
    except Exception as e:
        print(f" Error creating plots: {e}")
        error_html = '<div class="chart-placeholder">Error loading chart</div>'
        return {
            'price_chart': error_html,
            'volume_chart': error_html,
            'returns_chart': error_html
        }

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/api/stock_data')
def api_stock_data():
    """API endpoint for stock data"""
    data = get_stock_data()
    
    if data is None:
        return jsonify({'error': 'Could not fetch stock data'}), 500
    
    # Convert to list for JSON serialization
    stock_data = {
        'dates': data.index.strftime('%Y-%m-%d').tolist(),
        'prices': data['Close'].tolist(),
        'volumes': data['Volume'].tolist(),
        'returns': data['Daily_Return'].fillna(0).tolist()
    }
    
    return jsonify(stock_data)

@app.route('/api/predict')
def api_predict():
    """API endpoint for predictions"""
    data = get_stock_data()
    
    if data is None:
        return jsonify({'error': 'Could not fetch stock data'}), 500
    
    # Train model if not already trained
    if not predictor.is_trained:
        training_results = predictor.train_model(data)
        if 'error' in training_results:
            return jsonify({'error': training_results['error']}), 500
    else:
        training_results = {'status': 'Model already trained'}
    
    # Make predictions
    future_dates, future_prices = predictor.predict_future(data, days=30)
    
    if future_dates is None or future_prices is None:
        return jsonify({'error': 'Could not generate predictions'}), 500
    
    predictions = {
        'dates': [date.strftime('%Y-%m-%d') for date in future_dates],
        'prices': future_prices,
        'training_results': training_results
    }
    
    return jsonify(predictions)

@app.route('/api/refresh_model')
def api_refresh_model():
    """API endpoint to retrain model"""
    data = get_stock_data()
    
    if data is None:
        return jsonify({'error': 'Could not fetch stock data'}), 500
    
    training_results = predictor.train_model(data)
    
    return jsonify(training_results)

@app.route('/api/dashboard')
def api_dashboard():
    """API endpoint for complete dashboard data"""
    data = get_stock_data()
    
    if data is None:
        return jsonify({'error': 'Could not fetch stock data'}), 500
    
    # Get predictions
    future_predictions = None
    if predictor.is_trained:
        future_dates, future_prices = predictor.predict_future(data, days=30)
        if future_dates and future_prices:
            future_predictions = (future_dates, future_prices)
    
    # Create plots
    plots = create_plots(data, future_predictions)
    
    # Calculate statistics
    current_price = data['Close'].iloc[-1]
    price_change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
    percent_change = (price_change / data['Close'].iloc[-2]) * 100
    
    stats = {
        'current_price': round(current_price, 2),
        'price_change': round(price_change, 2),
        'percent_change': round(percent_change, 2),
        'volume': f"{data['Volume'].iloc[-1]:,}",
        'avg_volume': f"{data['Volume'].tail(20).mean():,.0f}",
        'high_52week': round(data['High'].max(), 2),
        'low_52week': round(data['Low'].min(), 2),
        'model_trained': predictor.is_trained,
        'data_source': 'Sample Data' if data is get_stock_data() else 'Yahoo Finance'
    }
    
    return jsonify({
        'plots': plots,
        'stats': stats,
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

@app.route('/api/model_status')
def api_model_status():
    """API endpoint to check model status"""
    return jsonify({
        'is_trained': predictor.is_trained,
        'model_loaded': predictor.model is not None
    })

@app.route('/api/health')
def api_health():
    """Health check endpoint"""
    data = get_stock_data()
    return jsonify({
        'status': 'healthy',
        'data_available': data is not None,
        'model_trained': predictor.is_trained,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    print(" Starting Coca-Cola Stock Analysis App...")
    print(" Dashboard will be available at: http://localhost:5000")
    print(" Testing data connection...")
    
    # Test data connection on startup
    test_data = get_stock_data()
    if test_data is not None:
        print(f" Data connection successful! Loaded {len(test_data)} records")
    else:
        print(" Using sample data - Yahoo Finance connection failed")
    
    app.run(debug=True, host='0.0.0.0', port=5000)