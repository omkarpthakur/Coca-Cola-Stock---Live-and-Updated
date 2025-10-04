# -*- coding: utf-8 -*-

from flask import Flask, jsonify, request
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

app = Flask(__name__)
CORS(app)

class StockAnalyzer:
    def __init__(self, ticker='KO'):
        self.ticker = ticker
        self.data = None
    
    def fetch_stock_data(self, start_date='2015-01-01', end_date=None):
        """Fetch stock data from Yahoo Finance"""
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        try:
            data = yf.download(self.ticker, start=start_date, end=end_date)
            data.reset_index(inplace=True)
            data.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
            
            # Calculate technical indicators
            data['MA_20'] = data['Close'].rolling(window=20).mean()
            data['MA_50'] = data['Close'].rolling(window=50).mean()
            data['Daily_Return'] = data['Close'].pct_change()
            data['Volatility'] = data['Daily_Return'].rolling(window=20).std()
            
            # Drop NaN values
            data.dropna(inplace=True)
            
            self.data = data
            return True
        except Exception as e:
            print(f"Error fetching data: {e}")
            return False
    
    def get_basic_stats(self):
        """Get basic statistics for the stock"""
        if self.data is None:
            return None
        
        stats = {
            'current_price': float(self.data['Close'].iloc[-1]),
            'price_change': float(self.data['Close'].iloc[-1] - self.data['Close'].iloc[-2]),
            'price_change_percent': float((self.data['Close'].iloc[-1] - self.data['Close'].iloc[-2]) / self.data['Close'].iloc[-2] * 100),
            'volume': int(self.data['Volume'].iloc[-1]),
            'high_52_week': float(self.data['High'].max()),
            'low_52_week': float(self.data['Low'].min()),
            'avg_volume': int(self.data['Volume'].mean()),
            'volatility': float(self.data['Volatility'].iloc[-1])
        }
        return stats
    
    def get_chart_data(self):
        """Prepare data for charts"""
        if self.data is None:
            return None
        
        chart_data = {
            'dates': self.data['Date'].dt.strftime('%Y-%m-%d').tolist(),
            'prices': {
                'close': self.data['Close'].tolist(),
                'open': self.data['Open'].tolist(),
                'high': self.data['High'].tolist(),
                'low': self.data['Low'].tolist()
            },
            'moving_averages': {
                'ma_20': self.data['MA_20'].tolist(),
                'ma_50': self.data['MA_50'].tolist()
            },
            'volume': self.data['Volume'].tolist(),
            'daily_returns': self.data['Daily_Return'].tolist()
        }
        return chart_data
    
    def get_analysis_metrics(self):
        """Calculate various analysis metrics"""
        if self.data is None:
            return None
        
        returns = self.data['Daily_Return'].dropna()
        
        metrics = {
            'total_return': float((self.data['Close'].iloc[-1] - self.data['Close'].iloc[0]) / self.data['Close'].iloc[0] * 100),
            'annualized_return': float(((self.data['Close'].iloc[-1] / self.data['Close'].iloc[0]) ** (252/len(self.data)) - 1) * 100),
            'volatility': float(returns.std() * np.sqrt(252) * 100),
            'sharpe_ratio': float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() != 0 else 0,
            'max_drawdown': float(((self.data['Close'].cummax() - self.data['Close']) / self.data['Close'].cummax()).max() * 100),
            'current_ma_20': float(self.data['MA_20'].iloc[-1]),
            'current_ma_50': float(self.data['MA_50'].iloc[-1])
        }
        return metrics

# Initialize stock analyzer
stock_analyzer = StockAnalyzer()

@app.route('/api/stock/update', methods=['POST'])
def update_stock_data():
    """Update stock data with custom date range"""
    try:
        data = request.get_json()
        start_date = data.get('start_date', '2015-01-01')
        end_date = data.get('end_date', datetime.now().strftime('%Y-%m-%d'))
        
        success = stock_analyzer.fetch_stock_data(start_date, end_date)
        
        if success:
            return jsonify({'message': 'Data updated successfully'})
        else:
            return jsonify({'error': 'Failed to fetch data'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stock/stats')
def get_stock_stats():
    """Get basic stock statistics"""
    if stock_analyzer.data is None:
        stock_analyzer.fetch_stock_data()
    
    stats = stock_analyzer.get_basic_stats()
    return jsonify(stats)

@app.route('/api/stock/chart-data')
def get_chart_data():
    """Get data for charts"""
    if stock_analyzer.data is None:
        stock_analyzer.fetch_stock_data()
    
    chart_data = stock_analyzer.get_chart_data()
    return jsonify(chart_data)

@app.route('/api/stock/analysis')
def get_analysis():
    """Get technical analysis metrics"""
    if stock_analyzer.data is None:
        stock_analyzer.fetch_stock_data()
    
    analysis = stock_analyzer.get_analysis_metrics()
    return jsonify(analysis)

@app.route('/api/stock/current')
def get_current_data():
    """Get current stock price and basic info"""
    try:
        ticker = yf.Ticker('KO')
        info = ticker.info
        current_data = {
            'current_price': info.get('currentPrice', info.get('regularMarketPrice')),
            'company_name': info.get('longName', 'The Coca-Cola Company'),
            'currency': info.get('currency', 'USD'),
            'market_cap': info.get('marketCap'),
            'pe_ratio': info.get('trailingPE'),
            'dividend_yield': info.get('dividendYield')
        }
        return jsonify(current_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    return jsonify({'message': 'Coca-Cola Stock Analysis API'})

if __name__ == '__main__':
    # Fetch initial data
    stock_analyzer.fetch_stock_data()
    app.run(debug=True, port=5000)