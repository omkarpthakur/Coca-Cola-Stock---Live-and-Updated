import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  BarChart, Bar, AreaChart, Area
} from 'recharts';

const API_BASE = 'http://localhost:5000/api';

function App() {
  const [stockStats, setStockStats] = useState(null);
  const [chartData, setChartData] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [currentData, setCurrentData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [dateRange, setDateRange] = useState({
    start_date: '2015-01-01',
    end_date: new Date().toISOString().split('T')[0]
  });

  const fetchData = async () => {
    try {
      setLoading(true);
      setError(null);

      const [statsRes, chartRes, analysisRes, currentRes] = await Promise.all([
        axios.get(`${API_BASE}/stock/stats`),
        axios.get(`${API_BASE}/stock/chart-data`),
        axios.get(`${API_BASE}/stock/analysis`),
        axios.get(`${API_BASE}/stock/current`)
      ]);

      setStockStats(statsRes.data);
      setChartData(chartRes.data);
      setAnalysis(analysisRes.data);
      setCurrentData(currentRes.data);
    } catch (err) {
      setError('Failed to fetch stock data');
      console.error('Error fetching data:', err);
    } finally {
      setLoading(false);
    }
  };

  const updateDateRange = async () => {
    try {
      setLoading(true);
      await axios.post(`${API_BASE}/stock/update`, dateRange);
      await fetchData();
    } catch (err) {
      setError('Failed to update data');
      console.error('Error updating data:', err);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  if (loading) {
    return (
      <div className="container">
        <div className="loading">Loading Coca-Cola Stock Data...</div>
      </div>
    );
  }

  const formatPrice = (price) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD'
    }).format(price);
  };

  const formatPercent = (value) => {
    return `${value?.toFixed(2)}%` || 'N/A';
  };

  const formatNumber = (num) => {
    if (num >= 1e9) return `$${(num / 1e9).toFixed(2)}B`;
    if (num >= 1e6) return `$${(num / 1e6).toFixed(2)}M`;
    return new Intl.NumberFormat('en-US').format(num);
  };

  return (
    <div className="container">
      <header className="header">
        <h1>🍹 Coca-Cola Stock Analysis</h1>
        <p>Real-time stock analysis and technical indicators for KO</p>
      </header>

      {error && <div className="error">{error}</div>}

      <div className="controls">
        <h2>Date Range</h2>
        <div className="date-inputs">
          <input
            type="date"
            value={dateRange.start_date}
            onChange={(e) => setDateRange(prev => ({ ...prev, start_date: e.target.value }))}
          />
          <span>to</span>
          <input
            type="date"
            value={dateRange.end_date}
            onChange={(e) => setDateRange(prev => ({ ...prev, end_date: e.target.value }))}
          />
          <button onClick={updateDateRange}>Update Data</button>
        </div>
      </div>

      {currentData && (
        <div className="stats-grid">
          <div className="stat-card">
            <h3>Current Price</h3>
            <div className="stat-value">{formatPrice(currentData.current_price)}</div>
          </div>
          <div className="stat-card">
            <h3>Market Cap</h3>
            <div className="stat-value">{formatNumber(currentData.market_cap)}</div>
          </div>
          <div className="stat-card">
            <h3>P/E Ratio</h3>
            <div className="stat-value">{currentData.pe_ratio?.toFixed(2) || 'N/A'}</div>
          </div>
          <div className="stat-card">
            <h3>Dividend Yield</h3>
            <div className="stat-value">
              {currentData.dividend_yield ? formatPercent(currentData.dividend_yield * 100) : 'N/A'}
            </div>
          </div>
        </div>
      )}

      {stockStats && (
        <div className="stats-grid">
          <div className="stat-card">
            <h3>Daily Change</h3>
            <div className="stat-value">{formatPrice(stockStats.price_change)}</div>
            <div className={`stat-change ${stockStats.price_change >= 0 ? 'positive' : 'negative'}`}>
              {formatPercent(stockStats.price_change_percent)}
            </div>
          </div>
          <div className="stat-card">
            <h3>Volume</h3>
            <div className="stat-value">{formatNumber(stockStats.volume)}</div>
          </div>
          <div className="stat-card">
            <h3>52-Week High</h3>
            <div className="stat-value">{formatPrice(stockStats.high_52_week)}</div>
          </div>
          <div className="stat-card">
            <h3>52-Week Low</h3>
            <div className="stat-value">{formatPrice(stockStats.low_52_week)}</div>
          </div>
        </div>
      )}

      {chartData && (
        <div className="charts-grid">
          <div className="chart-container">
            <h2>Price Chart with Moving Averages</h2>
            <ResponsiveContainer width="100%" height={400}>
              <LineChart data={chartData.dates.map((date, i) => ({
                date,
                Close: chartData.prices.close[i],
                'MA 20': chartData.moving_averages.ma_20[i],
                'MA 50': chartData.moving_averages.ma_50[i]
              }))}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="Close" stroke="#ff0000" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="MA 20" stroke="#8884d8" strokeWidth={1.5} dot={false} />
                <Line type="monotone" dataKey="MA 50" stroke="#82ca9d" strokeWidth={1.5} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="chart-container">
            <h2>Daily Returns</h2>
            <ResponsiveContainer width="100%" height={400}>
              <AreaChart data={chartData.dates.map((date, i) => ({
                date,
                return: (chartData.daily_returns[i] * 100) || 0
              }))}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis />
                <Tooltip formatter={(value) => [`${value?.toFixed(2)}%`, 'Return']} />
                <Area type="monotone" dataKey="return" stroke="#ff6b6b" fill="#ff6b6b" fillOpacity={0.3} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {chartData && (
        <div className="chart-container">
          <h2>Trading Volume</h2>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={chartData.dates.map((date, i) => ({
              date,
              volume: chartData.volume[i]
            }))}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis />
              <Tooltip formatter={(value) => [formatNumber(value), 'Volume']} />
              <Bar dataKey="volume" fill="#ffa726" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {analysis && (
        <>
          <h2 style={{ margin: '2rem 0 1rem 0' }}>Technical Analysis</h2>
          <div className="analysis-grid">
            <div className="analysis-card">
              <h4>Total Return</h4>
              <div className="analysis-value">{formatPercent(analysis.total_return)}</div>
            </div>
            <div className="analysis-card">
              <h4>Annualized Return</h4>
              <div className="analysis-value">{formatPercent(analysis.annualized_return)}</div>
            </div>
            <div className="analysis-card">
              <h4>Volatility</h4>
              <div className="analysis-value">{formatPercent(analysis.volatility)}</div>
            </div>
            <div className="analysis-card">
              <h4>Sharpe Ratio</h4>
              <div className="analysis-value">{analysis.sharpe_ratio?.toFixed(2)}</div>
            </div>
            <div className="analysis-card">
              <h4>Max Drawdown</h4>
              <div className="analysis-value">{formatPercent(analysis.max_drawdown)}</div>
            </div>
            <div className="analysis-card">
              <h4>Current vs MA 20</h4>
              <div className="analysis-value">
                {analysis.current_ma_20 ? formatPercent(
                  ((stockStats.current_price - analysis.current_ma_20) / analysis.current_ma_20) * 100
                ) : 'N/A'}
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

export default App;