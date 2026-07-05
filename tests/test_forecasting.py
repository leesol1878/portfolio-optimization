import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

def test_arima_model():
    """Test ARIMA model can be fit and make predictions"""
    # Create simple test data
    np.random.seed(42)
    data = np.random.randn(100).cumsum() + 100
    
    # Fit ARIMA model
    model = ARIMA(data, order=(1, 1, 1))
    model_fit = model.fit()
    
    # Make prediction
    pred = model_fit.forecast(steps=5)
    
    # Assertions
    assert len(pred) == 5
    assert not any(np.isnan(pred))
    assert isinstance(pred, (np.ndarray, pd.Series))

def test_data_loading():
    """Test that we can load the cleaned data"""
    try:
        # Try to load the data
        df = pd.read_csv('../data/processed/adj_close_data_clean.csv', 
                         index_col=0, parse_dates=True)
        assert not df.empty
        assert 'TSLA' in df.columns
        assert len(df) > 0
    except FileNotFoundError:
        pytest.skip("Data file not found - skipping test")

def test_returns_calculation():
    """Test that returns are calculated correctly"""
    # Create test data
    prices = pd.Series([100, 105, 102, 108, 110])
    returns = prices.pct_change().dropna()
    
    # Expected returns
    expected = pd.Series([0.05, -0.028571, 0.058824, 0.018519])
    
    # Assertions
    assert len(returns) == 4
    assert np.allclose(returns.values, expected.values, rtol=0.001)

def test_lstm_data_prep():
    """Test LSTM data preparation"""
    # Create sequences
    def create_sequences(data, lookback=10):
        X, y = [], []
        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)
    
    # Test with sample data
    data = np.random.randn(100).reshape(-1, 1)
    X, y = create_sequences(data, lookback=10)
    
    assert X.shape[0] == 90
    assert X.shape[1] == 10
    assert y.shape[0] == 90

def test_metrics_calculation():
    """Test performance metrics calculation"""
    actual = np.array([10, 12, 11, 13, 15])
    predicted = np.array([9.5, 12.5, 10.5, 13.5, 14.5])
    
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    assert mae == 0.5
    assert rmse == 0.5
    assert mape == 4.0