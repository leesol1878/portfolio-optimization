import pytest
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

def test_pandas_import():
    """Test that pandas is imported correctly"""
    assert pd is not None
    assert pd.__version__ is not None

def test_numpy_import():
    """Test that numpy is imported correctly"""
    assert np is not None
    assert np.__version__ is not None

def test_yfinance_import():
    """Test that yfinance is imported correctly"""
    assert yf is not None

def test_matplotlib_import():
    """Test that matplotlib is imported correctly"""
    assert plt is not None

def test_seaborn_import():
    """Test that seaborn is imported correctly"""
    assert sns is not None

def test_pandas_dataframe():
    """Test that we can create a pandas DataFrame"""
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    assert df.shape == (3, 2)
    assert list(df.columns) == ['A', 'B']