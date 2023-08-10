import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

def qcd_variance(series, window=12):
    """
    Returns the quartile coefficient of dispersion
    of the rolling variance of a series in a given window range.
    """
    variances = series.rolling(window).var().dropna()
    Q1 = np.percentile(variances, 25, method='midpoint')
    Q3 = np.percentile(variances, 75, method='midpoint')
    qcd = (Q3 - Q1) / (Q3 + Q1)
    return round(qcd, 6)

def p_values(series):
    """
    Returns p-values for ADF and KPSS Tests on a time series.
    """
    p_adf = adfuller(series, autolag="AIC")[1]
    p_kpss = kpss(series, regression="c", nlags="auto")[1]
    return round(p_adf, 6), round(p_kpss, 6)

def test_stationarity(series):
    """
    Prints conclusions about series stationarity.
    """
    qcd = qcd_variance(series)
    
    if qcd >= 0.50:
        print(f"\nNon-stationary: heteroscedastic (qcd = {qcd})\n")
    else:
        p_adf, p_kpss = p_values(series)
        print(f"\np_adf: {p_adf}, p_kpss: {p_kpss}")
        
        if p_adf < 0.01 and p_kpss >= 0.05:
            print('Stationary or seasonal-stationary')
        elif p_adf >= 0.1 and p_kpss < 0.05:
            print('Difference-stationary')
        elif p_adf < 0.1 and p_kpss < 0.05:
            print('Trend-stationary')
        else:
            print('Non-stationary; no robust conclusions')

def auto_correlation_plot(series):
    """
    Plots autocorrelations for a given series.
    """
    plt.figure(figsize=(10, 2), dpi=200)
    plot_acf(series, zero=False, lags=25)
    plt.xlabel('Number of lags')
    plt.ylabel('Autocorrelation')
    plt.show()

def partial_auto_correlation_plot(series):
    """
    Plots partial autocorrelations for a given series.
    """
    plt.figure(figsize=(10, 2), dpi=200)
    plot_pacf(series, zero=False, lags=25)
    plt.xlabel('Number of lags')
    plt.ylabel('Partial autocorrelation')
    plt.show()


def harmonic_transformer(frequency=1/365.24):
    sin_transformer = (
        FunctionTransformer(lambda t: np.sin(2*np.pi*frequency*t), validate=False),
        ["timestep"]
    )
    cos_transformer = (
        FunctionTransformer(lambda t: np.cos(2*np.pi*frequency*t), validate=False),
        ["timestep"]
    )
    harmonic_transformers = ColumnTransformer([sin_transformer, cos_transformer])
    return harmonic_transformers
