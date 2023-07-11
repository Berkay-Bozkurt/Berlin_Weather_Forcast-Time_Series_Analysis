import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
def qcd_variance(series,window=10):
    """
    This function returns the quartile coefficient of dispersion
    of the rolling variance of a series in a given window range 
    """
    # rolling variance for a given window 
    variances = series.rolling(window).var().dropna()
    
    # first quartile
    Q1 = np.percentile(variances, 25, method='midpoint')
    
    # third quartile
    Q3 = np.percentile(variances, 75, method='midpoint')
    
    # quartile coefficient of dispersion 
    qcd = (Q3-Q1)/(Q3+Q1)
    
    return round(qcd,6)
def p_values(series):
    """
    returns p-values for ADF and KPSS Tests on a time series
    """
    # p value from Augmented Dickey-Fuller (ADF) Test
    p_adf = adfuller(series, autolag="AIC")[1]
    
    # p value from Kwiatkowski–Phillips–Schmidt–Shin (KPSS) Test
    p_kpss = kpss(series, regression="c", nlags="auto")[1]
    
    return round(p_adf,6), round(p_kpss,6)
def test_stationarity(series):
    """
    returns likely conclusions about series stationarity
    """
    # test heteroscedasticity with qcd
    qcd = qcd_variance(series)
    
    if qcd >= 0.50:
        print(f"\n non-stationary: heteroscedastic (qcd = {qcd}) \n")
    
    # test stationarity
    else:
        p_adf, p_kpss = p_values(series)
        
        # print p-values
        print( f"\n p_adf: {p_adf}, p_kpss: {p_kpss}" )
    
        if (p_adf < 0.01) and (p_kpss >= 0.05):
            print('\n stationary or seasonal-stationary')
            
        elif (p_adf >= 0.1) and (p_kpss < 0.05):
            print('\n difference-stationary')
            
        elif (p_adf < 0.1) and (p_kpss < 0.05):
            print('\n trend-stationary')
        
        else:
            print('\n non-stationary; no robust conclusions\n')

def auto_correlation_plot(series):
    """
    plots autocorrelations for a given series
    """
    mpl.rc('figure',figsize=(10,2),dpi=200)
    plot_acf(series,zero=False,lags=25)
    plt.xlabel('number of lags')
    plt.ylabel('autocorrelation')
def partial_auto_correlation_plot(series):
    """
    plots partial autocorrelations for a given series
    """
    mpl.rc('figure',figsize=(10,2),dpi=200)
    plot_pacf(series,zero=False,lags=25)
    plt.xlabel('number of lags')
    plt.ylabel('partial autocorrelation')

# import module for simulating data
from statsmodels.tsa.arima_process import ArmaProcess

'''def arma_model(ar_coef=[], ma_coef=[]):
    """
    generates sample data for AR, MA, and ARMA processes
    """
    np.random.seed(12345)
    ar = np.array([1] + [-c for c in ar_coef])
    ma = np.array([1] + ma_coef)
    data = ArmaProcess(ar,ma).generate_sample(nsample=200)
    return data



def harmonic_transformer(frequency=1/365.24):
    sin_tuple= (
        FunctionTransformer(lambda t: np.sin(2*np.pi*frequency*t),validate = False),
    ["timestep"]
    )
    cos_tuple = (
        FunctionTransformer(lambda t: np.cos(2*np.pi*frequency*t), validate = False),
    ["timestep"]
    )
    ht = ColumnTransformer([sin_tuple,cos_tuple])
    return ht'''


