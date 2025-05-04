import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import datetime as dt
from dateutil.relativedelta import relativedelta

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import scipy.stats as stats

from statsmodels.tsa.stattools import adfuller, arma_order_select_ic
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

from arch import arch_model

"""----------------------------------------------------------------"""

st.title("Time Series Analysis on Kroger Stock Daily Returns")

st.markdown("""### 📝 Project Description

This project performs a time series analysis of **Kroger Co. (KR)** stock returns using classical models such as **ARMA** and **GARCH**. The goal is to investigate the structure of returns and volatility, build appropriate models, and assess their performance. The analysis includes:

- Retrieving historical price data from Yahoo Finance  
- Computing daily returns and conducting exploratory data analysis (EDA)  
- Estimating and diagnosing **ARMA models** for returns  
- Modeling volatility using **GARCH** on residuals from the ARMA model  
- Performing diagnostic tests, including the **Ljung-Box test** and **ARCH effects test**  
- Visualizing results  

The project is implemented in Python using `pandas`, `statsmodels`, `arch`, and `matplotlib`, and is presented through an interactive **Streamlit** app.
            """)




st.markdown("""***Start Date:*** 01-05-2025  
            \n***End Date:*** 01-05-2025  
            \n***Data:*** Daily Prices""")

"""----------------------------------------------------------------"""

#Download Data
tickers = ["KR"]
data_period = "5y"
data_interval = "1d" 
end_date = dt.datetime(2025, 5, 1)
start_date = end_date - relativedelta(years=5)

@st.cache_data
def load_data(ticker, start, end, interval="1d"):
    return yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=True)["Close"]
data = load_data(tickers, start_date, end_date)

prices = data.asfreq("D").ffill() 
log_prices = np.log(prices)
log_returns = np.log(prices / prices.shift(1)).dropna()
squared_log_returns = np.square(log_returns)

st.markdown(""":blue[
We plot the prices and returns to gain an initial understanding of the data's structure, including potential trends, seasonality, and volatility patterns. 
Over the 5-year period, Kroger's stock price shows a general upward trend, whereas the returns appear to fluctuate randomly around zero, 
resembling white noise—as typically expected for stock returns in an efficient market.]
            """)

"""----------------------------------------------------------------"""

#Plot Prices and Returns
st.subheader("Graph of Prices, Log Prices & Log Returns")
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15,10))

for i in range(0, len(axes)-1):
    axes[i].tick_params(bottom=False, labelbottom=False)
    
axes[0].plot(prices)
axes[0].set_title("Prices")

axes[1].plot(log_prices)
axes[1].set_title("Log Prices")

axes[2].plot(log_returns)
axes[2].set_title("Log Returns")
axes[2].set_xlabel("Date")
axes[2].xaxis.set_major_locator(mdates.YearLocator())
axes[2].xaxis.set_minor_locator(mdates.MonthLocator((1,4,7,10)))
axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
axes[2].xaxis.set_minor_formatter(mdates.DateFormatter("%b"))
plt.setp(axes[2].get_xticklabels(), rotation=0, ha="center")

plt.tight_layout()        
st.pyplot(fig)

"""----------------------------------------------------------------"""

#Check Stationarity of Returns
#Create a function for the ADF test to print the outputs clearly
def adf_test(input_data):
    result = adfuller(input_data)
    statistic = round(result[0], 2)
    pvalue = result[1] 
    conf_intervals = {key: round(value, 2) for key, value in result[4].items()} 

    return f"Augmented Dickey Fuller Test on Log Returns  \n Test Statistic: {statistic}  \n P-value: {pvalue}  \n Confidence Intervals: {conf_intervals}"

st.subheader("Check Stationarity of Returns using Augmented Dickey Fuller")
st.text(adf_test(log_returns))

st.markdown(""":blue[
The Augmented Dickey Fuller test shows a statistic value of -19.5 
with a p-value of approximately 0 which means that
we are comfortable rejecting the Null Hypothesis of non-stationarity at the 5% and 1% confidence intervals]
            """)

"""----------------------------------------------------------------"""

#Plot ACF and PACF of Returns
st.subheader("Graph of ACF & PACF of Log Returns. Check for Initial Suggestion of MA and AR Order Respectively")
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(20,10))
plot_acf(log_returns, title="ACF of Returns", ax=axes[0]) #To check for potential MA lag order
axes[0].tick_params(bottom=False, labelbottom=False)
plot_pacf(log_returns, title="PACF of Returns", ax=axes[1]) #To check for potentiam AR lag order
plt.tight_layout()
st.pyplot(fig)

st.markdown(""":blue[
The ACF of the log returns shows no significant lags except at lag 6 which could suggest 
a MA(6) model, but it is not a recurring feature which suggests that the autocorrelation 
can be attributed to random factors] \n
:blue[Similarly, the PACF of the log returns shows no significant lags except at lag 6 which could
be attributed to random factors, a common shock or data artifact instead of an AR(6) suggestion]
            """)

"""----------------------------------------------------------------"""

#Ljung Box on Returns
st.subheader("Ljung-Box an Box-Pierce Statistics on Log Returns for Joint Autocorrelation")
returns_lb = round(acorr_ljungbox(log_returns, boxpierce=True),2)
st.write(returns_lb)
st.markdown(""":blue[
The Ljung-Box statistic on log returns shows no significant autocorrelation except at lag 6
confirming our intuition from the visual inspection of the ACF and the PACF]
            """)

st.subheader("Statsmodels ARMA Order Suggestion Tool")
suggest_arma_order = arma_order_select_ic(log_returns, max_ar=4, max_ma=4, ic=("aic", "bic"))
st.markdown(
    f"**Suggested AIC ARMA order:** {suggest_arma_order.aic_min_order}  \n"
    f"**Suggested BIC ARMA order:** {suggest_arma_order.bic_min_order}"
)
st.write(":blue[The Statsmodels ARMA Suggestion Tool suggests an ARMA(1,1) based on the AIC and \
an ARMA(0,0) (i.e. White Noise) based on BIC]")

"""----------------------------------------------------------------"""

#Fit ARMA Model
st.subheader("Fit ARMA(1,1) on Log Returns and Check Model Outputs")
train_test_split = round(len(log_returns) * 0.8)
train_set = log_returns.iloc[:train_test_split]
test_set = log_returns.iloc[train_test_split:]

model = ARIMA(train_set, order=(1,0,1)).fit() #Fit an ARMA(1,1) model on returns 
st.write(model.summary())
st.markdown(""":blue[The estimated coefficients' p-values show that they are not significant at the
5% confidence interval. As such an ARMA model is not a good fit for the log returns of the stock 
This result is consistent with the with our findings from the earlier examination of the
ACF, PACF and the Ljung-Box statistic]""")

"""----------------------------------------------------------------"""

#Plot Fitted Values vs Actual Data
st.subheader("Graph of Actual Data vs ARMA(1,1) Fitted Values")
plt.figure(figsize=(15,6))
plt.plot(log_returns[train_test_split:], label="Actual Data", c="dimgrey")
plt.plot(train_set, label="Train Set", color="silver")
plt.plot(model.fittedvalues, label="ARMA(1,1) Fitted Values", color="firebrick")

plt.title("Actual Returns vs Model Fitted Returns")
plt.legend(fontsize=15)
plt.tight_layout()
st.pyplot(plt)

st.markdown(""":blue[The plot of the actual data and the fitted values shows clearly
that the ARMA(1,1) is not a fit to model the stock's returns and make forecasts. 
This result is consistent with our expectations from our previous tests]""")

"""----------------------------------------------------------------"""

#Residual Analysis
st.subheader("Residual Analysis & Plot")

st.write("Ljung-Box and Box-Pierce Statistics on Residuals")
residuals = model.resid
residual_lb = round(acorr_ljungbox(residuals, boxpierce=True),2)
st.write(residual_lb)


st.write("KDE Plot of Residuals. Test for Normality")
resid_skew = round(stats.skew(residuals), 2)
resid_kurt = round(stats.kurtosis(residuals), 2)
plt.figure(figsize=(15,6))
residuals.plot.kde(lw=5)
plt.title("KDE Plot of Residuals")
ax = plt.gca()
plt.text(0.85, 0.85, f"Skewness: {resid_skew} \n Kurtosis: {resid_kurt}", transform=ax.transAxes, fontsize=15)
plt.tight_layout()
st.pyplot(plt)

st.markdown(""":blue[The Ljung-Box and Box-Pierce statistics indicate no significant autocorrelation in the residuals, suggesting a good fit in terms of capturing serial dependence. 
The kernel density estimate (KDE) plot of the residuals resembles a normal distribution with mild positive skewness (0.3). 
However, the kurtosis is notably high at 9.9, indicating heavier tails than a normal distribution. These results are expected, 
as financial return series often exhibit leptokurtosis—extreme values occurring more frequently than under a normal distribution.]""")

"""----------------------------------------------------------------"""

#ARCH Effect Check
st.subheader("Check for ARCH Effects")
squared_returns = np.square(log_returns)
squared_residuals = np.square(residuals)

#Plot ACF/PACF of Squared Returns
st.write("Graph of Squared Returns and ACF/PACF of Squared Returns")
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15,10))
axes[0].plot(squared_returns)
axes[0].set_title("Squared Returns")
plot_acf(squared_returns, title="ACF of Squared Returns", ax=axes[1])
plot_pacf(squared_returns, title="PACF of Squared Returns", ax=axes[2])
plt.tight_layout()
st.pyplot(fig)

st.markdown(""":blue[Both the ACF and PACF of squared returns suggest a linear dependency on the first lag of the series but not for higher lags. This could be an indication for an ARCH(1) model]""")

#Plot ACF/PACF of Squared Residuals
st.write("Graph of ACF/PACF of Squared Residuals")
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15,10))
plot_acf(squared_residuals, title="ACF of Squared Residuals", ax=axes[0])
plot_pacf(squared_residuals, title="PACF of Squared Residuals", ax=axes[1])
plt.tight_layout()
st.pyplot(fig)

"""----------------------------------------------------------------"""

#Ljung Box on Squared Residuals
st.write("Ljung-Box and Box-Pierce Statistics on Squared Residuals")
sq_residual_lb = round(acorr_ljungbox(squared_residuals, boxpierce=True),2)
st.write(sq_residual_lb)

st.markdown(""":blue[The Ljung-Box and Box-Pierce statistics on squared residuals indicate no autocorrelation, contradicting the ACF/PACF suggestion]""")

"""----------------------------------------------------------------"""

#Fit GARCH Model
st.subheader("Fit GARCH(1,1) and Test Model Fit")
garch_model = arch_model(residuals, vol="GARCH", p=1, q=1, mean="Zero")
garch_result = garch_model.fit()
st.write(garch_result.summary())

st.markdown(""":blue[The coefficients of the fitted GARCH(1,1) are shown to be statistically significant. As such, we continue with assessing the fit of the model]""")


"""----------------------------------------------------------------"""

#Assess Model Fit
st.subheader("Model Fit Test using Squared Standardized Residuals")
garch_volatilities = garch_result.conditional_volatility
stand_resids = residuals / garch_volatilities
sq_stand_resids = np.square(stand_resids)

plot_acf(sq_stand_resids, title="ACF of Squared Standardized Residuals")
st.pyplot(plt)

st.write("Ljung-Box and Box-Pierce Statistics on Squared Standardized Residuals")
sq_stand_residual_lb = round(acorr_ljungbox(sq_stand_resids, boxpierce=True),2)
st.write(sq_stand_residual_lb)

st.markdown(""":blue[The ACF of the squared standardized residuals show no signs of dependencies. The Ljung-Box and Box-Pierce statistics support that claim.
As such, the GARCH(1,1) model seems a good fit for the conditional volatility of the stock]""")

st.subheader("Graph of Log Returns & Fitted Conditional Volatility")
plt.figure(figsize=(15,6))
plt.plot(garch_volatilities, label="Conditional Volatility", color="firebrick")
plt.plot(log_returns, label="Log Returns", c="dimgrey")

plt.legend()
plt.tight_layout()
st.pyplot(plt)