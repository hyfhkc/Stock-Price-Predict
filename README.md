# Stock-Price-Predict
#### Package install 
In this project, we used the packages and API below, please install them before running the code
* itertools
* numpy
* pandas
* matplotlib
* pandas_datareader
* statsmodels
* fix_yahoo_finance
* warnings

## Stock Price Prediction Project
* mission: 
    1. Use the API to download Alibaba stocks from Yahoo Finance from 2013 to the present
    2. Get the descriptive information of Alibaba stock
    3. Predicting the future price of Alibaba stock by time series model
    
#### Resources: Yahoo! Finance 
#### API to Download Financial Data
https://pypi.org/project/fix-yahoo-finance/
https://finance.yahoo.com/quote/AAPL?p=AAPL&.tsrc=fin-srch

## Step 1 - Download Data
### Download Alibaba stocks data from 2013 -01-01 to 2019-03-19 and save as dataframe

```
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
yf.pdr_override() # <== that's all it takes :-)
data = pdr.get_data_yahoo("BABA", start="2013-01-01", end="2019-03-19")
```

## Step 2 - Data Exploring
### Check whether there is any missing values
```
data.isnull().sum()
```
### viewing the descriptive information about the Alibaba stock history data  
```
data.describe()
```
## Resample data by week
* Reduce the fluctuation of data and better grasp the law of its change
* We take the closing price of the stock as the analysis target
* We resample the closing price in weeks and calculate the weekly average
```
y = data['Close'].resample('W').mean()
```
### Check the top ten data 
```
y[:10]
```
## Step 3 - Visualizing Time Series Data
```
y.plot(figsize = (15, 6))
plt.show
```
### Detecting the trend<rb>
   Decompose the total change trend
* Observed trend
* Overall trend
* Seasonal trend
* Unpredictble influences

- We can visualize our data using a method called time-series decomposition that allows us to decompose our time series into three distinct components: trend, seasonality, and noise
```
from pylab import rcParams
rcParams['figure.figsize'] = 30, 12
decomposition = sm.tsa.seasonal_decompose(y, model='additive')
fig = decomposition.plot()
plt.show()
```

## Step 4 - Time series forecasting with ARIMA
- We are going to apply one of the most commonly used method for time-series forecasting, known as ARIMA, which stands for Autoregressive Integrated Moving Average
ARIMA models are denoted with the notation ARIMA(p, d, q). These three parameters account for seasonality, trend, and noise in data
* First set the value range of the three parameters
* Generate all parameter combinations via itertools
* Group all parameters into the model

### we test our method here
```
p = d = q = range(0, 2)
pdq = list (itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in pdq]

print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
```
```
This step is parameter Selection for our stock’s price ARIMA Time Series Model. Our goal here is to use a “grid search” to find the optimal set of parameters that yields the best performance for our model

p = d = q = range(0, 3)
pdq = list (itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in pdq]

warnings.filterwarnings("ignore") # specify to ignore warning messages

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try: 
            mod = sm.tsa.statespace.SARIMAX(y,
                                                               order=param,
                                                               seasonal_order=param_seasonal,
                                                               enforce_stationarity=False,
                                                               enforce_invertibility=False)
            results = mod.fit()
            
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue
```
```
* The above output suggests that SARIMAX(2, 1, 2)x(0, 2, 2, 12) yields the lowest AIC value of 1153.8261167531941. Therefore we should consider this to be optimal option

mod = sm.tsa.statespace.SARIMAX(y,
                                order=(2, 1, 2),
                                seasonal_order=(0, 2, 2, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit()

print(results.summary().tables[1])
```

### Check the performance of model
* We should always run model diagnostics to investigate any unusual behavior
```
results.plot_diagnostics(figsize=(16, 8))
plt.show()
```

### Validating forecasts
To help us understand the accuracy of our forecasts, we compare predicted Alibaba stock price to real price of the time series, and we set forecasts to start at 2018–04–29 to the end of the data

```
pred = results.get_prediction('2018-04-29','2019-03-17', dynamic=False)
pred_ci = pred.conf_int()

ax = y['2016':].plot(label='observed')

pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))

ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)

ax.set_xlabel('Date')
ax.set_ylabel('Alibaba stock price')

plt.legend()
plt.show()
```

## Step 5 : Check the performance of the model
* In statistics, the mean squared error (MSE) of an estimator measures the average of the squares of the errors — that is, the average squared difference between the estimated values and what is estimated. The MSE is a measure of the quality of an estimator — it is always non-negative, and the smaller the MSE, the closer we are to finding the line of best fit

```
y_forecasted = pred.predicted_mean
y_truth = y['2017-01-01':]
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))
```

## Step6: Producing and visualizing forecasts
# Let the model predict the trend of Alibaba stock prices over the next 40 weeks and output the results 
```
pred_uc = results.get_forecast(steps=40)
pred_ci = pred_uc.conf_int()

# plot history price trend 
ax = y.plot(label='observed', figsize=(14, 7))
# plot the prediction trend 
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
# fill the confidence interval 
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)

# set x and y labels 
ax.set_xlabel('Date')
ax.set_ylabel('Alibaba stock price')

# show the image 
plt.legend()
plt.show()
```
