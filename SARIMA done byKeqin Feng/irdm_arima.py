
from __future__ import print_function
import numpy as np
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sms

Location = r'/Users/andyf/Desktop/Year4/IRDM_CW/Group/consumption_data_winter.csv'
'''
Location2 = r'/Users/andyf/Desktop/Year4/IRDM_CW/Group/consumption_data2008.csv'
data2 = pd.read_csv(Location2,parse_dates=True)
data2['DateTime'] = data2['DateTime'].apply(lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M'))
'''
data = pd.read_csv(Location,parse_dates=True)
data['DateTime'] = data['DateTime'].apply(lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M'))

active = data[['active']]
date = data2['DateTime'].iloc[-4783:]
active = active.set_index(date)

#active = active.iloc[-2928:]
active = np.log(active)

active_diff = active.diff()
active_diff = active_diff.iloc[1:]

decomposition = sm.tsa.seasonal_decompose(active_diff['active'].iloc[4039:].values,freq=24)
fig = plt.figure()  
fig = decomposition.plot()  
fig.set_size_inches(15, 8)

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(active_diff['active'], lags=24, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(active_diff['active'], lags=24, ax=ax2)


forecast =[]
sarima_forecast = pd.DataFrame()
for i in range(0,31):
    mod = sm.tsa.statespace.SARIMAX(active.iloc[0:-744+(i*24)], trend='t', order=(1,0,0), seasonal_order=(0,1,1,24))
    results = mod.fit()
    print (results.summary())
    forecast.append(results.predict(start=4039+(i*24) , end=4062+(i*24), dynamic=True, typ='levels'))


for i in range(0,31):
    sarima_forecast = sarima_forecast.append(pd.DataFrame(forecast[i].tolist()))
    

sarima_forecast = sarima_forecast.set_index(data['DateTime'].iloc[-744:])
sarima_forecast.columns = ['active']

'''
#qqplot, ACF & PACF of residue
resid = results.resid
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
fig = sm.qqplot(resid, line='q', ax=ax, fit=True)

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(resid.values.squeeze(), lags=24, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(resid, lags=24, ax=ax2)
'''

active = np.exp(active)
sarima_forecast = np.exp(sarima_forecast)

real = pd.DataFrame(active['active'].iloc[4039:])
real = real.set_index(data['DateTime'].iloc[-744:])

rmse = []
for i in range(0,31):
    rmse.append(sm.tools.eval_measures.rmse(real.iloc[0+i*24:24+i*24],sarima_forecast.iloc[0+i*24:24+i*24]))
rmse = np.mean(rmse)

mape = 0.0
for i in range(0,sarima_forecast.shape[0]):
    mape += np.abs((real.iloc[i] - sarima_forecast.iloc[i])/real.iloc[i])
mape = mape / sarima_forecast.shape[0] * 100

plt.subplots(figsize=(16,10))
plt.xlabel("Time")
plt.ylabel("Consumption")
plt.plot(real,c='red', label='Original Data')
plt.plot(sarima_forecast, c='green', label='Prediction')
plt.hold('on')
plt.legend()
plt.savefig( '11.png', fmt='png', dpi=100,bbox_inches='tight' )
plt.show()
