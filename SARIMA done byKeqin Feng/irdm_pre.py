# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 16:06:17 2016

@author: andyf
"""

from __future__ import print_function
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sms

#Location = r'/Users/andyf/Desktop/Year4/IRDM_CW/Group/household_power_consumption.txt'

Location = r'/Users/andyf/Desktop/Year4/IRDM_CW/Group/consumption_data.csv'
consumption_data = pd.read_csv(Location,parse_dates=True)
consumption_data['DateTime'] = consumption_data['DateTime'].apply(lambda x: pd.datetime.strptime(x, '%d/%m/%Y %H:%M'))
consumption_data = consumption_data.set_index(consumption_data['DateTime'])
'''
data['Global_active_power'] = data['Global_active_power'].convert_objects(convert_numeric=True)
data['Time'] = data['Time'].apply(lambda x: pd.datetime.strptime(x, '%H:%M:%S'))

data['HourOfDay'] = [dat.hour for dat in data['Time']]
data['HourOfDay'] = data['HourOfDay'].apply(str)
data['DateTime'] = data['Date'] + ' ' + data['HourOfDay']
data['DateTime'] = data['DateTime'].apply(lambda x: pd.datetime.strptime(x, '%d/%m/%Y %H'))

data = data.drop(['Date','HourOfDay'],axis=1)


data= data.convert_objects(convert_numeric=True)
consumption_data = data.groupby(['DateTime']).agg(['sum'])
consumption_data.columns = ['active','reactive','voltage','intensity','Sub1','Sub2','Sub3']
consumption_data = consumption_data.drop(['voltage'],axis=1)
#consumption_data = np.log(consumption_data)

consumption_data = consumption_data.to_csv('consumption_data.csv')
'''

plt.subplots(figsize=(14,10))
plt.xlabel("Time")
plt.ylabel("Usage")
plt.plot(consumption_data['active'].iloc[-365:],  label='active')
plt.hold('on')
plt.legend()
plt.savefig( 'compare.png', fmt='png', dpi=100, bbox_inches='tight')
plt.show()

