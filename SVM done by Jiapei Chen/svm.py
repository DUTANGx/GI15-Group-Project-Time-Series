import pandas as pd
import numpy as np
import math
from sklearn.svm import SVR
import matplotlib.pyplot as plt
Location = r'C:\Users\meteo\Desktop\consumption_data.csv'

data = pd.read_csv(Location)

data = data.iloc[7:-22]
data = data.reset_index(drop=True)

data['DateTime'] = pd.to_datetime(data['DateTime'])
data['Year'] = [dat.year for dat in data['DateTime']]
data['active'] = data['active'].map(lambda x: math.log(x))

data = data.sort(['DateTime'],ascending = False)
data = data.reset_index(drop=True)

lag = 24
active = data.iloc[0:-lag,:]
active2008 = active[active['Year']==2008]
active2008 = active2008.drop(['reactive','intensity','Sub1','Sub2','Sub3','Year'],axis=1)

#16656 is the position of 2008 end in data
for i in range(1,lag+1):
    active2008["Lag"+str(i)] = data.iloc[16656+i:16656+i+active2008.shape[0],1].values

active2008 = active2008.sort(['DateTime'],ascending = True)
active2008 = active2008.reset_index(drop=True)

start = int(np.floor(active2008.shape[0] *0.9))
window=1
prediction =[]

while start + window <= active2008.shape[0]:
    train, valid = active2008.iloc[0:start,:], active2008.iloc[start:start+window,:]
    svr = SVR(kernel='rbf', C=1e2, gamma=0.001)
    svr.fit(train.iloc[:,2:], train.iloc[:,1])
    pred = svr.predict(valid.iloc[:,2:])
        
    prediction.append(math.exp(pred))        
    start += 1

prediction = pd.DataFrame(prediction)
prediction['active'] = prediction[0]
prediction = prediction.drop([0],axis=1)
real = active2008.iloc[(active2008.shape[0]-prediction.shape[0])-1:-1]
real['active'] = real['active'].map(lambda x:math.exp(x))

#Calculate MAPE
mape = 0.0
for i in range (0,prediction.shape[0]):
    mape += abs((real['active'].iloc[i] - prediction['active'].iloc[i])/real['active'].iloc[i])
mape = mape/prediction.shape[0]

#Calculate RMSE
rmse = 0.0
for i in range (0,prediction.shape[0]):
    rmse += math.pow((real['active'].iloc[i] - prediction['active'].iloc[i]),2)
rmse = rmse/prediction.shape[0]
rmse = math.sqrt(rmse)

#Plot graph for december
dec_r = real.iloc[real.shape[0]-24*31:,0:2]
dec_p = prediction.iloc[prediction.shape[0]-24*31:,:]
dec_p = dec_p.set_index(dec_r.iloc[:,0])
dec_r = dec_r.set_index('DateTime')

plt.subplots(figsize=(16,10))
plt.suptitle('Compare Original and Prediction')
plt.xlabel("Time")
plt.ylabel("Active")
plt.plot(dec_p, c = 'red', label = 'predict')
plt.plot(dec_r, c = 'green', label = 'real')
plt.hold('on')
plt.legend()
plt.savefig('svm.png', fmt='png', dpi=100,bbox_inches='tight' )
plt.show()

