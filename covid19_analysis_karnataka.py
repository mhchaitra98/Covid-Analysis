
import numpy as np
import pandas as pd
covid=pd.read_csv('datasets_covid_19_india.csv')

covid.head()
covid.info()
covid.isnull().sum()

df=covid.loc[(covid['State/UnionTerritory'] == 'Karnataka')]

df.head(10)

import seaborn as sns 
sns.countplot(x='ConfirmedIndianNational',data=df)

#import plotly.offline as py
#import plotly.graph_objs as go

df1=df[['Confirmed']]
df1=df1.values

train_size = int(len(df1) * 0.00)
test_size = len(df1) - train_size
train, test =df1[0:train_size,:], df1[train_size:len(df1),:]
print(len(train),len(test))


def create_dataset(dataset, look_back=1):
    dataX= [],[]
    dataY= [],[]
   
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


look_back = 2
trainX, trainY = create_dataset(train, look_back=look_back)
testX, testY = create_dataset(test, look_back=look_back)

from sklearn.linear_model import LinearRegression
model=LinearRegression()

model.fit(trainX,trainY)

predict1 =model.predict(testX)

df=pd.DataFrame({'Actual': testY.flatten(), 'Predicted':predict1.flatten()})
df

df.plot(kind='bar',figsize=(16,10))


