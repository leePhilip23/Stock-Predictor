#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd

df = pd.read_csv('GOOG.csv')
df


# In[6]:


import datetime

# Convert To Date Object
def convertToDate(s):
    date = s.split('-')
    year, month, day = int(date[0]), int(date[1]), int(date[2])
    return datetime.datetime(year = year, month = month, day = day)


# In[7]:


df['Date'] = df['Date'].apply(convertToDate)
df.index = df.pop('Date')
df


# In[8]:


import matplotlib.pyplot as plt

plt.plot(df.index, df['Close'])


# In[70]:


# Since it's LSTM model convert to supervised learning problem
import numpy as np

def convertToSupervised(dataframe, first_date_str, last_date_str, n = 3):
    firstDate = first_date_str.split('-')
    lastDate = last_date_str.split('-')
    firstYear, firstMonth, firstDay = int(firstDate[0]), int(firstDate[1]), int(firstDate[2]) + 6
    lastYear, lastMonth, lastDay = int(lastDate[0]), int(lastDate[1]), int(lastDate[2])
    first_date = datetime.datetime(year = firstYear, month = firstMonth, day = firstDay)
    last_date  = datetime.datetime(year = lastYear, month = lastMonth, day = lastDay)

    target_date = first_date

    dates = []
    X, Y = [], []

    last_time = False
    while True:
        df_subset = dataframe.loc[:target_date].tail(n+1)
        #print(f'{dataframe.loc[:target_date].tail(n)}')

        if len(df_subset) != n+1:
            print(f'Error: Window size {n} is too large for date {target_date}')
            return

        values = df_subset['Close'].to_numpy()
        x, y = values[:-1], values[-1]

        dates.append(target_date)
        X.append(x)
        Y.append(y)

        next_week = dataframe.loc[target_date:target_date + datetime.timedelta(days = 7)]
        next_datetime_str = str(next_week.head(2).tail(1).index.values[0])
        next_date_str = next_datetime_str.split('T')[0]
        year_month_day = next_date_str.split('-')
        year, month, day = year_month_day
        next_date = datetime.datetime(day = int(day), month = int(month), year = int(year))

        if last_time:
            break

        target_date = next_date
        
        if target_date >= last_date:
            last_time = True

    ret_df = pd.DataFrame({})
    ret_df['Target Date'] = dates
        
    X = np.array(X)
    for i in range(0, n):
        X[:, i]
        if n - i == 1:
            ret_df[f'{n - i} Day Before'] = X[:, i]
        else:
            ret_df[f'{n - i} Days Before'] = X[:, i]

    ret_df['Target'] = Y

    return ret_df

minTime = str(min(df.index))
maxTime = str(max(df.index))
priceWindow = convertToSupervised(df, minTime[:10], maxTime[:10], 3)

priceWindow


# In[71]:


# Turn "Days Before" into input and "Target" into output to
# feed into Tensorflow

def changeToArrValues(dataFrameWindow):
    df_as_np = dataFrameWindow.to_numpy()
    
    dates = df_as_np[:, 0]
    
    midMatrix = df_as_np[:, 1:-1]
    
    X = midMatrix.reshape((len(dates), midMatrix.shape[1], 1))
    Y = df_as_np[:, -1]
    
    return dates, X.astype(np.float32), Y.astype(np.float32)

dates, X, Y = changeToArrValues(priceWindow)

dates.shape, X.shape, Y.shape


# In[72]:


q_80 = int(len(dates) * .8)
q_90 = int(len(dates) * .9)

dates_train, x_train, y_train, = dates[:q_80], X[:q_80], Y[:q_80]

dates_val, x_val, y_val = dates[q_80:q_90], X[q_80:q_90], Y[q_80:q_90]
dates_test, x_test, y_test = dates[q_90:], X[q_90:], Y[q_90:]

plt.plot(dates_train, y_train)
plt.plot(dates_val, y_val)
plt.plot(dates_test, y_test)

plt.legend(['Train', 'Validation', 'Test'])


# In[73]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers

model = Sequential([layers.Input((3, 1)),
                    layers.LSTM(64),
                    layers.Dense(32, activation='relu'),
                    layers.Dense(32, activation='relu'),
                    layers.Dense(1)])

model.compile(loss='mse', 
              optimizer = Adam(learning_rate = 0.001),
              metrics=['mean_absolute_error'])

model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100)


# In[74]:


train_predictions = model.predict(x_train).flatten()

plt.plot(dates_train, train_predictions)
plt.plot(dates_train, y_train)
plt.legend(['Training Predictions', 'Training Observations'])


# In[75]:


val_predictions = model.predict(x_val).flatten()
plt.plot(dates_val, val_predictions)
plt.plot(dates_val, y_val)
plt.legend(['Validation Predictions', 'Validation Observations'])


# In[76]:


test_predictions = model.predict(x_test).flatten()

plt.plot(dates_test, test_predictions)
plt.plot(dates_test, y_test)
plt.legend(['Testing Predictions', 'Testing Observations'])


# In[77]:


plt.plot(df.index, df['Close'])
plt.plot(dates_train, train_predictions)
plt.plot(dates_val, val_predictions)
plt.plot(dates_test, test_predictions)
plt.legend(['Actual Price',
            'Training Predictions',
            'Validation Predictions',
            'Testing Predictions'])

