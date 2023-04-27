import streamlit as st
import pandas as pd
import numpy as np
import yfinance as fn
from keras.models import load_model
import matplotlib.pyplot as plt
import datetime
from numpy import array

st.title("Stock Price Prediction")

st.sidebar.subheader('Query parameters')
start_date = st.sidebar.date_input("Start date", datetime.date(2002, 1, 1))
end_date = st.sidebar.date_input("End date", datetime.date(2021, 1, 31))


ticker_list = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/s-and-p-500-companies/master/data/constituents_symbols.txt')
tickerSymbol = st.sidebar.selectbox('Stock ticker', ticker_list) # Select ticker symbol
tickerData = fn.Ticker(tickerSymbol) 
df = tickerData.history(period='1d', start=start_date, end=end_date)



string_name = tickerData.info['longName']
st.header('**%s**' % string_name)

string_summary = tickerData.info['longBusinessSummary']
st.info(string_summary)

st.subheader('DATASET')
st.write(df)

st.subheader('Opening price vs Time chart')
fig = plt.figure(figsize = (13,6))
plt.xlabel('Date', fontsize=18)
plt.ylabel('Open Price', fontsize=18)
plt.plot(df.Open)
st.pyplot(fig)


st.subheader('Opening price vs Time chart with 150 Moving Average')
ma150 = df.Open.rolling(150).mean()
fig = plt.figure(figsize = (13,6))
plt.plot(ma150 , 'brown')
plt.plot(df.Open)
st.pyplot(fig)


st.subheader('Opening price vs Time chart with 150MA & 300MA')
ma300 = df.Open.rolling(300).mean()
fig = plt.figure(figsize = (13,6))
plt.plot(ma150 , 'brown')
plt.plot(ma300 , 'green')
plt.plot(df.Open)
st.pyplot(fig)

training_data = pd.DataFrame(df['Open'][0 : int(len(df)*.70)])
testing_data = pd.DataFrame(df['Open'][int(len(df)*.70) : int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
training_data_arr = scaler.fit_transform(training_data)
testing_data_arr = scaler.fit_transform(testing_data)



model = load_model('Keras_model.h5')

past_150_days = training_data.tail(150)
final_df = past_150_days.append(testing_data , ignore_index = True)

input_data = scaler.fit_transform(final_df)


X_test = []
Y_test = []

for i in range(150 , input_data.shape[0]):
    X_test.append(input_data[i-150:i])
    Y_test.append(input_data[i , 0])

X_test , Y_test = np.array(X_test) , np.array(Y_test)

y_predicted = model.predict(X_test)
scaler = scaler.scale_

scale_fact = 1/scaler[0]
y_predicted = y_predicted * scale_fact
Y_test =  Y_test * scale_fact


st.subheader('Prediction vs Original')
fig2 = plt.figure(figsize = (12,6))
plt.title( 'Company - Model prediciton comparison')
plt.plot(Y_test  , 'green' , label = 'Original price')
plt.plot(y_predicted , 'r' , label = 'Predicted price')
plt.xlabel('Time', fontsize=18)
plt.ylabel('Price', fontsize=18)
plt.legend()
st.pyplot(fig2)

X_input = testing_data[len(testing_data)-150:]
scaler = MinMaxScaler(feature_range=(0,1))
last_150_days_scaled = scaler.fit_transform(X_input)

x_test = []
x_test.append(last_150_days_scaled)
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))
pred_price = model.predict(x_test)
pred_price=scaler.inverse_transform(pred_price)
st.write("Opening price of tomorrow is" , pred_price[0,0])



temp_input = last_150_days_scaled
lst_output=[]
n_steps=149
i=0
while(i<30):
    
    if(len(temp_input)>100):
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input = np.append(temp_input, yhat[0])
        temp_input=temp_input[1:]
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input = np.append(temp_input, yhat[0])
        temp_input=temp_input[1:]
        lst_output.extend(yhat.tolist())
        i=i+1

lst_output = scaler.inverse_transform(lst_output)

st.subheader('Price of future 30 days')
arr = np.array(lst_output)
df1 = pd.DataFrame(arr, columns=["OPEN"])
df1['Day'] = range(1,31)
st.write(df1)

st.subheader('Plot of future 30 days')
fig3 = plt.figure(figsize = (12,6))
day_pred=np.arange(151,181)
plt.xlabel('Time', fontsize=18)
plt.ylabel('Price', fontsize=18)
plt.plot(day_pred,lst_output)
st.pyplot(fig3)

st.subheader('Combined with future 30 days prediction')
fig4 = plt.figure(figsize = (12,6))
df3=df['Open'].tolist()
df3.extend(lst_output)
plt.xlabel('Time', fontsize=18)
plt.ylabel('Price', fontsize=18)
plt.plot(df3 , 'g')
st.pyplot(fig4)

