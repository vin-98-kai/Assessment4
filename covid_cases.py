# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 09:10:20 2022

@author: Calvin
"""

import os
import pickle
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
from module_covid_cases import EDA, ModelCreation
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error,mean_absolute_percentage_error

# Static
DATA_PATH = os.path.join(os.getcwd(),'data','cases_malaysia_train.csv')
CSV_PATH = os.path.join(os.getcwd(),'data','cases_malaysia_test.csv')
MMS_TRAIN_PATH = os.path.join(os.getcwd(),'models','mms_train.pkl')
MMS_TEST_PATH = os.path.join(os.getcwd(),'models','mms_test.pkl')
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'models','model.h5')
log_dir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
LOG_FOLDER_PATH = os.path.join(os.getcwd(), 'logs',log_dir)

# EDA
# Step 1) Data Loading
df =  pd.read_csv(DATA_PATH)
test_df = pd.read_csv(CSV_PATH)

# Step 2) Data Inspection
df.info() # need to change object to float
test_df.info() # nans @ cases_new

# Visualize data in line plots (df)
eda = EDA() #alot of noise and sudden missing nan in the cases_new column in df
eda.plot_graph(df)

# Step 3) Data Cleaning
# convert object to float64 
df['cases_new'] = pd.to_numeric(df['cases_new'],errors='coerce')
df.isna().sum() # can see 12 new missing values in cases_new

# apply interpolate on cases_new for it's nans for df
df['cases_new'] = df['cases_new'].interpolate(method='polynomial',order=2)
df.duplicated().sum() 
df.isna().sum()
df.info() # at this point df is now clean

# apply interpolate on cases_new for it's nan for test_set
test_df['cases_new'] = test_df['cases_new'].interpolate(method='polynomial',
                                                        order=2)
test_df.duplicated().sum()
test_df.isna().sum()
test_df.info() # at this point test_df is now clean

df = df['cases_new'].values
test_df = test_df['cases_new'].values

# Step 4) Features Selection 
# selecting cases_new only because we are predicting the new cases of covid

# Step 5) Data Preprocessing
mms = MinMaxScaler() # minmax scale it
scaled_df = mms.fit_transform(np.expand_dims(df,axis=-1))
with open(MMS_TRAIN_PATH,'wb') as file:
    pickle.dump(mms,file)

scaled_test_df = mms.transform(np.expand_dims(test_df,axis=-1))
with open(MMS_TEST_PATH,'wb') as file:
    pickle.dump(mms,file)

X_train = [] # initialize/declare
y_train = []

win_size = 30

# df: train
for i in range(win_size,np.shape(df)[0]):
    X_train.append(scaled_df[i-win_size:i,0])
    y_train.append(scaled_df[i,0])

X_train = np.array(X_train)
y_train = np.array(y_train)

temp = np.concatenate((scaled_df,scaled_test_df))
length_win = win_size+len(scaled_test_df)
temp = temp[-length_win:]

X_test = [] # initialize/declare
y_test = []

# test_df: test
for i in range(win_size,np.shape(temp)[0]):
    X_test.append(temp[i-win_size:i,0])
    y_test.append(temp[i,0])

X_test = np.array(X_test)
y_test = np.array(y_test)

X_train = np.expand_dims(X_train,axis=-1)
X_test = np.expand_dims(X_test,axis=-1)

#%% Model Development
mc = ModelCreation()
model = mc.simple_lstm_layer(X_train,num_node=32)

plot_model(model,show_layer_names=(True),show_shapes=(True))

model.compile(optimizer='adam',loss='mse',metrics='mse')

# callbacks
tensorboard_callbacks=TensorBoard(log_dir=LOG_FOLDER_PATH)
early_stopping_callbacks = EarlyStopping(monitor='loss',patience=3)

hist = model.fit(X_train,y_train,batch_size=128,epochs=100,
                 validation_data=(X_test,y_test),
                 callbacks=[tensorboard_callbacks,early_stopping_callbacks])

hist.history.keys()

#%% plotting graphs
plt.figure()
plt.title('LOSS')
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.show()

plt.figure()
plt.title('MSE')
plt.plot(hist.history['mse'])
plt.plot(hist.history['val_mse'])
plt.show()

#%% Model Evaluation
predicted = []

for test in X_test:
    predicted.append(model.predict(np.expand_dims(test,axis=0)))

predicted = np.array(predicted)

#%% Model Analysis
plt.figure()
plt.plot(predicted.reshape(len(predicted),1))
plt.plot(y_test)
plt.legend(['Predicted','Actual'])
plt.show()

y_true = y_test
y_pred = predicted.reshape(len(predicted),1)

print("mae manual calculation:",(mean_absolute_error(y_true,y_pred)
                                 /sum(abs(y_true)))*100)
print("mae:",mean_absolute_error(y_true,y_pred))
print("mape:",mean_absolute_percentage_error(y_true,y_pred),"\n")

#%% Inversion (GETTING SAME RESULT AFTER INVERSE)
y_true = mms.inverse_transform(np.expand_dims(y_test,axis=-1))
y_pred = mms.inverse_transform(predicted.reshape(len(predicted),1))

plt.figure()
plt.plot(y_pred)
plt.plot(y_true)
plt.legend(['Predicted','Actual'])
plt.show()

print("mae inverse:",mean_absolute_error(y_true,y_pred))
print("mape inverse:",mean_absolute_percentage_error(y_true,y_pred))

#%% Saving Model
model.save(MODEL_SAVE_PATH)

#%% Discussion
# Applied EDA on the training and testing datasets respectively
# Applied interpolate for both target datasets for cases_new column
# MinMaxScaled them, concatenate the scaled datasets into a temp variable
#set win_size to 30 append them into X_train,X_test,y_train,y_test respectively
# create model with callbacks and tensorboard
# at the plotting graphs shows abit of overfitting so it maybe wise to add more
#dropout, likewise the so is the final graphs that shows predicted and actual 
#new cases of covid 19.