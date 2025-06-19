#Version 1
#Okay when taking the distance as the input feature the loss comes upto approx 17000 and starts hovering over there
#The main reasons for this is for a single value of x
#So ill try using two input feature that is taking the current angle as the input and predicting the rpm which will be the VERSION2



import tensorflow as tf
from tensorflow import keras
from keras import models , layers
from keras.optimizers import Adam,RMSprop
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split

df = pd.read_csv('src/launch_dataset_small.csv')

X = df['distance_m']
Y = df[['angle_deg' , 'rpm']]

x, x_test, y, y_test = train_test_split(X,Y,test_size=0.2,train_size=0.8)
x_train , x_cv , y_train , y_cv = train_test_split(x,y,test_size=0.25,train_size=0.75)

# print("The shapes of the dataset are just for checking ")
# print(df.shape ,x_train.shape, y_train.shape,x_cv.shape, y_cv.shape , x_test.shape , y_test.shape)

model = models.Sequential([
    layers.Dense(520, input_dim =1,activation='relu'),
    layers.Dense(256,activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(2)
])
early_stopper = EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)

#Tried fluctualting the learning rate, 
model.compile(optimizer=Adam(learning_rate=0.0001) , loss = 'mse')

# #This is not working too
# model.compile(optimizer=RMSprop(learning_rate=0.0005) , loss = 'mse')

model.fit(x_train,y_train,epochs=100)
# print(f" The version of the tensorflow is this {tf.__version__}")