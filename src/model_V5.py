#Trying using the random forest for the model and lets not scale here

import tensorflow as tf
from tensorflow import keras
from keras import models
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam,RMSprop
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error , mean_absolute_error,r2_score ,explained_variance_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('src/launch_dataset_large.csv')

df['angle_rad'] = np.radians(df['angle_deg'])
df['cos_angle'] = np.cos(df['angle_rad'])
df['sin_angle'] = np.sin(df['angle_rad'])
df['distance_sq'] = df['distance_m'] ** 2
df['angle_sq'] = df['angle_deg'] ** 2
df['interaction'] = df['distance_m'] * df['angle_deg']

X = df[['distance_m', 'angle_deg', 'interaction']]
Y = df[ 'rpm']
# scaler_X = MinMaxScaler()
# scaler_Y = MinMaxScaler()

# X_scaled = scaler_X.fit_transform(X)
# Y_scaled = scaler_Y.fit_transform(Y.values.reshape(-1, 1))

x, x_test, y, y_test = train_test_split(X,Y,test_size=0.2,train_size=0.8)
x_train , x_cv , y_train , y_cv = train_test_split(x,y,test_size=0.25,train_size=0.75)

model = RandomForestRegressor()
model.fit(x_train, y_train)
y_pred = model.predict(x_cv)

#not applicable for tree based models
# early_stopper = EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)

y_pred = model.predict(x_cv)

#Evaluating the model
rmse_rpm = np.sqrt(mean_squared_error(y_cv, y_pred))
print("RMSE in RPM:", rmse_rpm)
print("MAE:", mean_absolute_error(y_cv, y_pred))
print("R²:", r2_score(y_cv, y_pred))
print("Explained Variance Score:", explained_variance_score(y_cv, y_pred))
#For checking the installation of the tensorflow
# print(f" The version of the tensorflow is this {tf.__version__}")

#Stage1
#The condition worsened for this data and this model is not seeming to working, Ill try some other models 