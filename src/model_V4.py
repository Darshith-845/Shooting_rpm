#Trying a larger dataset

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

df = pd.read_csv('src/launch_dataset_large.csv')

df['angle_rad'] = np.radians(df['angle_deg'])
df['cos_angle'] = np.cos(df['angle_rad'])
df['sin_angle'] = np.sin(df['angle_rad'])
df['distance_sq'] = df['distance_m'] ** 2
df['angle_sq'] = df['angle_deg'] ** 2
df['interaction'] = df['distance_m'] * df['angle_deg']

X = df[['distance_m', 'angle_deg', 'cos_angle', 'sin_angle', 'distance_sq', 'angle_sq', 'interaction']]
Y = df[ 'rpm']
scaler_X = MinMaxScaler()
scaler_Y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
Y_scaled = scaler_Y.fit_transform(Y.values.reshape(-1, 1))

x, x_test, y, y_test = train_test_split(X_scaled,Y_scaled,test_size=0.2,train_size=0.8)
x_train , x_cv , y_train , y_cv = train_test_split(x,y,test_size=0.25,train_size=0.75)

# model = models.Sequential([
#     Dense(128,activation = 'relu'),
#     Dense(64, activation='relu'),
#     Dense(32, activation='relu'),
#     Dense(1)
# ])

model = models.Sequential([
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1)
])
early_stopper = EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)

model.compile(optimizer=Adam(learning_rate=0.0005) , loss = 'mse')
model.fit(x_train,y_train,epochs=100,callbacks=[early_stopper])

y_pred_scaled = model.predict(x_cv)
y_cv_real = scaler_Y.inverse_transform(y_cv.reshape(-1,1))
y_pred_real = scaler_Y.inverse_transform(y_pred_scaled)

#Evaluating the model
rmse_rpm = np.sqrt(mean_squared_error(y_cv_real, y_pred_real))
print("RMSE in RPM:", rmse_rpm)
print("MAE:", mean_absolute_error(y_cv_real, y_pred_real))
print("R²:", r2_score(y_cv_real, y_pred_real))
print("Explained Variance Score:", explained_variance_score(y_cv_real, y_pred_real))
#For checking the installation of the tensorflow
# print(f" The version of the tensorflow is this {tf.__version__}")

#Stage 1
#Started with mse, still the same errors, lets try Batch normalization and droupout to check whether overfitting or underfiting

#Stage2
#Getting worse, the explained variance is negative, which is worse than just predicting the mean, so lets change the entire model to random forest in version 5