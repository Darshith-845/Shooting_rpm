#Trying a larger dataset

import tensorflow as tf
from tensorflow import keras
from keras import models , layers
from keras.optimizers import Adam,RMSprop
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
#StandardScalar is not working as the scaling and descaling of the rpm is leading to the plateauing of the rpm and lets use MinMaxScalar
# from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error , mean_absolute_error,r2_score
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('src/launch_dataset_large.csv')

#Stage2 application of V2
# sns.scatterplot(data=df, x='distance_m', y='rpm', hue='angle_deg')
# plt.show()

X = df[['distance_m' , 'angle_deg']]
Y = df[ 'rpm']
scaler_X = MinMaxScaler()
scaler_Y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
Y_scaled = scaler_Y.fit_transform(Y.values.reshape(-1, 1))

x, x_test, y, y_test = train_test_split(X_scaled,Y_scaled,test_size=0.2,train_size=0.8)
x_train , x_cv , y_train , y_cv = train_test_split(x,y,test_size=0.25,train_size=0.75)

# print("The shapes of the dataset are just for checking ")
# print(df.shape ,x_train.shape, y_train.shape,x_cv.shape, y_cv.shape , x_test.shape , y_test.shape)

model = models.Sequential([
    layers.Dense(128,activation = 'relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])
early_stopper = EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)

model.compile(optimizer=Adam(learning_rate=0.001) , loss = 'mse')
model.fit(x_train,y_train,epochs=100,callbacks=[early_stopper])

y_pred_scaled = model.predict(x_cv)
y_cv_real = scaler_Y.inverse_transform(y_cv.reshape(-1,1))
y_pred_real = scaler_Y.inverse_transform(y_pred_scaled)

#Evaluating the model
rmse_rpm = np.sqrt(mean_squared_error(y_cv_real, y_pred_real))
print("RMSE in RPM:", rmse_rpm)
print("MAE:", mean_absolute_error(y_cv_real, y_pred_real))
print("R²:", r2_score(y_cv_real, y_pred_real))
#For checking the installation of the tensorflow
# print(f" The version of the tensorflow is this {tf.__version__}")

#For understand better
plt.figure(figsize=(8,6))
plt.scatter(y_cv_real, y_pred_real, alpha=0.3)
plt.plot([y_cv_real.min(), y_cv_real.max()], [y_cv_real.min(), y_cv_real.max()], 'r--')
plt.xlabel("Actual RPM")
plt.ylabel("Predicted RPM")
plt.title("Predicted vs Actual RPM")
plt.grid(True)
plt.show()

#Stage1 Tried just increaseing the dataset but didn't work so now lets try increasing the layers and also decrease the learning rate for the stage 2

#Stage2 Changing the number of layers and the decrease in the learning rate didn't work is much, so lets change the scalling from standard scaller to minmax scallera and 
#get back the learning rate to 0.001

#Stage3 Tried scaling, with minmax, not much progress, lets try feature engineering in the version 4