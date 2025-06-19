import tensorflow as tf
from tensorflow import keras
from keras import models , layers
from keras.optimizers import Adam,RMSprop
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error , mean_absolute_error,r2_score
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('src/launch_dataset_small.csv')

#Stage2 application 
# sns.scatterplot(data=df, x='distance_m', y='rpm', hue='angle_deg')
# plt.show()

X = df[['distance_m' , 'angle_deg']]
Y = df[ 'rpm']
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
Y_scaled = scaler_Y.fit_transform(Y.values.reshape(-1, 1))

x, x_test, y, y_test = train_test_split(X_scaled,Y_scaled,test_size=0.2,train_size=0.8)
x_train , x_cv , y_train , y_cv = train_test_split(x,y,test_size=0.25,train_size=0.75)

# print("The shapes of the dataset are just for checking ")
# print(df.shape ,x_train.shape, y_train.shape,x_cv.shape, y_cv.shape , x_test.shape , y_test.shape)

model = models.Sequential([
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])
early_stopper = EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)

#Tried fluctualting the learning rate , 0.001 seems right , 0.0001 is too small and even after 100 epochs didnt reach the optimal one
model.compile(optimizer=Adam(learning_rate=0.001) , loss = 'mse')
model.fit(x_train,y_train,epochs=100)

#Predicting using model and the ycv values and you can't go y_cv.values.reshape.. because it is already a numpy array format and not in pandas format
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


#Stage1 of checking 
#With 5 layers the model stoped at approx 35000 loss , trying to reduce the number of results

#Stage2 of checking 
#Now with the 4 layers still the same result, Lets try normalizing the dataset and reducing the number of layers and try seeing through graphs what might went wrong 
#Still not wroking properly lets move to Stage3 with rmse test to see the actual issue

#Stage 3
#The model is working fine but lets move to the thrid version, i tried increasing the synthetic dataset and included every possible possibility