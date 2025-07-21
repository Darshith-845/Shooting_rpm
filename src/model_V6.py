#This one model is not entirely written by me, I had this senior in my club who helped me in writng the model 6 and model 7
# I have learned a lot 



import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
 
df = pd.read_csv('launch_dataset_small.csv')
 
rpm_low, rpm_high = df['rpm'].quantile([0.01, 0.99])
speed_low, speed_high = df['launch_speed_mps'].quantile([0.01, 0.99])
df = df[(df['rpm'] >= rpm_low) & (df['rpm'] <= rpm_high)]
df = df[(df['launch_speed_mps'] >= speed_low) & (df['launch_speed_mps'] <= speed_high)]
 
df['distance_m2'] = df['distance_m'] ** 2
df['angle_deg2'] = df['angle_deg'] ** 2
df['launch_speed_mps2'] = df['launch_speed_mps'] ** 2
df['distance_angle'] = df['distance_m'] * df['angle_deg']
df['distance_speed'] = df['distance_m'] * df['launch_speed_mps']
df['angle_speed'] = df['angle_deg'] * df['launch_speed_mps']
df['rpm_per_speed'] = df['rpm'] / df['launch_speed_mps']
df['distance_per_speed'] = df['distance_m'] / df['launch_speed_mps']
if (df['rpm'] > 0).all() and (df['launch_speed_mps'] > 0).all():
    df['log_rpm'] = np.log(df['rpm'])
    df['log_launch_speed_mps'] = np.log(df['launch_speed_mps'])
else:
    df['log_rpm'] = 0
    df['log_launch_speed_mps'] = 0
 
clustering_features = ['distance_m', 'angle_deg', 'launch_speed_mps']
kmeans = KMeans(n_clusters=4, random_state=42)
df['kmeans_cluster'] = kmeans.fit_predict(df[clustering_features])
 
binner = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile')
df['distance_bin'] = binner.fit_transform(df[['distance_m']]).astype(int)
df['angle_bin'] = binner.fit_transform(df[['angle_deg']]).astype(int)
df['speed_bin'] = binner.fit_transform(df[['launch_speed_mps']]).astype(int)
 
linreg = LinearRegression()
linreg.fit(df[clustering_features], df['log_rpm'])
df['linreg_residual'] = df['log_rpm'] - linreg.predict(df[clustering_features])
 
def jitter_df(df, n_aug=1, noise_scale=0.01):
    aug_rows = []
    for _ in range(n_aug):
        jittered = df.copy()
        for col in ['distance_m', 'angle_deg', 'launch_speed_mps']:
            jittered[col] += np.random.normal(0, noise_scale * df[col].std(), size=len(df))
        aug_rows.append(jittered)
    return pd.concat([df] + aug_rows, ignore_index=True)
df = jitter_df(df, n_aug=2, noise_scale=0.01)
 
features = [
    'distance_m', 'angle_deg', 'launch_speed_mps',
    'distance_m2', 'angle_deg2', 'launch_speed_mps2',
    'distance_angle', 'distance_speed', 'angle_speed',
    'rpm_per_speed', 'distance_per_speed', 'log_launch_speed_mps',
    'kmeans_cluster', 'distance_bin', 'angle_bin', 'speed_bin', 'linreg_residual'
]


Y = df['log_rpm']
X = df[features]
 
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
 
x, x_test, y, y_test = train_test_split(X_scaled, Y.values, test_size=0.2, train_size=0.8, random_state=42, shuffle=True)
x_train, x_cv, y_train, y_cv = train_test_split(x, y, test_size=0.25, train_size=0.75, random_state=42, shuffle=True)
from xgboost import XGBRegressor
xgb = XGBRegressor(n_estimators=200, random_state=42, n_jobs=-1)
xgb.fit(x_train, y_train)
xgb_val_pred_log = xgb.predict(x_cv)
xgb_test_pred_log = xgb.predict(x_test)
xgb_val_pred = np.exp(xgb_val_pred_log)
xgb_test_pred = np.exp(xgb_test_pred_log)
xgb_val_true = np.exp(y_cv)
xgb_test_true = np.exp(y_test)
xgb_val_rmse = np.sqrt(mean_squared_error(xgb_val_true, xgb_val_pred))
xgb_val_mae = mean_absolute_error(xgb_val_true, xgb_val_pred)
xgb_val_r2 = r2_score(xgb_val_true, xgb_val_pred)
xgb_test_rmse = np.sqrt(mean_squared_error(xgb_test_true, xgb_test_pred))
xgb_test_mae = mean_absolute_error(xgb_test_true, xgb_test_pred)
xgb_test_r2 = r2_score(xgb_test_true, xgb_test_pred)
print("[XGBoost] Validation RMSE:", xgb_val_rmse)
print("[XGBoost] Validation MAE:", xgb_val_mae)
print("[XGBoost] Validation R²:", xgb_val_r2)
print("[XGBoost] Test RMSE:", xgb_test_rmse)
print("[XGBoost] Test MAE:", xgb_test_mae)
print("[XGBoost] Test R²:", xgb_test_r2)