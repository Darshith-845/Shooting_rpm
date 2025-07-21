#This one model is not entirely written by me, I had this senior in my club who helped me in writng the model 6 and model 7
# I have learned a lot 

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')
import time

df = pd.read_csv('augmented_reduced_large_unique_rpm.csv')
def remove_outliers_iqr(df, columns, factor=2.0):
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

df = remove_outliers_iqr(df, ['rpm', 'launch_speed_mps', 'distance_m', 'angle_deg'])

def create_advanced_features(df):
    df = df.copy()
    df['distance_m2'] = df['distance_m'] ** 2
    df['angle_deg2'] = df['angle_deg'] ** 2
    df['launch_speed_mps2'] = df['launch_speed_mps'] ** 2
    df['distance_angle'] = df['distance_m'] * df['angle_deg']
    df['distance_speed'] = df['distance_m'] * df['launch_speed_mps']
    df['angle_speed'] = df['angle_deg'] * df['launch_speed_mps']
    df['sin_angle'] = np.sin(np.radians(df['angle_deg']))
    df['cos_angle'] = np.cos(np.radians(df['angle_deg']))
    df['kinetic_energy'] = 0.5 * df['launch_speed_mps'] ** 2
    df['potential_energy'] = df['distance_m'] * df['sin_angle']
    df['speed_distance_ratio'] = df['launch_speed_mps'] / (df['distance_m'] + 1e-8)
    df['angle_distance_ratio'] = df['angle_deg'] / (df['distance_m'] + 1e-8)
    df['distance_speed_angle'] = df['distance_m'] * df['launch_speed_mps'] * df['angle_deg']
    df['distance_sin_angle'] = df['distance_m'] * df['sin_angle']
    for col in ['distance_m', 'launch_speed_mps', 'rpm']:
        if (df[col] > 0).all():
            df[f'log_{col}'] = np.log(df[col])
            df[f'sqrt_{col}'] = np.sqrt(df[col])
    return df

df = create_advanced_features(df)

def select_best_features(X, y, k=15):
    selector = SelectKBest(score_func=f_regression, k=k)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    feature_scores = pd.DataFrame({
        'feature': X.columns[selector.get_support()],
        'score': selector.scores_[selector.get_support()]
    }).sort_values('score', ascending=False)
    return selected_features, feature_scores

feature_columns = [col for col in df.columns if col not in ['rpm']]
X = df[feature_columns]
y = df['rpm']
y_log = np.log(y)
selected_features, feature_scores = select_best_features(X, y_log, k=15)
X_selected = X[selected_features]

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_selected, y_log, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42
)

scalers = {
    'MinMax': MinMaxScaler(),
    'Standard': StandardScaler(),
    'Robust': RobustScaler()
}

results = {}
best_scaler = None
best_score = float('inf')
best_metrics = None

for scaler_name, scaler in scalers.items():
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    models = {
        'XGBoost': xgb.XGBRegressor(
            n_estimators=100, 
            max_depth=5, 
            learning_rate=0.1,
            random_state=42, 
            n_jobs=-1,
            verbosity=0
        ),
        'LightGBM': lgb.LGBMRegressor(
            n_estimators=100, 
            random_state=42, 
            n_jobs=-1, 
            verbose=-1
        ),
        'RandomForest': RandomForestRegressor(
            n_estimators=100, 
            random_state=42, 
            n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=100, 
            random_state=42
        )
    }
    scaler_results = {}
    for model_name, model in models.items():
        model.fit(X_train_scaled, y_train)
        val_pred_log = model.predict(X_val_scaled)
        test_pred_log = model.predict(X_test_scaled)
        val_pred = np.exp(val_pred_log)
        test_pred = np.exp(test_pred_log)
        y_val_orig = np.exp(y_val)
        y_test_orig = np.exp(y_test)
        val_rmse = np.sqrt(mean_squared_error(y_val_orig, val_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test_orig, test_pred))
        scaler_results[model_name] = {
            'val_rmse': val_rmse,
            'test_rmse': test_rmse
        }
    ensemble_val_pred_log = np.mean([
        models[name].predict(X_val_scaled) for name in models.keys()
    ], axis=0)
    ensemble_test_pred_log = np.mean([
        models[name].predict(X_test_scaled) for name in models.keys()
    ], axis=0)
    ensemble_val_pred = np.exp(ensemble_val_pred_log)
    ensemble_test_pred = np.exp(ensemble_test_pred_log)
    y_val_orig = np.exp(y_val)
    y_test_orig = np.exp(y_test)
    ensemble_val_rmse = np.sqrt(mean_squared_error(y_val_orig, ensemble_val_pred))
    ensemble_test_rmse = np.sqrt(mean_squared_error(y_test_orig, ensemble_test_pred))
    scaler_results['Ensemble'] = {
        'val_rmse': ensemble_val_rmse,
        'test_rmse': ensemble_test_rmse
    }
    results[scaler_name] = scaler_results
    if ensemble_val_rmse < best_score:
        best_score = ensemble_val_rmse
        best_scaler = scaler_name
        best_metrics = scaler_results['Ensemble']

print(f"Best model: {best_scaler} scaler with Ensemble | Val RMSE: {best_metrics['val_rmse']:.2f} | Test RMSE: {best_metrics['test_rmse']:.2f}")

