# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 12:22:54 2023

@author: ars16
"""

# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, median_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
import time



dataset_path = "preprocessed__.csv"  
df = pd.read_csv(dataset_path)


columns_to_drop = ['Title','Overview','Keywords','Runtime','Cast','Director','Editor','Poster URL','ImdbId','Release Date']
#Used columns:"vote_average', 'vote_count', 'vote_popularity', 'Genre values', 'Month', 'Day'"
df = df.drop(columns=columns_to_drop)
print(df.head())

X = df.drop('Income', axis=1)
y = df['Income']

# HOLD-OUT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
start_time = time.time()
linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)
linear_pred = linear_regressor.predict(X_test)
end_time = time.time()
linear_time = end_time - start_time
print(f"Linear Regression took {linear_time} seconds.")

# Decision Tree Regression
start_time = time.time()
dt_regressor = DecisionTreeRegressor(random_state=42)
dt_regressor.fit(X_train, y_train)
dt_pred = dt_regressor.predict(X_test)
end_time = time.time()
dt_time = end_time - start_time
print(f"Decision Tree Regression took {dt_time} seconds.")

# Random Forest Regression
start_time = time.time()
rf_regressor = RandomForestRegressor(random_state=42)
rf_regressor.fit(X_train, y_train)
rf_pred = rf_regressor.predict(X_test)
end_time = time.time()
rf_time = end_time - start_time
print(f"Random Forest Regression took {rf_time} seconds.")
# KNN Regression
start_time = time.time()
knn_regressor = KNeighborsRegressor(n_neighbors=5)
knn_regressor.fit(X_train, y_train)
knn_pred = knn_regressor.predict(X_test)
end_time = time.time()
knn_time = end_time - start_time
print(f"KNN Regression took {knn_time} seconds.")

# XGBoost Regression
start_time = time.time()
xgb_regressor = XGBRegressor(random_state=42)
xgb_regressor.fit(X_train, y_train)
xgb_pred = xgb_regressor.predict(X_test)
end_time = time.time()
xgb_time = end_time - start_time
print(f"XGBoost Regression took {xgb_time} seconds.")
print("  ")
# Evaluate and print metrics
def evaluate_model(name, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    evs = explained_variance_score(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)

    print(f"{name} Metrics:")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R-squared (R2): {r2}")
    print(f"Explained Variance Score (EVS): {evs}")
    print(f"Median Absolute Error (MedAE): {medae}")
    print("\n")

evaluate_model("Linear Regression", y_test, linear_pred)

evaluate_model("Decision Tree Regression", y_test, dt_pred)
evaluate_model("Random Forest Regression", y_test, rf_pred)
evaluate_model("KNN Regression", y_test, knn_pred)
evaluate_model("XGBoost Regression", y_test, xgb_pred)
# Bar charts 
models = ['Linear Regression', 'Decision Tree Regression', 'Random Forest Regression', 'KNN Regression']

mse_values = [mean_squared_error(y_test, linear_pred), mean_squared_error(y_test, dt_pred),
              mean_squared_error(y_test, rf_pred), mean_squared_error(y_test, knn_pred)]

mae_values = [mean_absolute_error(y_test, linear_pred), mean_absolute_error(y_test, dt_pred),
              mean_absolute_error(y_test, rf_pred), mean_absolute_error(y_test, knn_pred)]

r2_values = [r2_score(y_test, linear_pred), r2_score(y_test, dt_pred),
             r2_score(y_test, rf_pred), r2_score(y_test, knn_pred)]

evs_values = [explained_variance_score(y_test, linear_pred), explained_variance_score(y_test, dt_pred),
              explained_variance_score(y_test, rf_pred), explained_variance_score(y_test, knn_pred)]

medae_values = [median_absolute_error(y_test, linear_pred), median_absolute_error(y_test, dt_pred),
               median_absolute_error(y_test, rf_pred), median_absolute_error(y_test, knn_pred)]
models.append('XGBoost Regression')
mse_values.append(mean_squared_error(y_test, xgb_pred))
mae_values.append(mean_absolute_error(y_test, xgb_pred))
r2_values.append(r2_score(y_test, xgb_pred))
evs_values.append(explained_variance_score(y_test, xgb_pred))
medae_values.append(median_absolute_error(y_test, xgb_pred))
fig, axes = plt.subplots(5, 1, figsize=(10, 15))

axes[0].bar(models, mse_values, color='blue', alpha=0.7)
axes[0].set_title('Mean Squared Error (MSE)')

axes[1].bar(models, mae_values, color='green', alpha=0.7)
axes[1].set_title('Mean Absolute Error (MAE)')

axes[2].bar(models, r2_values, color='red', alpha=0.7)
axes[2].set_title('R-squared (R2)')

axes[3].bar(models, evs_values, color='purple', alpha=0.7)
axes[3].set_title('Explained Variance Score (EVS)')

axes[4].bar(models, medae_values, color='orange', alpha=0.7)
axes[4].set_title('Median Absolute Error (MedAE)')
plt.tight_layout()
plt.show()
# Step 2 hyperparameter tuning
from xgboost import XGBRegressor
from sklearn.metrics import make_scorer, mean_squared_error

xgb_regressor = XGBRegressor(random_state=42)

param_grid_xgb = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.2],
}

scoring = make_scorer(mean_squared_error, greater_is_better=False)
# cross-validation
grid_search_xgb = GridSearchCV(xgb_regressor, param_grid_xgb, cv=5, scoring=scoring, n_jobs=-1)

grid_search_xgb.fit(X_train, y_train)

best_params_xgb = grid_search_xgb.best_params_

best_model_xgb = grid_search_xgb.best_estimator_
y_pred_improved_xgb = best_model_xgb.predict(X_test)

mse_improved_xgb = mean_squared_error(y_test, y_pred_improved_xgb)
mae_improved_xgb = mean_absolute_error(y_test, y_pred_improved_xgb)
r2_improved_xgb = r2_score(y_test, y_pred_improved_xgb)
evs_improved_xgb = explained_variance_score(y_test, y_pred_improved_xgb)
medae_improved_xgb = median_absolute_error(y_test, y_pred_improved_xgb)

print("Improved XGBoost Regression Metrics:")
print(f"MSE: {mse_improved_xgb}")
print(f"MAE: {mae_improved_xgb}")
print(f"R-squared: {r2_improved_xgb}")
print(f"EVS: {evs_improved_xgb}")
print(f"MedAE: {medae_improved_xgb}")
#Original XGBoost metrics:
mse_xgb = mean_squared_error(y_test, xgb_pred)
mae_xgb = mean_absolute_error(y_test, xgb_pred)
r2_xgb = r2_score(y_test, xgb_pred)
evs_xgb = explained_variance_score(y_test, xgb_pred)
medae_xgb = median_absolute_error(y_test, xgb_pred)
models = ['Original XGBoost', 'Improved XGBoost']


metric_names = ['MSE', 'MAE', 'R-squared', 'EVS', 'MedAE']
original_metric_values = [mean_squared_error(y_test, xgb_pred), 
                          mean_absolute_error(y_test, xgb_pred), 
                          r2_score(y_test, xgb_pred), 
                          explained_variance_score(y_test, xgb_pred), 
                          median_absolute_error(y_test, xgb_pred)]

improved_metric_values = [mse_improved_xgb, mae_improved_xgb, r2_improved_xgb, evs_improved_xgb, medae_improved_xgb]

fig, axes = plt.subplots(len(metric_names), 1, figsize=(8, 10))

for i, metric_name in enumerate(metric_names):
    axes[i].bar(models, [original_metric_values[i], improved_metric_values[i]], color=['blue', 'orange'], alpha=0.7)
    axes[i].set_title(f'{metric_name} - Original vs Improved XGBoost Regression')

plt.tight_layout()
plt.show()