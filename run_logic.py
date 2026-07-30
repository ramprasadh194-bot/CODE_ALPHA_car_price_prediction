import os
import glob
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

csv_files = glob.glob("*.csv")
if not csv_files:
    raise FileNotFoundError("No CSV file found in the current directory.")

data_file = csv_files[0]
print(f"Loading dataset: {data_file}")
df = pd.read_csv(data_file)
print("\n--- Dataset Shape ---")
print(df.shape)
print("\n--- Missing Values ---")
print(df.isnull().sum())

# Preprocessing
current_year = 2024
df["Car_Age"] = current_year - df["Year"]
df_model = df.drop(["Car_Name", "Year"], axis=1)

label_cols = ["Fuel_Type", "Selling_type", "Transmission"]
le = LabelEncoder()
for col in label_cols:
    df_model[col] = le.fit_transform(df_model[col])

# Model Training
X = df_model.drop("Selling_Price", axis=1)
y = df_model["Selling_Price"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
}

results = []
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results.append({"Model": name, "RMSE": rmse, "MAE": mae, "R2 Score": r2})

results_df = pd.DataFrame(results).sort_values(by="R2 Score", ascending=False)
print("\n--- Base Model Performance ---")
print(results_df.to_string(index=False))

# Hyperparameter Tuning
print("\n--- Tuning Random Forest ---")
param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [None, 10],
    "min_samples_split": [2, 5],
}

grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=param_grid,
    cv=5,
    n_jobs=1,
    scoring="r2",
    verbose=0,
)
grid_search.fit(X_train_scaled, y_train)

print(f"Best Parameters: {grid_search.best_params_}")
best_rf = grid_search.best_estimator_
y_pred_best = best_rf.predict(X_test_scaled)
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_best)):.4f}")
print(f"Test MAE:  {mean_absolute_error(y_test, y_pred_best):.4f}")
print(f"Test R2:   {r2_score(y_test, y_pred_best):.4f}")
