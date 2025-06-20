# src/models/model_trainer.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import pickle
import os

print("Starting model training script...")

# --- Configuration ---
DATA_PATH = '../../data/processed_data.csv' # Assuming preprocessed data is here
MODEL_SAVE_PATH = 'tuned_liquidity_model.pkl'
SCALER_SAVE_PATH = 'liquidity_scaler.pkl'
FEATURES = ['24h_volume', 'mkt_cap', '1h_change', 'price'] # Example features
TARGET = 'liquidity'

# --- 1. Load Data ---
try:
    # In a real scenario, you'd load preprocessed data from notebooks
    # For this conceptual script, let's create dummy data if file doesn't exist
    if not os.path.exists(DATA_PATH):
        print(f"'{DATA_PATH}' not found. Generating dummy data for demonstration.")
        num_samples = 1000
        data = {
            '24h_volume': np.random.rand(num_samples) * 1e9,
            'mkt_cap': np.random.rand(num_samples) * 1e11,
            '1h_change': np.random.randn(num_samples) * 5,
            'price': np.random.rand(num_samples) * 1000,
            'liquidity': np.random.rand(num_samples) # Target variable 0-1
        }
        df = pd.DataFrame(data)
        # Create a dummy processed_data.csv
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
        print(f"Dummy data saved to '{DATA_PATH}'")
    else:
        df = pd.read_csv(DATA_PATH)
        print(f"Data loaded successfully from '{DATA_PATH}'")

    X = df[FEATURES]
    y = df[TARGET]

except FileNotFoundError:
    print(f"Error: Data file not found at {DATA_PATH}. Please ensure your data processing notebooks generate this file.")
    exit()
except Exception as e:
    print(f"An error occurred during data loading: {e}")
    exit()

# --- 2. Split Data ---
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# --- 3. Feature Scaling ---
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Features scaled.")

# --- 4. Model Training (Random Forest Regressor) ---
print("Initializing Random Forest Regressor...")
model = RandomForestRegressor(random_state=42)

# --- 5. Hyperparameter Tuning (GridSearchCV) ---
print("Starting GridSearchCV for hyperparameter tuning (this may take a while)...")
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10],
    'min_samples_split': [2, 5]
}
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='r2')
grid_search.fit(X_train_scaled, y_train)

best_model = grid_search.best_estimator_
print(f"Best parameters found: {grid_search.best_params_}")
print(f"Best cross-validation R2 score: {grid_search.best_score_:.4f}")

# --- 6. Model Evaluation ---
print("Evaluating the best model on the test set...")
y_pred = best_model.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"Test Set R-squared: {r2:.4f}")
print(f"Test Set Mean Absolute Error (MAE): {mae:.4f}")

# --- 7. Save Model and Scaler ---
print("Saving trained model and scaler...")
try:
    # Ensure the directory exists
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    with open(MODEL_SAVE_PATH, 'wb') as model_file:
        pickle.dump(best_model, model_file)
    print(f"Model saved to '{MODEL_SAVE_PATH}'")

    with open(SCALER_SAVE_PATH, 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)
    print(f"Scaler saved to '{SCALER_SAVE_PATH}'")

except Exception as e:
    print(f"Error saving model/scaler: {e}")

print("Model training script finished.")
