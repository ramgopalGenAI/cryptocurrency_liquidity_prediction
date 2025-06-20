# src/models/model_trainer.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import pickle
import os
import sys

def create_dummy_data(num_samples=1000, data_path='../../data/processed_data.csv'):
    """
    Generates dummy data for demonstration purposes if the specified data file
    does not exist. This simulates the output of a data preprocessing step.
    """
    print(f"'{data_path}' not found. Generating dummy data for demonstration.")
    data = {
        '24h_volume': np.random.rand(num_samples) * 1e9,   # Volume in billions
        'mkt_cap': np.random.rand(num_samples) * 1e11,     # Market Cap in hundreds of billions
        '1h_change': np.random.randn(num_samples) * 5,     # 1-hour change percentage (can be negative)
        'price': np.random.rand(num_samples) * 1000,       # Price up to 1000
        'liquidity': np.random.rand(num_samples)          # Target variable: liquidity (0 to 1)
    }
    df = pd.DataFrame(data)

    # Ensure the directory for the dummy data exists
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    df.to_csv(data_path, index=False)
    print(f"Dummy data saved to '{data_path}' with {num_samples} samples.")
    return df

def load_data(data_path, features, target):
    """
    Loads data from the specified path, attempting to generate dummy data if not found.
    Extracts features (X) and target (y).
    """
    if not os.path.exists(data_path):
        df = create_dummy_data(data_path=data_path)
    else:
        try:
            df = pd.read_csv(data_path)
            print(f"Data loaded successfully from '{data_path}'")
        except Exception as e:
            print(f"Error reading data from '{data_path}': {e}")
            sys.exit(1) # Exit if real data cannot be loaded

    if not all(col in df.columns for col in features + [target]):
        print(f"Error: Missing required columns in data. Expected: {features + [target]}, Found: {df.columns.tolist()}")
        sys.exit(1)

    X = df[features]
    y = df[target]
    return X, y

def train_model(X_train, y_train):
    """
    Trains a RandomForestRegressor model with hyperparameter tuning using GridSearchCV.
    """
    print("Initializing Random Forest Regressor and performing hyperparameter tuning...")
    model = RandomForestRegressor(random_state=42)

    # Define a grid of hyperparameters to search
    param_grid = {
        'n_estimators': [50, 100, 200],  # Number of trees in the forest
        'max_depth': [5, 10, 15, None],  # Maximum depth of the tree (None means unlimited)
        'min_samples_split': [2, 5, 10]  # Minimum number of samples required to split an internal node
    }

    # Set up GridSearchCV for comprehensive hyperparameter search
    # cv=3 for 3-fold cross-validation
    # n_jobs=-1 to use all available CPU cores for parallel processing
    # verbose=2 for detailed output during training
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        verbose=2,
        scoring='r2' # Use R-squared as the scoring metric for regression
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best cross-validation R2 score: {grid_search.best_score_:.4f}")
    return best_model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model on the test set and prints performance metrics.
    """
    print("Evaluating the best model on the test set...")
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Test Set R-squared: {r2:.4f}")
    print(f"Test Set Mean Absolute Error (MAE): {mae:.4f}")

def save_artifacts(model, scaler, model_path, scaler_path):
    """
    Saves the trained model and scaler to disk using pickle.
    """
    print("Saving trained model and scaler...")
    try:
        # Ensure the directory for saving artifacts exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        with open(model_path, 'wb') as model_file:
            pickle.dump(model, model_file)
        print(f"Model saved to '{model_path}'")

        with open(scaler_path, 'wb') as scaler_file:
            pickle.dump(scaler, scaler_file)
        print(f"Scaler saved to '{scaler_path}'")

    except Exception as e:
        print(f"Error saving model/scaler: {e}")
        sys.exit(1)

def predict_liquidity(new_data_point, model_path, scaler_path, features):
    """
    Loads a saved model and scaler to make a prediction on a new data point.
    new_data_point: A dictionary with feature names as keys and values.
    """
    print("\nAttempting to load model and scaler for prediction...")
    try:
        with open(model_path, 'rb') as model_file:
            loaded_model = pickle.load(model_file)
        with open(scaler_path, 'rb') as scaler_file:
            loaded_scaler = pickle.load(scaler_file)
        print("Model and scaler loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model or scaler file not found at '{model_path}' or '{scaler_path}'.")
        print("Please ensure the training script was run successfully.")
        return None
    except Exception as e:
        print(f"Error loading model/scaler: {e}")
        return None

    # Convert new_data_point to DataFrame in the correct feature order
    new_df = pd.DataFrame([new_data_point], columns=features)

    # Scale the new data point using the loaded scaler
    scaled_new_data = loaded_scaler.transform(new_df)

    # Make prediction
    prediction = loaded_model.predict(scaled_new_data)
    print(f"Prediction for new data: {prediction[0]:.4f}")
    return prediction[0]


def main():
    """
    Main function to orchestrate the model training workflow.
    """
    print("Starting model training script...")

    # --- Configuration ---
    # Path to your processed data file (expected output from notebooks)
    DATA_PATH = '../../data/processed_data.csv'
    # Paths where the trained model and scaler will be saved
    MODEL_SAVE_PATH = 'tuned_liquidity_model.pkl'
    SCALER_SAVE_PATH = 'liquidity_scaler.pkl'
    # Features used for training the model
    FEATURES = ['24h_volume', 'mkt_cap', '1h_change', 'price']
    # The target variable to predict
    TARGET = 'liquidity'

    # --- 1. Load Data ---
    X, y = load_data(DATA_PATH, FEATURES, TARGET)

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

    # --- 4. Model Training and Hyperparameter Tuning ---
    best_model = train_model(X_train_scaled, y_train)

    # --- 5. Model Evaluation ---
    evaluate_model(best_model, X_test_scaled, y_test)

    # --- 6. Save Model and Scaler ---
    save_artifacts(best_model, scaler, MODEL_SAVE_PATH, SCALER_SAVE_PATH)

    print("Model training script finished.")

    # --- Optional: Demonstrate Prediction with a loaded model ---
    print("\n--- Demonstrating Prediction Functionality ---")
    dummy_new_data = {
        '24h_volume': 1.5e9,
        'mkt_cap': 8.0e10,
        '1h_change': -0.8,
        'price': 45000
    }
    predict_liquidity(dummy_new_data, MODEL_SAVE_PATH, SCALER_SAVE_PATH, FEATURES)

if __name__ == "__main__":
    main()
