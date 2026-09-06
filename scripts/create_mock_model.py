import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

def create_mock_model(model_path="../models/cooling_model.pkl"):
    """
    Creates a dummy regression model that mimics the risk_score behavior.
    Features: [cpu_norm, gpu_norm, mem_norm, disk_io_norm, network_io_norm, heat_norm, 
              cpu_roll_5, gpu_roll_5, heat_roll_5, cpu_roll_10, gpu_roll_10, heat_roll_10,
              cpu_delta, gpu_delta, temp_delta]
    Total 15 features.
    """
    print("Generating mock training data...")
    # Generate 1000 rows of random data
    X = np.random.rand(1000, 15)
    
    # Target: Simple linear combination + noise
    # risk_score = 0.6*cpu_norm + 0.4*gpu_norm
    y = 0.6 * X[:, 0] + 0.4 * X[:, 1] + 0.05 * np.random.randn(1000)
    y = np.clip(y, 0, 1)

    print(f"Training dummy model...")
    model = RandomForestRegressor(n_estimators=10)
    model.fit(X, y)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Mock model saved to {model_path}")

if __name__ == "__main__":
    create_mock_model()
