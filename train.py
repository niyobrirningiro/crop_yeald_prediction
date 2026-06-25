import os
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def train_and_evaluate():
    # Load processed data
    X_train = pd.read_csv('data/processed/X_train.csv')
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
    y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
    
    # Define models
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42)
    }
    
    results = {}
    best_r2 = -float('inf')
    best_model = None
    best_name = ""
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        results[name] = {"MAE": round(mae, 4), "R2": round(r2, 4)}
        print(f"{name} -> MAE: {mae:.4f}, R²: {r2:.4f}")
        
        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_name = name
            
    # Save the absolute best model
    joblib.dump(best_model, 'models/best_crop_model.pkl')
    # Save results summary for dashboard metrics
    joblib.dump(results, 'models/evaluation_results.pkl')
    print(f"\nSaved Best Model: {best_name} (R²: {best_r2:.4f})")

if __name__ == "__main__":
    train_and_evaluate()