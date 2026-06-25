import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def prepare_data():
    # Paths
    raw_path = 'data/raw/crop_yield_dataset.csv'
    processed_dir = 'data/processed/'
    os.makedirs(processed_dir, exist_ok=True)
    
    # Load
    df = pd.read_csv(raw_path)
    
    # Features and Target split
    X = df.drop(columns=['yield_tons_per_hectare'])
    y = df['yield_tons_per_hectare']
    
    # Define columns
    num_cols = ['rainfall', 'temperature', 'irrigation']
    cat_cols = ['soil_type', 'fertilizer_use', 'season']
    
    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_cols)
        ])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Transform
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    # Get feature names for later use
    cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols)
    all_features = num_cols + list(cat_features)
    
    # Save processed components as CSVs
    pd.DataFrame(X_train_transformed, columns=all_features).to_csv(f"{processed_dir}X_train.csv", index=False)
    pd.DataFrame(X_test_transformed, columns=all_features).to_csv(f"{processed_dir}X_test.csv", index=False)
    y_train.to_csv(f"{processed_dir}y_train.csv", index=False)
    y_test.to_csv(f"{processed_dir}y_test.csv", index=False)
    
    import joblib
    joblib.dump(preprocessor, 'models/preprocessor.pkl')
    print("Data Preprocessing Complete. Preprocessor saved.")

if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    prepare_data()