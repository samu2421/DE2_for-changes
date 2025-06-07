# src/ml/simple_data_prep.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def simple_ml_data_prep():
    """Simplified ML data preparation focusing on core features"""
    
    print("Starting Simple ML Data Preparation...")
    
    # Load and clean data
    try:
        df = pd.read_csv('data/OnlineRetail.csv', encoding='latin1')
        logger.info(f"Loaded {len(df):,} rows")
        
        # Basic cleaning
        df = df.dropna(subset=['CustomerID', 'InvoiceDate', 'Quantity', 'UnitPrice'])
        df = df[df['Quantity'] > 0]
        df = df[df['UnitPrice'] > 0]
        
        # Remove extreme outliers
        df = df[df['Quantity'] <= df['Quantity'].quantile(0.99)]
        df = df[df['UnitPrice'] <= df['UnitPrice'].quantile(0.99)]
        
        logger.info(f"After cleaning: {len(df):,} rows")
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return
    
    # Create target variable
    df['Revenue'] = df['Quantity'] * df['UnitPrice']
    
    # Create basic features
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['Hour'] = df['InvoiceDate'].dt.hour
    df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek
    df['Month'] = df['InvoiceDate'].dt.month
    df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
    df['IsBusinessHours'] = ((df['Hour'] >= 9) & (df['Hour'] <= 17)).astype(int)
    
    # Create simple categorical features
    df['PriceRange'] = pd.cut(df['UnitPrice'], bins=[0, 5, 15, 50, float('inf')], 
                             labels=['Low', 'Medium', 'High', 'Premium'])
    df['QuantityRange'] = pd.cut(df['Quantity'], bins=[0, 2, 5, 20, float('inf')],
                                labels=['Single', 'Few', 'Multiple', 'Bulk'])
    
    # Select features for ML
    numerical_features = ['Quantity', 'UnitPrice', 'Hour', 'DayOfWeek', 'Month', 
                         'IsWeekend', 'IsBusinessHours']
    
    categorical_features = ['Country', 'PriceRange', 'QuantityRange']
    
    # Create feature matrix
    X = df[numerical_features].copy()
    
    # Encode categorical features safely
    encoders = {}
    for cat_col in categorical_features:
        if cat_col in df.columns:
            # Convert to string and handle missing values
            cat_data = df[cat_col].astype(str).fillna('Unknown')
            
            # Encode
            le = LabelEncoder()
            X[f'{cat_col}_encoded'] = le.fit_transform(cat_data)
            encoders[cat_col] = le
    
    # Target variable
    y = df['Revenue']
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrames for consistency
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)
    
    # Save everything
    Path('data/ml_ready').mkdir(exist_ok=True)
    Path('src/ml').mkdir(exist_ok=True)
    
    # Save datasets
    X_train_scaled.to_csv('data/ml_ready/X_train.csv', index=False)
    X_test_scaled.to_csv('data/ml_ready/X_test.csv', index=False)
    y_train.to_csv('data/ml_ready/y_train.csv', index=False)
    y_test.to_csv('data/ml_ready/y_test.csv', index=False)
    
    # Save preprocessing objects
    with open('src/ml/label_encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)
    
    with open('src/ml/feature_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    with open('src/ml/feature_names.txt', 'w') as f:
        f.write('\n'.join(X.columns))
    
    # Print summary
    print(f"\nML Data Preparation Complete!")
    print(f"Features: {X.shape[1]} features, {len(y):,} samples")
    print(f"Train set: {X_train_scaled.shape[0]:,} samples")
    print(f"Test set: {X_test_scaled.shape[0]:,} samples")
    print(f"Target range: £{y.min():.2f} - £{y.max():.2f}")
    print(f"Average revenue: £{y.mean():.2f}")
    
    print(f"\nFeatures included:")
    for i, feature in enumerate(X.columns, 1):
        print(f"  {i:2d}. {feature}")
    
    print(f"\nFiles saved:")
    print("  • data/ml_ready/X_train.csv")
    print("  • data/ml_ready/X_test.csv") 
    print("  • data/ml_ready/y_train.csv")
    print("  • data/ml_ready/y_test.csv")
    print("  • src/ml/label_encoders.pkl")
    print("  • src/ml/feature_scaler.pkl")
    print("  • src/ml/feature_names.txt")

if __name__ == "__main__":
    simple_ml_data_prep()