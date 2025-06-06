import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np

def prepare_data():
    """Load and prepare data for ML"""
    # Load real Kaggle data
    df = pd.read_csv('data/OnlineRetail.csv', encoding='latin1')
    
    # Simple feature engineering
    df = df.dropna()  # Remove missing values
    df['Revenue'] = df['Quantity'] * df['UnitPrice']
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['Hour'] = df['InvoiceDate'].dt.hour
    df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek
    
    # Create features and target
    features = ['Quantity', 'UnitPrice', 'Hour', 'DayOfWeek']
    target = 'Revenue'
    
    X = df[features]
    y = df[target]
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_revenue_predictor():
    """Train a simple model to predict order revenue"""
    print("Preparing data...")
    X_train, X_test, y_train, y_test = prepare_data()
    
    print("Training model...")
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Model trained! MSE: {mse:.2f}")
    
    # Save model
    joblib.dump(model, 'src/ml/revenue_model.pkl')
    print("Model saved to src/ml/revenue_model.pkl")
    
    return model

if __name__ == "__main__":
    model = train_revenue_predictor()