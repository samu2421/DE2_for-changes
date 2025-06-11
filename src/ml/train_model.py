# src/ml/train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np
import logging
from google.cloud import bigquery
from pathlib import Path
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EcommerceMLTrainer:
    """
    BigQuery-powered ML model trainer for e-commerce revenue prediction
    """
    
    def __init__(self, project_id='ecommerce-analytics-462115', dataset_id='ecommerce_data', table_id='historical_orders'):
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.table_id = table_id
        self.table_ref = f"{project_id}.{dataset_id}.{table_id}"
        self.client = None
        self.model = None
        self.feature_names = []
        self.label_encoders = {}
        
    def initialize_bigquery_client(self):
        """Initialize BigQuery client"""
        try:
            self.client = bigquery.Client(project=self.project_id)
            logger.info(f"BigQuery client initialized for project: {self.project_id}")
            return True
        except Exception as e:
            logger.error(f"Error initializing BigQuery client: {e}")
            return False
    
    def load_training_data_from_bigquery(self, sample_size=50000):
        """Load and prepare training data from BigQuery"""
        if not self.client:
            logger.error("BigQuery client not initialized")
            return None, None
            
        logger.info(f"Loading training data from BigQuery (sample size: {sample_size:,})")
        
        try:
            # Enhanced query with more features for better ML performance
            query = f"""
            SELECT 
                InvoiceNo,
                StockCode,
                Description,
                Quantity,
                InvoiceDate,
                UnitPrice,
                CustomerID,
                Country,
                (Quantity * UnitPrice) as Revenue
            FROM `{self.table_ref}`
            WHERE 
                Quantity > 0 
                AND UnitPrice > 0
                AND Quantity <= 100  -- Remove extreme outliers
                AND UnitPrice <= 100  -- Remove extreme outliers
                AND CustomerID IS NOT NULL
                AND InvoiceDate IS NOT NULL
                AND Country IS NOT NULL
            ORDER BY RAND()  -- Random sampling
            LIMIT {sample_size}
            """
            
            logger.info("Executing BigQuery query for training data...")
            df = self.client.query(query).to_dataframe()
            
            if df.empty:
                logger.error("No training data returned from BigQuery")
                return None, None
                
            logger.info(f"Loaded {len(df):,} training samples from BigQuery")
            
            # Feature engineering
            df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
            df['Hour'] = df['InvoiceDate'].dt.hour
            df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek
            df['Month'] = df['InvoiceDate'].dt.month
            df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
            df['IsBusinessHours'] = ((df['Hour'] >= 9) & (df['Hour'] <= 17)).astype(int)
            
            # Create price and quantity categories for better learning
            df['PriceCategory'] = pd.cut(df['UnitPrice'], 
                                       bins=[0, 2, 5, 15, float('inf')], 
                                       labels=['Low', 'Medium', 'High', 'Premium'])
            
            df['QuantityCategory'] = pd.cut(df['Quantity'], 
                                          bins=[0, 1, 3, 10, float('inf')],
                                          labels=['Single', 'Few', 'Multiple', 'Bulk'])
            
            # Prepare features and target
            X, y = self.prepare_features_and_target(df)
            
            logger.info(f"Feature engineering complete. Features: {X.shape[1]}, Samples: {len(y):,}")
            logger.info(f"Target range: £{y.min():.2f} - £{y.max():.2f}")
            logger.info(f"Average revenue: £{y.mean():.2f}")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error loading training data from BigQuery: {e}")
            return None, None
    
    def prepare_features_and_target(self, df):
        """Prepare features and target variable"""
        
        # Numerical features
        numerical_features = ['Quantity', 'UnitPrice', 'Hour', 'DayOfWeek', 'Month', 
                             'IsWeekend', 'IsBusinessHours']
        
        # Categorical features
        categorical_features = ['Country', 'PriceCategory', 'QuantityCategory']
        
        # Start with numerical features
        X = df[numerical_features].copy()
        
        # Encode categorical features
        for cat_feature in categorical_features:
            if cat_feature in df.columns:
                # Handle missing values
                cat_data = df[cat_feature].astype(str).fillna('Unknown')
                
                # Use label encoding
                le = LabelEncoder()
                X[f'{cat_feature}_encoded'] = le.fit_transform(cat_data)
                
                # Store encoder for later use
                self.label_encoders[cat_feature] = le
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Target variable
        y = df['Revenue']
        
        return X, y
    
    def train_revenue_predictor(self, test_size=0.2, random_state=42):
        """Train the revenue prediction model"""
        
        logger.info("Starting ML model training...")
        
        # Load data
        X, y = self.load_training_data_from_bigquery()
        if X is None or y is None:
            logger.error("Failed to load training data")
            return None
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        logger.info(f"Training set: {X_train.shape[0]:,} samples")
        logger.info(f"Test set: {X_test.shape[0]:,} samples")
        
        # Train model
        logger.info("Training Random Forest model...")
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        logger.info("Evaluating model performance...")
        
        # Training predictions
        train_predictions = self.model.predict(X_train)
        train_mse = mean_squared_error(y_train, train_predictions)
        train_r2 = r2_score(y_train, train_predictions)
        train_mae = mean_absolute_error(y_train, train_predictions)
        
        # Test predictions
        test_predictions = self.model.predict(X_test)
        test_mse = mean_squared_error(y_test, test_predictions)
        test_r2 = r2_score(y_test, test_predictions)
        test_mae = mean_absolute_error(y_test, test_predictions)
        
        # Model performance summary
        performance = {
            'training': {
                'mse': float(train_mse),
                'r2_score': float(train_r2),
                'mae': float(train_mae),
                'rmse': float(np.sqrt(train_mse))
            },
            'testing': {
                'mse': float(test_mse),
                'r2_score': float(test_r2),
                'mae': float(test_mae),
                'rmse': float(np.sqrt(test_mse))
            },
            'feature_count': len(self.feature_names),
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        # Log results
        logger.info(f"Training Results:")
        logger.info(f"  R² Score: {train_r2:.4f}")
        logger.info(f"  RMSE: £{np.sqrt(train_mse):.2f}")
        logger.info(f"  MAE: £{train_mae:.2f}")
        
        logger.info(f"Test Results:")
        logger.info(f"  R² Score: {test_r2:.4f}")
        logger.info(f"  RMSE: £{np.sqrt(test_mse):.2f}")
        logger.info(f"  MAE: £{test_mae:.2f}")
        
        # Feature importance
        feature_importance = list(zip(self.feature_names, self.model.feature_importances_))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        logger.info("Top 5 Most Important Features:")
        for i, (feature, importance) in enumerate(feature_importance[:5], 1):
            logger.info(f"  {i}. {feature}: {importance:.4f}")
        
        return performance
    
    def save_model_and_artifacts(self):
        """Save the trained model and preprocessing artifacts"""
        
        if self.model is None:
            logger.error("No model to save. Train a model first.")
            return False
        
        try:
            # Create ml directory if it doesn't exist
            Path('src/ml').mkdir(exist_ok=True)
            
            # Save the trained model
            joblib.dump(self.model, 'src/ml/revenue_model.pkl')
            logger.info("Model saved to src/ml/revenue_model.pkl")
            
            # Save label encoders
            with open('src/ml/label_encoders.pkl', 'wb') as f:
                pickle.dump(self.label_encoders, f)
            logger.info("Label encoders saved to src/ml/label_encoders.pkl")
            
            # Save feature names
            with open('src/ml/feature_names.txt', 'w') as f:
                f.write('\n'.join(self.feature_names))
            logger.info("Feature names saved to src/ml/feature_names.txt")
            
            # Save model metadata
            metadata = {
                'feature_names': self.feature_names,
                'feature_count': len(self.feature_names),
                'model_type': 'RandomForestRegressor',
                'training_data_source': 'BigQuery',
                'table_reference': self.table_ref
            }
            
            with open('src/ml/model_metadata.json', 'w') as f:
                import json
                json.dump(metadata, f, indent=2)
            logger.info("Model metadata saved to src/ml/model_metadata.json")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving model artifacts: {e}")
            return False
    
    def test_model_prediction(self):
        """Test the trained model with sample predictions"""
        
        if self.model is None:
            logger.error("No model loaded. Train a model first.")
            return
        
        logger.info("Testing model with sample predictions...")
        
        # Create test samples
        test_samples = [
            {
                'Quantity': 3,
                'UnitPrice': 15.50,
                'Hour': 14,
                'DayOfWeek': 2,
                'Month': 6,
                'IsWeekend': 0,
                'IsBusinessHours': 1,
                'Country': 'United Kingdom',
                'PriceCategory': 'Medium',
                'QuantityCategory': 'Few'
            },
            {
                'Quantity': 1,
                'UnitPrice': 25.99,
                'Hour': 10,
                'DayOfWeek': 1,
                'Month': 12,
                'IsWeekend': 0,
                'IsBusinessHours': 1,
                'Country': 'Germany',
                'PriceCategory': 'High',
                'QuantityCategory': 'Single'
            },
            {
                'Quantity': 10,
                'UnitPrice': 5.00,
                'Hour': 20,
                'DayOfWeek': 5,
                'Month': 3,
                'IsWeekend': 1,
                'IsBusinessHours': 0,
                'Country': 'France',
                'PriceCategory': 'Medium',
                'QuantityCategory': 'Multiple'
            }
        ]
        
        for i, sample in enumerate(test_samples, 1):
            try:
                # Prepare features
                features = []
                
                # Numerical features
                numerical_features = ['Quantity', 'UnitPrice', 'Hour', 'DayOfWeek', 'Month', 
                                     'IsWeekend', 'IsBusinessHours']
                for feature in numerical_features:
                    features.append(sample[feature])
                
                # Categorical features
                categorical_features = ['Country', 'PriceCategory', 'QuantityCategory']
                for cat_feature in categorical_features:
                    if cat_feature in self.label_encoders:
                        try:
                            encoded_value = self.label_encoders[cat_feature].transform([sample[cat_feature]])[0]
                        except ValueError:
                            # Handle unseen categories
                            encoded_value = 0
                        features.append(encoded_value)
                
                # Make prediction
                prediction = self.model.predict([features])[0]
                actual_revenue = sample['Quantity'] * sample['UnitPrice']
                
                logger.info(f"Test Sample {i}:")
                logger.info(f"  Input: {sample['Quantity']} x £{sample['UnitPrice']} from {sample['Country']}")
                logger.info(f"  Predicted Revenue: £{prediction:.2f}")
                logger.info(f"  Actual Revenue: £{actual_revenue:.2f}")
                logger.info(f"  Difference: £{abs(prediction - actual_revenue):.2f}")
                
            except Exception as e:
                logger.error(f"Error testing sample {i}: {e}")


def main():
    """Main training function"""
    print("Starting E-Commerce ML Model Training with BigQuery...")
    
    # Initialize trainer
    trainer = EcommerceMLTrainer()
    
    # Initialize BigQuery client
    if not trainer.initialize_bigquery_client():
        print("Failed to initialize BigQuery client. Exiting.")
        return
    
    # Train model
    performance = trainer.train_revenue_predictor()
    if performance is None:
        print("Model training failed. Exiting.")
        return
    
    # Save model and artifacts
    if trainer.save_model_and_artifacts():
        print("Model and artifacts saved successfully!")
    else:
        print("Failed to save model artifacts.")
    
    # Test model
    trainer.test_model_prediction()
    
    print("\nML model training completed successfully!")
    print("Model ready for use in prediction API.")


if __name__ == "__main__":
    main()