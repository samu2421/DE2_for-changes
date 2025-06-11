# src/ml/prediction_api.py
from flask import Flask, request, jsonify
import joblib
import numpy as np
import pickle
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
app = Flask(__name__)

# Load the trained model and preprocessors
try:
    model = joblib.load('src/ml/revenue_model.pkl')
    
    # Load label encoders
    with open('src/ml/label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    
    print("✅ Model and encoders loaded successfully")
    print(f"✅ Available encoders: {list(label_encoders.keys())}")
    
except Exception as e:
    print(f"❌ Error loading model or encoders: {e}")
    model = None
    label_encoders = {}

@app.route('/predict', methods=['POST'])
def predict_revenue():
    """Predict revenue for a new order with proper feature engineering"""
    try:
        data = request.json
        
        # Extract basic features
        quantity = data.get('quantity', 1)
        unit_price = data.get('unit_price', 10.0)
        hour = data.get('hour', 14)
        day_of_week = data.get('day_of_week', 2)
        month = data.get('month', 6)
        
        # Extract categorical features with defaults
        country = data.get('country', 'United Kingdom')
        
        # Auto-determine price category based on unit_price
        if unit_price <= 2:
            price_category = 'Low'
        elif unit_price <= 5:
            price_category = 'Medium'  
        elif unit_price <= 15:
            price_category = 'High'
        else:
            price_category = 'Premium'
            
        # Auto-determine quantity category
        if quantity <= 1:
            quantity_category = 'Single'
        elif quantity <= 3:
            quantity_category = 'Few'
        elif quantity <= 10:
            quantity_category = 'Multiple'
        else:
            quantity_category = 'Bulk'
        
        # Calculate derived features
        is_weekend = 1 if day_of_week >= 5 else 0
        is_business_hours = 1 if 9 <= hour <= 17 else 0
        
        # Build feature vector (must match training order)
        features = [quantity, unit_price, hour, day_of_week, month, is_weekend, is_business_hours]
        
        # Add encoded categorical features
        categorical_features = ['Country', 'PriceCategory', 'QuantityCategory'] 
        categorical_values = [country, price_category, quantity_category]
        
        for cat_feature, cat_value in zip(categorical_features, categorical_values):
            if cat_feature in label_encoders:
                try:
                    encoded_value = label_encoders[cat_feature].transform([cat_value])[0]
                except ValueError:
                    # Handle unseen categories with default value
                    encoded_value = 0
                features.append(encoded_value)
            else:
                features.append(0)  # Default if encoder missing
        
        # Validate feature count
        if len(features) != 10:
            return jsonify({
                'error': f'Expected 10 features, got {len(features)}',
                'features_received': features,
                'status': 'error'
            })
        
        # Make prediction
        if model is None:
            return jsonify({
                'error': 'Model not loaded',
                'status': 'error'
            })
            
        prediction = model.predict([features])[0]
        
        return jsonify({
            'predicted_revenue': round(prediction, 2),
            'input_features': {
                'quantity': quantity,
                'unit_price': unit_price,
                'hour': hour,
                'day_of_week': day_of_week,
                'month': month,
                'country': country,
                'price_category': price_category,
                'quantity_category': quantity_category,
                'is_weekend': is_weekend,
                'is_business_hours': is_business_hours
            },
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'encoders_available': list(label_encoders.keys())
    })

@app.route('/predict/simple', methods=['POST'])  
def predict_simple():
    """Simple prediction endpoint with minimal inputs"""
    try:
        data = request.json
        
        # Only require quantity and unit_price
        quantity = data['quantity']
        unit_price = data['unit_price']
        
        # Use defaults for other features
        prediction_data = {
            'quantity': quantity,
            'unit_price': unit_price,
            'hour': 14,
            'day_of_week': 2,
            'month': 6,
            'country': 'United Kingdom'
        }
        
        # Build features manually
        is_weekend = 0  # Tuesday
        is_business_hours = 1  # 2 PM
        
        features = [quantity, unit_price, 14, 2, 6, is_weekend, is_business_hours]
        
        # Add categorical features
        for cat_feature, cat_value in [('Country', 'United Kingdom'), ('PriceCategory', 'Medium'), ('QuantityCategory', 'Few')]:
            if cat_feature in label_encoders:
                try:
                    encoded_value = label_encoders[cat_feature].transform([cat_value])[0]
                except ValueError:
                    encoded_value = 0
                features.append(encoded_value)
            else:
                features.append(0)
        
        # Make prediction
        if model is None:
            return jsonify({'error': 'Model not loaded', 'status': 'error'})
            
        prediction = model.predict([features])[0]
        
        return jsonify({
            'predicted_revenue': round(prediction, 2),
            'actual_revenue': round(quantity * unit_price, 2),
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        })

if __name__ == "__main__":
    print("Starting ML Prediction API...")
    print("Available endpoints:")
    print("  POST /predict - Full prediction with all features")
    print("  POST /predict/simple - Simple prediction with quantity + unit_price")
    print("  GET /health - Health check")
    app.run(host='0.0.0.0', port=5001, debug=True)