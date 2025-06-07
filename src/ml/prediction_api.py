from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('src/ml/revenue_model.pkl')

@app.route('/predict', methods=['POST'])
def predict_revenue():
    """Predict revenue for a new order"""
    try:
        data = request.json
        
        # Extract features
        features = [
            data['quantity'],
            data['unit_price'], 
            data['hour'],
            data['day_of_week']
        ]
        
        # Make prediction
        prediction = model.predict([features])[0]
        
        return jsonify({
            'predicted_revenue': round(prediction, 2),
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == "__main__":
    print("Starting ML Prediction API...")
    app.run(host='0.0.0.0', port=5001, debug=True)
