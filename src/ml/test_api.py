import requests
import json

def test_prediction_api():
    """Test the ML prediction API"""
    
    # Test data
    test_order = {
        'quantity': 3,
        'unit_price': 15.50,
        'hour': 14,
        'day_of_week': 2
    }
    
    try:
        # Make prediction request
        response = requests.post(
            'http://localhost:5000/predict',
            json=test_order
        )
        
        print("API Response:", response.json())
        
    except Exception as e:
        print(f"Error testing API: {e}")

if __name__ == "__main__":
    print("Testing ML API...")
    test_prediction_api()
