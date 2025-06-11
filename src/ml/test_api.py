# src/ml/test_api.py
import requests
import json

def test_prediction_api():
    """Test the ML prediction API with proper feature format"""
    
    base_url = 'http://localhost:5001'
    
    print("🧪 Testing ML Prediction API")
    print("=" * 40)
    
    # Test 1: Health check
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get(f'{base_url}/health')
        print(f"Health check: {response.json()}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return
    
    # Test 2: Full prediction with all features
    print("\n2. Testing full prediction endpoint...")
    test_order_full = {
        'quantity': 3,
        'unit_price': 15.50,
        'hour': 14,
        'day_of_week': 2,
        'month': 6,
        'country': 'United Kingdom'
    }
    
    try:
        response = requests.post(f'{base_url}/predict', json=test_order_full)
        result = response.json()
        print(f"Full prediction result:")
        print(f"  Input: {test_order_full['quantity']} x £{test_order_full['unit_price']}")
        print(f"  Predicted Revenue: £{result.get('predicted_revenue', 'N/A')}")
        print(f"  Actual Revenue: £{test_order_full['quantity'] * test_order_full['unit_price']}")
        print(f"  Status: {result.get('status', 'unknown')}")
        
        if result.get('status') == 'error':
            print(f"  Error: {result.get('error')}")
            
    except Exception as e:
        print(f"❌ Full prediction failed: {e}")
    
    # Test 3: Simple prediction endpoint
    print("\n3. Testing simple prediction endpoint...")
    test_order_simple = {
        'quantity': 2,
        'unit_price': 25.99
    }
    
    try:
        response = requests.post(f'{base_url}/predict/simple', json=test_order_simple)
        result = response.json()
        print(f"Simple prediction result:")
        print(f"  Input: {test_order_simple['quantity']} x £{test_order_simple['unit_price']}")
        print(f"  Predicted Revenue: £{result.get('predicted_revenue', 'N/A')}")
        print(f"  Actual Revenue: £{result.get('actual_revenue', 'N/A')}")
        print(f"  Status: {result.get('status', 'unknown')}")
        
        if result.get('status') == 'error':
            print(f"  Error: {result.get('error')}")
            
    except Exception as e:
        print(f"❌ Simple prediction failed: {e}")
    
    # Test 4: Multiple test cases
    print("\n4. Testing multiple scenarios...")
    test_cases = [
        {'quantity': 1, 'unit_price': 5.99, 'country': 'Germany', 'hour': 10},
        {'quantity': 5, 'unit_price': 12.50, 'country': 'France', 'hour': 16},
        {'quantity': 10, 'unit_price': 3.25, 'country': 'Netherlands', 'hour': 20}
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        try:
            response = requests.post(f'{base_url}/predict', json=test_case)
            result = response.json()
            
            if result.get('status') == 'success':
                predicted = result['predicted_revenue']
                actual = test_case['quantity'] * test_case['unit_price']
                difference = abs(predicted - actual)
                accuracy = (1 - difference / actual) * 100 if actual > 0 else 0
                
                print(f"  Test {i}: {test_case['quantity']} x £{test_case['unit_price']} from {test_case.get('country', 'UK')}")
                print(f"    Predicted: £{predicted:.2f} | Actual: £{actual:.2f} | Accuracy: {accuracy:.1f}%")
            else:
                print(f"  Test {i}: Error - {result.get('error')}")
                
        except Exception as e:
            print(f"  Test {i}: Exception - {e}")

def test_streaming_integration():
    """Test how the API would work with streaming data"""
    
    print("\n🌊 Testing Streaming Integration Simulation")
    print("=" * 50)
    
    # Simulate streaming orders
    streaming_orders = [
        {'quantity': 2, 'unit_price': 19.99, 'country': 'United Kingdom'},
        {'quantity': 1, 'unit_price': 45.00, 'country': 'Germany'},
        {'quantity': 3, 'unit_price': 8.75, 'country': 'France'},
        {'quantity': 7, 'unit_price': 6.50, 'country': 'Netherlands'},
        {'quantity': 1, 'unit_price': 125.00, 'country': 'Spain'}
    ]
    
    total_predicted = 0
    total_actual = 0
    high_value_alerts = 0
    
    for i, order in enumerate(streaming_orders, 1):
        try:
            response = requests.post('http://localhost:5001/predict', json=order)
            result = response.json()
            
            if result.get('status') == 'success':
                predicted = result['predicted_revenue']
                actual = order['quantity'] * order['unit_price']
                
                total_predicted += predicted
                total_actual += actual
                
                print(f"📦 Order {i}: {order['quantity']} x £{order['unit_price']} from {order['country']}")
                print(f"   💰 Predicted: £{predicted:.2f} | Actual: £{actual:.2f}")
                
                # Simulate high-value order alert
                if predicted > 100:
                    print(f"   🚨 HIGH VALUE ORDER ALERT: £{predicted:.2f}")
                    high_value_alerts += 1
            else:
                print(f"❌ Order {i} prediction failed: {result.get('error')}")
                
        except Exception as e:
            print(f"❌ Order {i} failed: {e}")
    
    print(f"\n📊 Streaming Session Summary:")
    print(f"   Total Predicted Revenue: £{total_predicted:.2f}")
    print(f"   Total Actual Revenue: £{total_actual:.2f}")
    print(f"   Prediction Accuracy: {(1 - abs(total_predicted - total_actual) / total_actual) * 100:.1f}%")
    print(f"   High-Value Alerts: {high_value_alerts}")

if __name__ == "__main__":
    print("Testing ML API...")
    print("Make sure the API is running: python src/ml/prediction_api.py")
    
    test_prediction_api()
    test_streaming_integration()
    
    print("\n✅ API testing complete!")