# test_consumer_with_littles_law.py
"""
Simple test of consumer functionality with Little's Law
Simulates order processing without needing GCP setup
"""

import json
import requests
import time
import sys
import os

# Add current directory to path for imports
sys.path.append('.')
sys.path.append(os.getcwd())

# Import Little's Law tracker
from simple_littles_law import order_arrives, order_completes, show_metrics

def simulate_order_processing():
    """Simulate order processing with Little's Law tracking"""
    
    print("🧪 Testing Consumer with Little's Law Integration")
    print("=" * 60)
    print("📊 Little's Law tracking is now active!")
    print("💡 Simulating orders (no GCP needed for this test)")
    print("=" * 60)
    
    # Simulate realistic order data (like what comes from Pub/Sub)
    sample_orders = [
        {
            'InvoiceNo': 'TEST_001',
            'Quantity': 3,
            'UnitPrice': 15.99,
            'Country': 'United Kingdom'
        },
        {
            'InvoiceNo': 'TEST_002', 
            'Quantity': 1,
            'UnitPrice': 45.50,
            'Country': 'Germany'
        },
        {
            'InvoiceNo': 'TEST_003',
            'Quantity': 2,
            'UnitPrice': 22.75,
            'Country': 'France'
        },
        {
            'InvoiceNo': 'TEST_004',
            'Quantity': 5,
            'UnitPrice': 8.99,
            'Country': 'Netherlands'
        },
        {
            'InvoiceNo': 'TEST_005',
            'Quantity': 1,
            'UnitPrice': 125.00,
            'Country': 'Spain'
        }
    ]
    
    processed_orders = 0
    
    for order in sample_orders:
        print(f"\n📦 Processing order {order['InvoiceNo']}...")
        
        # Simulate the exact flow from your consumer
        order_id = order['InvoiceNo']
        
        print(f"📦 Received order: {order_id} - {order['Quantity']} x £{order['UnitPrice']}")
        
        # LITTLE'S LAW: Track order arrival
        order_arrives(order_id)
        
        # Calculate actual revenue (your existing logic)
        actual_revenue = float(order['Quantity']) * float(order['UnitPrice'])
        
        # Simulate ML prediction call
        prediction_data = {
            'quantity': order['Quantity'],
            'unit_price': order['UnitPrice'], 
            'hour': 14,
            'day_of_week': 2
        }
        
        print(f"🤖 Calling ML API...")
        
        try:
            # Try to call your actual ML API
            response = requests.post('http://localhost:5001/predict', 
                                   json=prediction_data, timeout=5)
            if response.status_code == 200:
                predicted_revenue = response.json()['predicted_revenue']
                print(f"🤖 ML Prediction: £{predicted_revenue:.2f} | Actual: £{actual_revenue:.2f}")
                
                # Alert for high-value orders
                if actual_revenue > 100:
                    print(f"🚨 HIGH VALUE ORDER: £{actual_revenue:.2f}")
            else:
                print(f"⚠️ ML API Error: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"⚠️ ML API unavailable: {e}")
            print(f"💰 Order Revenue: £{actual_revenue:.2f}")
        
        # Simulate processing time
        processing_time = 0.5 + (processed_orders * 0.2)  # Vary processing time
        time.sleep(processing_time)
        
        # LITTLE'S LAW: Track order completion
        order_completes(order_id)
        
        processed_orders += 1
        
        # Show metrics every 2 orders
        if processed_orders % 2 == 0:
            print(f"\n📊 Little's Law Status (after {processed_orders} orders):")
            show_metrics()
            print("\n" + "="*50)
    
    print(f"\n🎉 Test Complete! Processed {processed_orders} orders")
    print(f"\n📊 Final Little's Law Report:")
    show_metrics()

def test_just_littles_law():
    """Quick test of just Little's Law functionality"""
    print("\n⚡ Quick Little's Law Test")
    print("=" * 30)
    
    # Quick test orders
    test_orders = ['QUICK_A', 'QUICK_B', 'QUICK_C']
    
    # Orders arrive
    for order_id in test_orders:
        order_arrives(order_id)
        time.sleep(0.1)
    
    # Orders complete
    for order_id in test_orders:
        time.sleep(0.3)
        order_completes(order_id)
    
    show_metrics()

if __name__ == "__main__":
    print("🎯 CONSUMER + LITTLE'S LAW INTEGRATION TEST")
    print("=" * 50)
    print("Choose test mode:")
    print("1. Full consumer simulation (with ML API calls)")
    print("2. Quick Little's Law test only")
    
    try:
        choice = input("\nEnter choice (1 or 2): ").strip()
        
        if choice == "2":
            test_just_littles_law()
        else:
            simulate_order_processing()
            
    except KeyboardInterrupt:
        print("\n🛑 Test cancelled")
        print("\n📊 Final Little's Law Status:")
        show_metrics()
    except Exception as e:
        print(f"\n❌ Error: {e}")