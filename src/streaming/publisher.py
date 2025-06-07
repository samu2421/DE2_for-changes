import json
import time
import sys
import os
from datetime import datetime, timedelta
from google.cloud import pubsub_v1

# Add parent directory to path to import fake_data_generator
sys.path.append('..')
sys.path.append('.')

# Your project ID
PROJECT_ID = "ecommerce-analytics-462115"
TOPIC_NAME = "orders-stream"

# Initialize publisher
publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path(PROJECT_ID, TOPIC_NAME)

def generate_fake_order():
    """Generate a realistic fake order for testing"""
    import random
    from faker import Faker
    
    fake = Faker()
    
    # More realistic product data
    products = [
        {'code': 'GIFT001', 'desc': 'Gift Card £10', 'price': 10.0},
        {'code': 'BOOK123', 'desc': 'Programming Book', 'price': 25.99},
        {'code': 'MUG456', 'desc': 'Coffee Mug', 'price': 8.50},
        {'code': 'TSHIRT789', 'desc': 'Cotton T-Shirt', 'price': 15.00},
        {'code': 'LAPTOP001', 'desc': 'Laptop Computer', 'price': 599.99},
        {'code': 'PHONE123', 'desc': 'Mobile Phone', 'price': 299.99}
    ]
    
    product = random.choice(products)
    quantity = random.randint(1, 5)
    
    order = {
        'InvoiceNo': fake.uuid4()[:8].upper(),
        'StockCode': product['code'],
        'Description': product['desc'],
        'Quantity': quantity,
        'InvoiceDate': (datetime.now() - timedelta(minutes=random.randint(0, 60))).isoformat(),
        'UnitPrice': product['price'],
        'CustomerID': random.randint(10000, 99999),
        'Country': random.choice(['United Kingdom', 'Germany', 'France', 'Netherlands', 'Spain'])
    }
    
    return order

def publish_fake_orders(num_orders=20, delay_seconds=3):
    """Send fake orders to Pub/Sub"""
    print(f"🚀 Starting to publish {num_orders} fake orders...")
    print(f"📡 Publishing to: {topic_path}")
    print(f"⏱️ Delay between orders: {delay_seconds} seconds")
    print("=" * 50)
    
    try:
        for i in range(num_orders):
            # Generate fake order
            order = generate_fake_order()
            revenue = order['Quantity'] * order['UnitPrice']
            
            # Convert to JSON and publish
            message = json.dumps(order, default=str).encode('utf-8')
            future = publisher.publish(topic_path, message)
            
            print(f"📦 Published order {i+1}/{num_orders}: {order['InvoiceNo']} - £{revenue:.2f}")
            
            # Wait before next order
            if i < num_orders - 1:  # Don't wait after the last order
                time.sleep(delay_seconds)
        
        print("✅ All orders published successfully!")
        
    except Exception as e:
        print(f"❌ Error publishing orders: {e}")
        print("💡 Make sure you've set up GCP authentication and Pub/Sub topic")

if __name__ == "__main__":
    # You can customize these parameters
    publish_fake_orders(num_orders=10, delay_seconds=2)