import json
import time
from google.cloud import pubsub_v1
import sys
sys.path.append('..')
from fake_data_generator import generate_fake_order

# Initialize publisher
publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path('your-project-id', 'orders-stream')

def publish_fake_orders():
    """Send fake orders to Pub/Sub every few seconds"""
    print("Starting to publish fake orders...")
    
    for i in range(20):  # Publish 20 orders for testing
        # Generate fake order
        order = generate_fake_order()
        
        # Convert to JSON and publish
        message = json.dumps(order, default=str).encode('utf-8')
        future = publisher.publish(topic_path, message)
        
        print(f"Published order {i+1}: {order['InvoiceNo']}")
        time.sleep(3)  # Wait 3 seconds between orders

if __name__ == "__main__":
    publish_fake_orders()

