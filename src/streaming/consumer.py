import json
from google.cloud import pubsub_v1

subscriber = pubsub_v1.SubscriberClient()
subscription_path = subscriber.subscription_path('your-project-id', 'orders-consumer')

def callback(message):
    """Process each streaming order"""
    try:
        order = json.loads(message.data.decode('utf-8'))
        print(f"Received order: {order['InvoiceNo']} - ${order['Quantity']} x {order['UnitPrice']}")
        
        # Simple real-time analytics
        revenue = float(order['Quantity']) * float(order['UnitPrice'])
        if revenue > 100:
            print(f"🚨 HIGH VALUE ORDER: ${revenue:.2f}")
        
        message.ack()  # Acknowledge the message
    except Exception as e:
        print(f"Error processing message: {e}")
        message.ack()

def start_consuming():
    """Start listening for streaming orders"""
    print("Starting to consume orders...")
    streaming_pull_future = subscriber.subscribe(subscription_path, callback=callback)
    print(f"Listening for messages on {subscription_path}...")
    
    try:
        streaming_pull_future.result()  # Keep listening
    except KeyboardInterrupt:
        streaming_pull_future.cancel()

if __name__ == "__main__":
    start_consuming()