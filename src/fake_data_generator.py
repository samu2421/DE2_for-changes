# src/fake_data_generator.py
import pandas as pd
from faker import Faker
import random
from datetime import datetime, timedelta
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EcommerceFakeDataGenerator:
    """
    Enhanced fake data generator for e-commerce analytics
    Compatible with streaming publisher and realistic data patterns
    """
    
    def __init__(self, real_data_path='data/OnlineRetail.csv'):
        self.fake = Faker()
        self.real_products = []
        self.real_countries = []
        self.price_patterns = {}
        self.load_real_patterns(real_data_path)
    
    def load_real_patterns(self, data_path):
        """Load patterns from real Kaggle dataset"""
        try:
            logger.info("Loading real data patterns...")
            real_data = pd.read_csv(data_path, encoding='latin1')
            
            # Extract valid products and countries
            self.real_products = real_data['StockCode'].dropna().unique()[:200]  # Use more products
            self.real_countries = real_data['Country'].dropna().unique()
            
            # Analyze price patterns for realistic generation
            clean_data = real_data.dropna(subset=['StockCode', 'UnitPrice'])
            clean_data = clean_data[clean_data['UnitPrice'] > 0]
            
            # Create price patterns by product category
            for product in self.real_products:
                product_data = clean_data[clean_data['StockCode'] == product]
                if not product_data.empty:
                    self.price_patterns[product] = {
                        'min_price': product_data['UnitPrice'].min(),
                        'max_price': product_data['UnitPrice'].max(),
                        'avg_price': product_data['UnitPrice'].mean(),
                        'description': product_data['Description'].iloc[0] if 'Description' in product_data.columns else f"Product {product}"
                    }
            
            logger.info(f"Loaded patterns for {len(self.real_products)} products and {len(self.real_countries)} countries")
            
        except Exception as e:
            logger.warning(f"Could not load real data patterns: {e}")
            # Fallback to default patterns
            self.setup_fallback_patterns()
    
    def setup_fallback_patterns(self):
        """Setup fallback patterns if real data not available"""
        logger.info("Using fallback patterns...")
        
        self.real_products = ['GIFT001', 'BOOK123', 'MUG456', 'TSHIRT789', 'LAPTOP001', 'PHONE123']
        self.real_countries = ['United Kingdom', 'Germany', 'France', 'Netherlands', 'Spain', 'Italy']
        
        self.price_patterns = {
            'GIFT001': {'min_price': 5.0, 'max_price': 50.0, 'avg_price': 15.0, 'description': 'Gift Card'},
            'BOOK123': {'min_price': 10.0, 'max_price': 40.0, 'avg_price': 25.0, 'description': 'Programming Book'},
            'MUG456': {'min_price': 5.0, 'max_price': 15.0, 'avg_price': 8.5, 'description': 'Coffee Mug'},
            'TSHIRT789': {'min_price': 10.0, 'max_price': 25.0, 'avg_price': 15.0, 'description': 'Cotton T-Shirt'},
            'LAPTOP001': {'min_price': 400.0, 'max_price': 800.0, 'avg_price': 600.0, 'description': 'Laptop Computer'},
            'PHONE123': {'min_price': 200.0, 'max_price': 400.0, 'avg_price': 300.0, 'description': 'Mobile Phone'}
        }
    
    def generate_realistic_order(self, order_type='random'):
        """
        Generate a realistic fake order based on real data patterns
        
        Args:
            order_type: 'random', 'high_value', 'bulk', 'single'
        """
        
        # Select product based on order type
        if order_type == 'high_value':
            # Favor expensive products
            expensive_products = [p for p, pattern in self.price_patterns.items() 
                                if pattern['avg_price'] > 50]
            product_code = random.choice(expensive_products) if expensive_products else random.choice(self.real_products)
        else:
            product_code = random.choice(self.real_products)
        
        # Get product pattern
        if product_code in self.price_patterns:
            pattern = self.price_patterns[product_code]
            # Generate price within realistic range
            unit_price = round(random.uniform(
                pattern['min_price'], 
                min(pattern['max_price'], pattern['avg_price'] * 1.5)
            ), 2)
            description = pattern['description']
        else:
            unit_price = round(random.uniform(5.0, 50.0), 2)
            description = f"Product {product_code}"
        
        # Generate quantity based on order type
        if order_type == 'bulk':
            quantity = random.randint(10, 50)
        elif order_type == 'single':
            quantity = 1
        else:
            # Realistic quantity distribution (mostly 1-5, occasionally more)
            if random.random() < 0.7:
                quantity = random.randint(1, 5)
            else:
                quantity = random.randint(6, 20)
        
        # Generate realistic timestamp (recent, with business hour bias)
        if random.random() < 0.6:  # 60% during business hours
            hour = random.randint(9, 17)
        else:
            hour = random.randint(0, 23)
        
        minutes_ago = random.randint(0, 1440)  # Up to 24 hours ago
        timestamp = datetime.now() - timedelta(minutes=minutes_ago)
        timestamp = timestamp.replace(hour=hour, minute=random.randint(0, 59))
        
        order = {
            'InvoiceNo': self.fake.uuid4()[:8].upper(),
            'StockCode': product_code,
            'Description': description[:50],  # Limit description length
            'Quantity': quantity,
            'InvoiceDate': timestamp.isoformat(),
            'UnitPrice': unit_price,
            'CustomerID': random.randint(10000, 99999),
            'Country': random.choice(self.real_countries)
        }
        
        # Add calculated revenue for convenience
        order['Revenue'] = round(quantity * unit_price, 2)
        
        return order
    
    def generate_batch_orders(self, num_orders=100, order_mix=None):
        """
        Generate a batch of orders with realistic distribution
        
        Args:
            num_orders: Number of orders to generate
            order_mix: Dict with order type distribution, e.g., {'random': 0.7, 'high_value': 0.2, 'bulk': 0.1}
        """
        
        if order_mix is None:
            order_mix = {
                'random': 0.6,
                'high_value': 0.2,
                'bulk': 0.1,
                'single': 0.1
            }
        
        orders = []
        
        for i in range(num_orders):
            # Select order type based on distribution
            rand_val = random.random()
            cumulative = 0
            selected_type = 'random'
            
            for order_type, probability in order_mix.items():
                cumulative += probability
                if rand_val <= cumulative:
                    selected_type = order_type
                    break
            
            order = self.generate_realistic_order(selected_type)
            orders.append(order)
        
        return orders
    
    def generate_streaming_order(self):
        """Generate order optimized for streaming (lightweight, realistic)"""
        return self.generate_realistic_order('random')
    
    def export_to_csv(self, orders, filename):
        """Export generated orders to CSV file"""
        df = pd.DataFrame(orders)
        df.to_csv(filename, index=False)
        logger.info(f"Exported {len(orders)} orders to {filename}")
    
    def get_data_quality_summary(self, orders):
        """Generate data quality summary for generated orders"""
        df = pd.DataFrame(orders)
        
        summary = {
            'total_orders': len(orders),
            'total_revenue': df['Revenue'].sum(),
            'avg_order_value': df['Revenue'].mean(),
            'unique_products': df['StockCode'].nunique(),
            'unique_customers': df['CustomerID'].nunique(),
            'unique_countries': df['Country'].nunique(),
            'revenue_range': {
                'min': df['Revenue'].min(),
                'max': df['Revenue'].max(),
                'median': df['Revenue'].median()
            },
            'quantity_distribution': {
                'min': df['Quantity'].min(),
                'max': df['Quantity'].max(),
                'avg': df['Quantity'].mean()
            }
        }
        
        return summary

# Backward compatibility functions for existing code
def generate_fake_order():
    """Generate one fake e-commerce order (backward compatibility)"""
    generator = EcommerceFakeDataGenerator()
    return generator.generate_realistic_order()

# Demonstration and testing functions
def demo_generator():
    """Demonstrate the fake data generator capabilities"""
    print("🏭 E-Commerce Fake Data Generator Demo")
    print("=" * 50)
    
    generator = EcommerceFakeDataGenerator()
    
    print("\n1. Generating different order types:")
    
    # Generate sample orders
    order_types = ['random', 'high_value', 'bulk', 'single']
    for order_type in order_types:
        order = generator.generate_realistic_order(order_type)
        print(f"\n{order_type.title()} Order:")
        print(f"  Product: {order['StockCode']} - {order['Description']}")
        print(f"  Quantity: {order['Quantity']} x £{order['UnitPrice']} = £{order['Revenue']}")
        print(f"  Customer: {order['CustomerID']} from {order['Country']}")
    
    print("\n2. Generating batch of orders:")
    orders = generator.generate_batch_orders(20)
    summary = generator.get_data_quality_summary(orders)
    
    print(f"Generated {summary['total_orders']} orders:")
    print(f"  Total Revenue: £{summary['total_revenue']:.2f}")
    print(f"  Avg Order Value: £{summary['avg_order_value']:.2f}")
    print(f"  Products: {summary['unique_products']}")
    print(f"  Countries: {summary['unique_countries']}")
    print(f"  Revenue Range: £{summary['revenue_range']['min']:.2f} - £{summary['revenue_range']['max']:.2f}")
    
    print("\n3. Sample orders for streaming:")
    print("Perfect for Publisher compatibility:")
    for i in range(3):
        streaming_order = generator.generate_streaming_order()
        print(f"  {i+1}. {streaming_order['InvoiceNo']}: £{streaming_order['Revenue']:.2f} ({streaming_order['Country']})")

def generate_test_data_for_team():
    """Generate test data files for team development"""
    print("📁 Generating test data files for team...")
    
    generator = EcommerceFakeDataGenerator()
    
    # Generate different datasets
    datasets = {
        'streaming_test_data.json': generator.generate_batch_orders(50, {'random': 0.8, 'high_value': 0.2}),
        'batch_test_data.json': generator.generate_batch_orders(200, {'random': 0.5, 'high_value': 0.3, 'bulk': 0.2}),
        'ml_test_data.json': generator.generate_batch_orders(100, {'random': 1.0})
    }
    
    # Save datasets
    for filename, orders in datasets.items():
        with open(f'data/{filename}', 'w') as f:
            json.dump(orders, f, indent=2, default=str)
        
        summary = generator.get_data_quality_summary(orders)
        print(f"✅ {filename}: {len(orders)} orders, £{summary['total_revenue']:.2f} revenue")

if __name__ == "__main__":
    # Run demo
    demo_generator()
    
    # Ask if user wants to generate test files
    print(f"\n❓ Generate test data files for team development? (y/n): ", end="")
    response = input().lower().strip()
    if response == 'y':
        generate_test_data_for_team()
        print("📁 Test data files saved in data/ directory")
    
    print("\n✅ Fake data generator ready for use!")