import pandas as pd
from faker import Faker
import random
from datetime import datetime, timedelta

# Initialize Faker
fake = Faker()

# Load real product and country data from the Kaggle dataset
real_data = pd.read_csv('data/OnlineRetail.csv', encoding='latin1')

# Get a sample of valid product codes and countries
real_products = real_data['StockCode'].dropna().unique()[:100]
real_countries = real_data['Country'].dropna().unique()

def generate_fake_order():
    """Generate one fake e-commerce order"""
    return {
        'InvoiceNo': fake.uuid4()[:6],  # Short fake invoice number
        'StockCode': random.choice(real_products),
        'Description': fake.text(max_nb_chars=30),
        'Quantity': random.randint(1, 10),
        'InvoiceDate': datetime.now() - timedelta(minutes=random.randint(0, 1440)),
        'UnitPrice': round(random.uniform(1.0, 50.0), 2),
        'CustomerID': random.randint(10000, 99999),
        'Country': random.choice(real_countries)
    }

# Test run
if __name__ == "__main__":
    print("Generating 5 fake orders:\n")
    for i in range(5):
        order = generate_fake_order()
        print(order)