# src/batch/daily_processor.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('docs/batch_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class EcommerceBatchProcessor:
    """
    Batch processor for e-commerce analytics
    Handles daily/historical data processing with robust error handling
    """
    
    def __init__(self, data_path='data/OnlineRetail.csv'):
        self.data_path = data_path
        self.df = None
        self.processed_metrics = {}
        
    def load_data(self):
        """Load and prepare the dataset"""
        try:
            logger.info("Loading e-commerce dataset...")
            self.df = pd.read_csv(self.data_path, encoding='latin1')
            
            # Data cleaning
            self.df = self.df.dropna()
            self.df = self.df[self.df['Quantity'] > 0]
            self.df = self.df[self.df['UnitPrice'] > 0]
            
            # Feature engineering
            self.df['Revenue'] = self.df['Quantity'] * self.df['UnitPrice']
            self.df['InvoiceDate'] = pd.to_datetime(self.df['InvoiceDate'])
            self.df['Date'] = self.df['InvoiceDate'].dt.date
            self.df['Hour'] = self.df['InvoiceDate'].dt.hour
            self.df['DayOfWeek'] = self.df['InvoiceDate'].dt.dayofweek
            self.df['Month'] = self.df['InvoiceDate'].dt.month
            self.df['Year'] = self.df['InvoiceDate'].dt.year
            
            logger.info(f"Dataset loaded: {len(self.df):,} clean transactions")
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def process_daily_metrics(self, target_date=None):
        """
        Process daily business metrics
        If target_date is None, processes the most recent date
        """
        if self.df is None:
            logger.error("Data not loaded. Call load_data() first.")
            return None
            
        if target_date is None:
            # Get the most recent date in dataset
            target_date = self.df['Date'].max()
        
        logger.info(f"Processing metrics for {target_date}")
        
        # Filter data for target date
        daily_data = self.df[self.df['Date'] == target_date]
        
        if daily_data.empty:
            logger.warning(f"No data found for {target_date}")
            return None
        
        # Calculate daily metrics
        metrics = {
            'date': str(target_date),
            'total_orders': len(daily_data),
            'total_revenue': round(daily_data['Revenue'].sum(), 2),
            'unique_customers': daily_data['CustomerID'].nunique(),
            'unique_products': daily_data['StockCode'].nunique(),
            'avg_order_value': round(daily_data['Revenue'].mean(), 2),
            'total_quantity': int(daily_data['Quantity'].sum()),
            'countries_served': daily_data['Country'].nunique(),
            'peak_hour': int(daily_data.groupby('Hour')['Revenue'].sum().idxmax()),
            'peak_hour_revenue': round(daily_data.groupby('Hour')['Revenue'].sum().max(), 2)
        }
        
        # Top performers for the day
        top_customers = daily_data.groupby('CustomerID')['Revenue'].sum().sort_values(ascending=False).head(3)
        top_products = daily_data.groupby('StockCode')['Revenue'].sum().sort_values(ascending=False).head(3)
        top_countries = daily_data.groupby('Country')['Revenue'].sum().sort_values(ascending=False).head(3)
        
        metrics['top_customers'] = [
            {'customer_id': int(cid), 'revenue': round(rev, 2)} 
            for cid, rev in top_customers.items()
        ]
        
        metrics['top_products'] = [
            {'product_code': code, 'revenue': round(rev, 2)} 
            for code, rev in top_products.items()
        ]
        
        metrics['top_countries'] = [
            {'country': country, 'revenue': round(rev, 2)} 
            for country, rev in top_countries.items()
        ]
        
        self.processed_metrics = metrics
        logger.info(f"Daily metrics processed: {metrics['total_orders']} orders, £{metrics['total_revenue']}")
        
        return metrics
    
    def process_weekly_trends(self):
        """Process weekly trend analysis"""
        if self.df is None:
            logger.error("Data not loaded. Call load_data() first.")
            return None
            
        logger.info("Processing weekly trends...")
        
        # Get last 7 days of data
        latest_date = self.df['Date'].max()
        week_start = latest_date - timedelta(days=6)
        week_data = self.df[self.df['Date'] >= week_start]
        
        # Daily trends
        daily_trends = week_data.groupby('Date').agg({
            'Revenue': 'sum',
            'InvoiceNo': 'count',
            'CustomerID': 'nunique'
        }).round(2)
        
        trends = {
            'week_start': str(week_start),
            'week_end': str(latest_date),
            'total_weekly_revenue': round(week_data['Revenue'].sum(), 2),
            'daily_trends': [
                {
                    'date': str(date),
                    'revenue': float(row['Revenue']),
                    'orders': int(row['InvoiceNo']),
                    'customers': int(row['CustomerID'])
                }
                for date, row in daily_trends.iterrows()
            ],
            'revenue_growth': self._calculate_growth_rate(daily_trends['Revenue']),
            'order_growth': self._calculate_growth_rate(daily_trends['InvoiceNo'])
        }
        
        logger.info(f"Weekly trends processed: £{trends['total_weekly_revenue']} total revenue")
        return trends
    
    def process_customer_segmentation(self):
        """Process customer segmentation analysis"""
        if self.df is None:
            logger.error("Data not loaded. Call load_data() first.")
            return None
            
        logger.info("Processing customer segmentation...")
        
        # Customer lifetime metrics
        customer_metrics = self.df.groupby('CustomerID').agg({
            'Revenue': 'sum',
            'InvoiceNo': 'nunique',
            'Quantity': 'sum',
            'InvoiceDate': ['min', 'max']
        }).round(2)
        
        customer_metrics.columns = ['total_revenue', 'order_frequency', 'total_quantity', 'first_order', 'last_order']
        customer_metrics['days_active'] = (customer_metrics['last_order'] - customer_metrics['first_order']).dt.days
        
        # Customer segments based on revenue
        revenue_quantiles = customer_metrics['total_revenue'].quantile([0.33, 0.67, 0.95])
        
        def categorize_customer(revenue):
            if revenue >= revenue_quantiles[0.95]:
                return 'VIP'
            elif revenue >= revenue_quantiles[0.67]:
                return 'High Value'
            elif revenue >= revenue_quantiles[0.33]:
                return 'Medium Value'
            else:
                return 'Low Value'
        
        customer_metrics['segment'] = customer_metrics['total_revenue'].apply(categorize_customer)
        segment_summary = customer_metrics.groupby('segment').agg({
            'total_revenue': ['count', 'sum', 'mean'],
            'order_frequency': 'mean'
        }).round(2)
        
        segmentation = {
            'total_customers': len(customer_metrics),
            'segments': {
                segment: {
                    'customer_count': int(data[('total_revenue', 'count')]),
                    'total_revenue': float(data[('total_revenue', 'sum')]),
                    'avg_revenue_per_customer': float(data[('total_revenue', 'mean')]),
                    'avg_order_frequency': float(data[('order_frequency', 'mean')])
                }
                for segment, data in segment_summary.iterrows()
            }
        }
        
        logger.info(f"Customer segmentation complete: {len(customer_metrics)} customers analyzed")
        return segmentation
    
    def _calculate_growth_rate(self, series):
        """Calculate growth rate for a time series"""
        if len(series) < 2:
            return 0.0
        return round(((series.iloc[-1] - series.iloc[0]) / series.iloc[0]) * 100, 2)
    
    def save_results(self, output_dir='data/processed'):
        """Save processed results to files"""
        Path(output_dir).mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if self.processed_metrics:
            # Save daily metrics
            daily_file = f"{output_dir}/daily_metrics_{timestamp}.json"
            with open(daily_file, 'w') as f:
                json.dump(self.processed_metrics, f, indent=2)
            logger.info(f"Daily metrics saved to {daily_file}")
        
        # Save weekly trends
        weekly_trends = self.process_weekly_trends()
        if weekly_trends:
            weekly_file = f"{output_dir}/weekly_trends_{timestamp}.json"
            with open(weekly_file, 'w') as f:
                json.dump(weekly_trends, f, indent=2)
            logger.info(f"Weekly trends saved to {weekly_file}")
        
        # Save customer segmentation
        customer_seg = self.process_customer_segmentation()
        if customer_seg:
            seg_file = f"{output_dir}/customer_segmentation_{timestamp}.json"
            with open(seg_file, 'w') as f:
                json.dump(customer_seg, f, indent=2)
            logger.info(f"Customer segmentation saved to {seg_file}")
    
    def generate_daily_report(self):
        """Generate a comprehensive daily report"""
        if not self.processed_metrics:
            logger.error("No processed metrics available. Run process_daily_metrics() first.")
            return
            
        print("\n" + "="*50)
        print("DAILY E-COMMERCE ANALYTICS REPORT")
        print("="*50)
        
        metrics = self.processed_metrics
        print(f"Date: {metrics['date']}")
        print(f"Total Revenue: £{metrics['total_revenue']:,}")
        print(f"Total Orders: {metrics['total_orders']:,}")
        print(f"Unique Customers: {metrics['unique_customers']:,}")
        print(f"Unique Products: {metrics['unique_products']:,}")
        print(f"Avg Order Value: £{metrics['avg_order_value']}")
        print(f"Total Quantity: {metrics['total_quantity']:,}")
        print(f"Countries Served: {metrics['countries_served']}")
        print(f"Peak Hour: {metrics['peak_hour']:02d}:00 (£{metrics['peak_hour_revenue']})")
        
        print(f"\n TOP PERFORMERS:")
        print("Top Customers:")
        for i, customer in enumerate(metrics['top_customers'], 1):
            print(f"  {i}. Customer {customer['customer_id']}: £{customer['revenue']}")
        
        print("Top Products:")
        for i, product in enumerate(metrics['top_products'], 1):
            print(f"  {i}. {product['product_code']}: £{product['revenue']}")
        
        print("Top Countries:")
        for i, country in enumerate(metrics['top_countries'], 1):
            print(f"  {i}. {country['country']}: £{country['revenue']}")
        
        print("="*50 + "\n")

def main():
    """Main batch processing function"""
    print("Starting E-Commerce Batch Processing...")
    
    # Initialize processor
    processor = EcommerceBatchProcessor()
    
    # Load data
    if not processor.load_data():
        print("Failed to load data. Exiting.")
        return
    
    # Process daily metrics
    metrics = processor.process_daily_metrics()
    if metrics:
        processor.generate_daily_report()
    
    # Save all results
    processor.save_results()
    
    print("Batch processing completed successfully!")

if __name__ == "__main__":
    main()