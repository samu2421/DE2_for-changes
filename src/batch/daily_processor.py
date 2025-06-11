# src/batch/daily_processor.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path
import logging
from google.cloud import bigquery
import os

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
    BigQuery-powered batch processor for e-commerce analytics
    Handles daily/historical data processing with robust error handling
    """
    
    def __init__(self, project_id='ecommerce-analytics-462115', dataset_id='ecommerce_data', table_id='historical_orders'):
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.table_id = table_id
        self.table_ref = f"{project_id}.{dataset_id}.{table_id}"
        self.client = None
        self.df = None
        self.processed_metrics = {}
        
    def initialize_bigquery_client(self):
        """Initialize BigQuery client with authentication"""
        try:
            # Initialize BigQuery client
            self.client = bigquery.Client(project=self.project_id)
            logger.info(f"BigQuery client initialized for project: {self.project_id}")
            
            # Test connection by listing datasets
            datasets = list(self.client.list_datasets())
            logger.info(f"Found {len(datasets)} datasets in project")
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing BigQuery client: {e}")
            logger.error("Make sure GOOGLE_APPLICATION_CREDENTIALS is set and points to your service account key")
            return False
    
    def load_data_from_bigquery(self, limit=None):
        """Load and prepare the dataset from BigQuery"""
        if not self.client:
            logger.error("BigQuery client not initialized. Call initialize_bigquery_client() first.")
            return False
            
        try:
            logger.info(f"Loading e-commerce dataset from BigQuery table: {self.table_ref}")
            
            # Construct query with optional limit for testing
            limit_clause = f"LIMIT {limit}" if limit else ""
            
            query = f"""
            SELECT 
                InvoiceNo,
                StockCode,
                Description,
                Quantity,
                InvoiceDate,
                UnitPrice,
                CustomerID,
                Country
            FROM `{self.table_ref}`
            WHERE 
                Quantity > 0 
                AND UnitPrice > 0
                AND CustomerID IS NOT NULL
                AND InvoiceDate IS NOT NULL
            ORDER BY InvoiceDate DESC
            {limit_clause}
            """
            
            logger.info("Executing BigQuery query...")
            self.df = self.client.query(query).to_dataframe()
            
            if self.df.empty:
                logger.warning("No data returned from BigQuery query")
                return False
            
            # Feature engineering
            self.df['Revenue'] = self.df['Quantity'] * self.df['UnitPrice']
            self.df['InvoiceDate'] = pd.to_datetime(self.df['InvoiceDate'])
            self.df['Date'] = self.df['InvoiceDate'].dt.date
            self.df['Hour'] = self.df['InvoiceDate'].dt.hour
            self.df['DayOfWeek'] = self.df['InvoiceDate'].dt.dayofweek
            self.df['Month'] = self.df['InvoiceDate'].dt.month
            self.df['Year'] = self.df['InvoiceDate'].dt.year
            
            logger.info(f"Dataset loaded from BigQuery: {len(self.df):,} clean transactions")
            logger.info(f"Date range: {self.df['InvoiceDate'].min()} to {self.df['InvoiceDate'].max()}")
            logger.info(f"Total revenue: £{self.df['Revenue'].sum():,.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading data from BigQuery: {e}")
            return False
    
    def load_data_sample_for_testing(self):
        """Load a small sample from BigQuery for testing purposes"""
        logger.info("Loading sample data for testing...")
        return self.load_data_from_bigquery(limit=10000)
    
    def process_daily_metrics(self, target_date=None):
        """
        Process daily business metrics from BigQuery data
        If target_date is None, processes the most recent date
        """
        if self.df is None:
            logger.error("Data not loaded. Call load_data_from_bigquery() first.")
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
    
    def run_advanced_bigquery_analytics(self):
        """Run advanced analytics directly in BigQuery for better performance - FIXED VERSION"""
        if not self.client:
            logger.error("BigQuery client not initialized.")
            return None
            
        logger.info("Running advanced analytics in BigQuery...")
        
        try:
            # Fixed analytics query - separate queries instead of problematic UNION
            
            # Query 1: Daily trends
            daily_query = f"""
            SELECT 
                DATE(InvoiceDate) as order_date,
                COUNT(*) as total_orders,
                ROUND(SUM(Quantity * UnitPrice), 2) as total_revenue,
                COUNT(DISTINCT CustomerID) as unique_customers,
                COUNT(DISTINCT StockCode) as unique_products,
                ROUND(AVG(Quantity * UnitPrice), 2) as avg_order_value
            FROM `{self.table_ref}`
            WHERE 
                Quantity > 0 
                AND UnitPrice > 0
                AND CustomerID IS NOT NULL
            GROUP BY DATE(InvoiceDate)
            ORDER BY order_date DESC
            LIMIT 7
            """
            
            logger.info("Executing daily trends query...")
            daily_results = self.client.query(daily_query).to_dataframe()
            
            # Query 2: Customer segments  
            segment_query = f"""
            WITH customer_metrics AS (
                SELECT 
                    CustomerID,
                    SUM(Quantity * UnitPrice) as total_revenue,
                    COUNT(DISTINCT InvoiceNo) as order_frequency
                FROM `{self.table_ref}`
                WHERE 
                    Quantity > 0 
                    AND UnitPrice > 0
                    AND CustomerID IS NOT NULL
                GROUP BY CustomerID
            )
            SELECT 
                CASE 
                    WHEN total_revenue >= 2000 THEN 'VIP'
                    WHEN total_revenue >= 1000 THEN 'High Value'
                    WHEN total_revenue >= 200 THEN 'Medium Value'
                    ELSE 'Low Value'
                END as customer_segment,
                COUNT(*) as customer_count,
                ROUND(AVG(total_revenue), 2) as avg_revenue,
                ROUND(AVG(order_frequency), 2) as avg_order_frequency
            FROM customer_metrics
            GROUP BY customer_segment
            ORDER BY avg_revenue DESC
            """
            
            logger.info("Executing customer segmentation query...")
            segment_results = self.client.query(segment_query).to_dataframe()
            
            # Query 3: Product performance
            product_query = f"""
            SELECT 
                StockCode,
                ANY_VALUE(Description) as product_name,
                SUM(Quantity) as total_quantity_sold,
                ROUND(SUM(Quantity * UnitPrice), 2) as total_revenue,
                COUNT(DISTINCT CustomerID) as unique_customers,
                COUNT(DISTINCT InvoiceNo) as total_orders
            FROM `{self.table_ref}`
            WHERE 
                Quantity > 0 
                AND UnitPrice > 0
                AND CustomerID IS NOT NULL
            GROUP BY StockCode
            ORDER BY total_revenue DESC
            LIMIT 10
            """
            
            logger.info("Executing product performance query...")
            product_results = self.client.query(product_query).to_dataframe()
            
            # Process results
            analytics_results = {
                'daily_trends': [],
                'customer_segments': {},
                'top_products': [],
                'summary': {
                    'total_days_analyzed': len(daily_results),
                    'customer_segments_identified': len(segment_results),
                    'top_products_analyzed': len(product_results)
                }
            }
            
            # Process daily trends
            for _, row in daily_results.iterrows():
                analytics_results['daily_trends'].append({
                    'date': str(row['order_date']),
                    'revenue': float(row['total_revenue']),
                    'orders': int(row['total_orders']),
                    'customers': int(row['unique_customers']),
                    'products': int(row['unique_products']),
                    'avg_order_value': float(row['avg_order_value'])
                })
            
            # Process customer segments
            for _, row in segment_results.iterrows():
                analytics_results['customer_segments'][row['customer_segment']] = {
                    'customer_count': int(row['customer_count']),
                    'avg_revenue': float(row['avg_revenue']),
                    'avg_order_frequency': float(row['avg_order_frequency'])
                }
            
            # Process top products
            for _, row in product_results.iterrows():
                analytics_results['top_products'].append({
                    'stock_code': row['StockCode'],
                    'product_name': row['product_name'][:50] if row['product_name'] else 'Unknown',
                    'total_quantity': int(row['total_quantity_sold']),
                    'total_revenue': float(row['total_revenue']),
                    'unique_customers': int(row['unique_customers']),
                    'total_orders': int(row['total_orders'])
                })
            
            logger.info("Advanced BigQuery analytics completed successfully")
            logger.info(f"✅ Processed {len(analytics_results['daily_trends'])} days of trends")
            logger.info(f"✅ Identified {len(analytics_results['customer_segments'])} customer segments")
            logger.info(f"✅ Analyzed {len(analytics_results['top_products'])} top products")
            
            # Log some interesting insights
            if analytics_results['daily_trends']:
                latest_day = analytics_results['daily_trends'][0]
                logger.info(f"📊 Latest day revenue: £{latest_day['revenue']:,} from {latest_day['orders']:,} orders")
            
            if analytics_results['customer_segments']:
                vip_segment = analytics_results['customer_segments'].get('VIP')
                if vip_segment:
                    logger.info(f"👑 VIP customers: {vip_segment['customer_count']} (avg £{vip_segment['avg_revenue']:,.2f})")
            
            return analytics_results
            
        except Exception as e:
            logger.error(f"Error running BigQuery analytics: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def process_weekly_trends(self):
        """Process weekly trend analysis"""
        if self.df is None:
            logger.error("Data not loaded. Call load_data_from_bigquery() first.")
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
            logger.error("Data not loaded. Call load_data_from_bigquery() first.")
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
    
    def save_results_to_bigquery(self, output_table_suffix='processed_metrics'):
        """Save processed results back to BigQuery"""
        if not self.client or not self.processed_metrics:
            logger.error("No processed metrics to save or BigQuery client not initialized")
            return False
            
        try:
            # Create output table name
            output_table = f"{self.dataset_id}.{output_table_suffix}"
            
            # Convert metrics to DataFrame
            metrics_df = pd.DataFrame([self.processed_metrics])
            metrics_df['processing_timestamp'] = datetime.now()
            
            # Configure job
            job_config = bigquery.LoadJobConfig(
                write_disposition="WRITE_APPEND",  # Append to existing table
                autodetect=True  # Auto-detect schema
            )
            
            # Load data to BigQuery
            job = self.client.load_table_from_dataframe(
                metrics_df, 
                f"{self.project_id}.{output_table}", 
                job_config=job_config
            )
            
            job.result()  # Wait for job to complete
            
            logger.info(f"Metrics saved to BigQuery table: {output_table}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving results to BigQuery: {e}")
            return False
    
    def save_results(self, output_dir='data/processed'):
        """Save processed results to files (fallback method)"""
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
        print("DAILY E-COMMERCE ANALYTICS REPORT (BigQuery)")
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

    def generate_advanced_analytics_report(self, analytics_results):
        """Generate a comprehensive analytics report from BigQuery results"""
        if not analytics_results:
            logger.error("No analytics results available.")
            return
            
        print("\n" + "="*60)
        print("ADVANCED E-COMMERCE ANALYTICS REPORT (BigQuery)")
        print("="*60)
        
        # Daily trends summary
        if analytics_results.get('daily_trends'):
            print(f"\n📈 DAILY TRENDS (Last {len(analytics_results['daily_trends'])} Days):")
            total_revenue = sum(day['revenue'] for day in analytics_results['daily_trends'])
            total_orders = sum(day['orders'] for day in analytics_results['daily_trends'])
            
            print(f"   Total Revenue: £{total_revenue:,.2f}")
            print(f"   Total Orders: {total_orders:,}")
            print(f"   Average Daily Revenue: £{total_revenue / len(analytics_results['daily_trends']):,.2f}")
            
            print(f"\n   Recent Days:")
            for day in analytics_results['daily_trends'][:3]:
                print(f"     {day['date']}: £{day['revenue']:,} ({day['orders']:,} orders, {day['customers']:,} customers)")
        
        # Customer segments
        if analytics_results.get('customer_segments'):
            print(f"\n👥 CUSTOMER SEGMENTATION:")
            for segment, data in analytics_results['customer_segments'].items():
                print(f"   {segment}: {data['customer_count']:,} customers")
                print(f"     Avg Revenue: £{data['avg_revenue']:,.2f}")
                print(f"     Avg Orders: {data['avg_order_frequency']:.1f}")
        
        # Top products
        if analytics_results.get('top_products'):
            print(f"\n🏆 TOP PERFORMING PRODUCTS:")
            for i, product in enumerate(analytics_results['top_products'][:5], 1):
                print(f"   {i}. {product['stock_code']}: £{product['total_revenue']:,}")
                print(f"      {product['total_quantity']:,} units sold to {product['unique_customers']:,} customers")
        
        print("="*60 + "\n")


def main():
    """Main batch processing function with BigQuery integration"""
    print("Starting E-Commerce Batch Processing with BigQuery...")
    
    # Initialize processor
    processor = EcommerceBatchProcessor()
    
    # Initialize BigQuery client
    if not processor.initialize_bigquery_client():
        print("Failed to initialize BigQuery client. Exiting.")
        print("Make sure:")
        print("1. GOOGLE_APPLICATION_CREDENTIALS environment variable is set")
        print("2. Service account key file exists and has BigQuery permissions")
        print("3. Project ID and dataset exist in BigQuery")
        return
    
    # Load data from BigQuery
    print("Loading data from BigQuery...")
    if not processor.load_data_from_bigquery():
        print("Failed to load data from BigQuery. Trying sample data...")
        if not processor.load_data_sample_for_testing():
            print("Failed to load any data. Exiting.")
            return
    
    # Process daily metrics
    metrics = processor.process_daily_metrics()
    if metrics:
        processor.generate_daily_report()
    
    # Run advanced BigQuery analytics
    print("\nRunning advanced BigQuery analytics...")
    advanced_results = processor.run_advanced_bigquery_analytics()
    if advanced_results:
        print("✅ Advanced analytics completed successfully!")
        processor.generate_advanced_analytics_report(advanced_results)
    else:
        print("⚠️ Advanced analytics encountered issues, but basic processing completed.")
    
    # Save results to BigQuery
    print("\nSaving results back to BigQuery...")
    if processor.save_results_to_bigquery():
        print("✅ Results saved to BigQuery successfully!")
    else:
        print("⚠️ Failed to save to BigQuery, saving to local files...")
        processor.save_results()
    
    print("\n🎉 Batch processing completed successfully!")


if __name__ == "__main__":
    main()