# src/tasks/data_processing_tasks.py
"""
Airflow task functions for existing e-commerce project
"""

import pandas as pd
from datetime import datetime
import json
import logging
from pathlib import Path
import os

logger = logging.getLogger(__name__)

def initialize_bigquery_connection(**context):
    """Initialize BigQuery connection using existing credentials"""
    try:
        # Use existing service key
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'gcp-service-key.json'
        
        from google.cloud import bigquery
        
        project_id = context['params'].get('project_id', 'ecommerce-analytics-462115')
        client = bigquery.Client(project=project_id)
        
        # Test connection
        datasets = list(client.list_datasets())
        
        logger.info(f"✅ BigQuery connection successful. Found {len(datasets)} datasets")
        
        return {
            'status': 'success',
            'project_id': project_id,
            'dataset_count': len(datasets),
            'connection_time': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.warning(f"BigQuery connection failed: {e}")
        # Fall back to sample data
        logger.info("Using sample data for testing...")
        return {
            'status': 'success_sample',
            'project_id': 'sample-project',
            'dataset_count': 1,
            'connection_time': datetime.now().isoformat()
        }


def extract_daily_data(**context):
    """Extract data - try BigQuery first, fall back to local CSV"""
    bigquery_config = context['task_instance'].xcom_pull(task_ids='initialize_bigquery')
    
    try:
        if bigquery_config['status'] == 'success':
            # Try BigQuery extraction
            return extract_from_bigquery(context, bigquery_config)
        else:
            # Fall back to local CSV
            return extract_from_local_csv()
            
    except Exception as e:
        logger.warning(f"BigQuery extraction failed: {e}")
        return extract_from_local_csv()


def extract_from_bigquery(context, bigquery_config):
    """Extract from BigQuery"""
    from google.cloud import bigquery
    
    project_id = bigquery_config['project_id']
    client = bigquery.Client(project=project_id)
    
    query = f"""
    SELECT 
        InvoiceNo,
        StockCode,
        Description,
        Quantity,
        InvoiceDate,
        UnitPrice,
        CustomerID,
        Country,
        (Quantity * UnitPrice) as Revenue
    FROM `{project_id}.ecommerce_data.historical_orders`
    WHERE 
        Quantity > 0 
        AND UnitPrice > 0
        AND CustomerID IS NOT NULL
    ORDER BY InvoiceDate DESC
    LIMIT 10000
    """
    
    df = client.query(query).to_dataframe()
    
    # Add feature engineering
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['Date'] = df['InvoiceDate'].dt.date
    df['Hour'] = df['InvoiceDate'].dt.hour
    df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek
    
    # Save to data/processed
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    data_path = f"data/processed/bigquery_extract_{timestamp}.csv"
    df.to_csv(data_path, index=False)
    
    logger.info(f"✅ BigQuery data extracted: {len(df):,} rows saved to {data_path}")
    
    return {
        'status': 'success',
        'data_source': 'bigquery',
        'data_path': data_path,
        'row_count': len(df),
        'total_revenue': float(df['Revenue'].sum())
    }


def extract_from_local_csv():
    """Extract from local CSV file"""
    try:
        # Try to load your existing OnlineRetail.csv
        csv_files = [
            'data/OnlineRetail.csv',
            'data/raw/OnlineRetail.csv',
            'data/cleaned/OnlineRetail.csv'
        ]
        
        df = None
        source_file = None
        
        for csv_file in csv_files:
            if Path(csv_file).exists():
                df = pd.read_csv(csv_file, encoding='latin1')
                source_file = csv_file
                break
        
        if df is None:
            # Create sample data if no CSV found
            return create_sample_data()
        
        # Clean and process the data
        df = df.dropna(subset=['CustomerID', 'Quantity', 'UnitPrice'])
        df = df[df['Quantity'] > 0]
        df = df[df['UnitPrice'] > 0]
        
        # Take a sample for processing
        if len(df) > 10000:
            df = df.sample(n=10000, random_state=42)
        
        # Add feature engineering
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        df['Revenue'] = df['Quantity'] * df['UnitPrice']
        df['Date'] = df['InvoiceDate'].dt.date
        df['Hour'] = df['InvoiceDate'].dt.hour
        df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek
        
        # Save processed data
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        data_path = f"data/processed/csv_extract_{timestamp}.csv"
        df.to_csv(data_path, index=False)
        
        logger.info(f"✅ Local CSV data extracted: {len(df):,} rows from {source_file}")
        
        return {
            'status': 'success',
            'data_source': 'local_csv',
            'source_file': source_file,
            'data_path': data_path,
            'row_count': len(df),
            'total_revenue': float(df['Revenue'].sum())
        }
        
    except Exception as e:
        logger.warning(f"Local CSV extraction failed: {e}")
        return create_sample_data()


def create_sample_data():
    """Create sample data for testing"""
    sample_data = {
        'InvoiceNo': ['INV001', 'INV002', 'INV003', 'INV004', 'INV005'] * 20,
        'StockCode': ['PROD1', 'PROD2', 'PROD3', 'PROD4', 'PROD5'] * 20,
        'Description': ['Product 1', 'Product 2', 'Product 3', 'Product 4', 'Product 5'] * 20,
        'Quantity': [2, 1, 3, 1, 2] * 20,
        'UnitPrice': [10.50, 25.99, 8.75, 15.00, 12.50] * 20,
        'CustomerID': [12345, 12346, 12347, 12348, 12349] * 20,
        'Country': ['UK', 'Germany', 'France', 'UK', 'Spain'] * 20,
        'Revenue': [21.0, 25.99, 26.25, 15.00, 25.00] * 20
    }
    
    df = pd.DataFrame(sample_data)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    data_path = f"data/processed/sample_extract_{timestamp}.csv"
    df.to_csv(data_path, index=False)
    
    logger.info(f"✅ Sample data created: {len(df)} rows")
    
    return {
        'status': 'success',
        'data_source': 'sample',
        'data_path': data_path,
        'row_count': len(df),
        'total_revenue': float(df['Revenue'].sum())
    }


def validate_data_quality(**context):
    """Validate data quality"""
    extract_result = context['task_instance'].xcom_pull(task_ids='extract_daily_data')
    data_path = extract_result['data_path']
    
    try:
        df = pd.read_csv(data_path)
        
        # Quality checks
        quality_checks = {
            'total_rows': len(df),
            'missing_values': df.isnull().sum().to_dict(),
            'negative_quantities': (df['Quantity'] <= 0).sum(),
            'negative_prices': (df['UnitPrice'] <= 0).sum(),
            'duplicate_invoices': df['InvoiceNo'].duplicated().sum()
        }
        
        # Calculate quality score
        quality_score = 100
        issues = []
        
        if quality_checks['negative_quantities'] > 0:
            quality_score -= 10
            issues.append(f"Found {quality_checks['negative_quantities']} negative quantities")
        
        if quality_checks['negative_prices'] > 0:
            quality_score -= 10
            issues.append(f"Found {quality_checks['negative_prices']} negative prices")
        
        status = 'passed' if quality_score >= 80 else 'failed'
        
        logger.info(f"📋 Data quality check: {status} (Score: {quality_score})")
        if issues:
            for issue in issues:
                logger.warning(f"⚠️ {issue}")
        
        return {
            'status': status,
            'quality_score': quality_score,
            'quality_checks': quality_checks,
            'issues': issues,
            'data_path': data_path,
            'data_source': extract_result['data_source']
        }
        
    except Exception as e:
        logger.error(f"❌ Data quality validation failed: {e}")
        raise


def process_business_metrics(**context):
    """Process business metrics"""
    quality_result = context['task_instance'].xcom_pull(task_ids='validate_data_quality')
    
    if quality_result['status'] != 'passed':
        raise ValueError(f"Data quality check failed with score: {quality_result['quality_score']}")
    
    data_path = quality_result['data_path']
    
    try:
        df = pd.read_csv(data_path)
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        df['Date'] = df['InvoiceDate'].dt.date
        
        # Use the most recent date in the data
        latest_date = df['Date'].max()
        daily_data = df[df['Date'] == latest_date]
        
        if daily_data.empty:
            daily_data = df  # Use all data if no specific date
        
        # Calculate business metrics
        metrics = {
            'date': str(latest_date),
            'data_source': quality_result['data_source'],
            'total_orders': len(daily_data),
            'total_revenue': round(daily_data['Revenue'].sum(), 2),
            'unique_customers': daily_data['CustomerID'].nunique(),
            'unique_products': daily_data['StockCode'].nunique(),
            'avg_order_value': round(daily_data['Revenue'].mean(), 2),
            'total_quantity': int(daily_data['Quantity'].sum()),
            'countries_served': daily_data['Country'].nunique(),
        }
        
        # Add peak hour if we have hour data
        if 'Hour' in daily_data.columns:
            hourly_revenue = daily_data.groupby('Hour')['Revenue'].sum()
            if not hourly_revenue.empty:
                metrics['peak_hour'] = int(hourly_revenue.idxmax())
                metrics['peak_hour_revenue'] = round(hourly_revenue.max(), 2)
        
        # Top performers
        top_customers = daily_data.groupby('CustomerID')['Revenue'].sum().sort_values(ascending=False).head(3)
        top_products = daily_data.groupby('StockCode')['Revenue'].sum().sort_values(ascending=False).head(3)
        top_countries = daily_data.groupby('Country')['Revenue'].sum().sort_values(ascending=False).head(3)
        
        metrics.update({
            'top_customers': [
                {'customer_id': int(cid), 'revenue': round(rev, 2)} 
                for cid, rev in top_customers.items()
            ],
            'top_products': [
                {'product_code': code, 'revenue': round(rev, 2)} 
                for code, rev in top_products.items()
            ],
            'top_countries': [
                {'country': country, 'revenue': round(rev, 2)} 
                for country, rev in top_countries.items()
            ]
        })
        
        logger.info(f"💰 Business metrics: £{metrics['total_revenue']:,} revenue from {metrics['total_orders']:,} orders")
        
        return metrics
        
    except Exception as e:
        logger.error(f"❌ Business metrics processing failed: {e}")
        raise


def process_customer_segmentation(**context):
    """Process customer segmentation"""
    quality_result = context['task_instance'].xcom_pull(task_ids='validate_data_quality')
    data_path = quality_result['data_path']
    
    try:
        df = pd.read_csv(data_path)
        
        # Customer lifetime metrics
        customer_metrics = df.groupby('CustomerID').agg({
            'Revenue': 'sum',
            'InvoiceNo': 'nunique',
            'Quantity': 'sum'
        }).round(2)
        
        customer_metrics.columns = ['total_revenue', 'order_frequency', 'total_quantity']
        
        # Customer segments based on revenue percentiles
        if len(customer_metrics) > 0:
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
            
            # Segment summary
            segment_summary = customer_metrics.groupby('segment').agg({
                'total_revenue': ['count', 'sum', 'mean'],
                'order_frequency': 'mean'
            }).round(2)
            
            segmentation = {
                'total_customers': len(customer_metrics),
                'segments': {}
            }
            
            for segment in segment_summary.index:
                segmentation['segments'][segment] = {
                    'customer_count': int(segment_summary.loc[segment, ('total_revenue', 'count')]),
                    'total_revenue': float(segment_summary.loc[segment, ('total_revenue', 'sum')]),
                    'avg_revenue_per_customer': float(segment_summary.loc[segment, ('total_revenue', 'mean')]),
                    'avg_order_frequency': float(segment_summary.loc[segment, ('order_frequency', 'mean')])
                }
        else:
            segmentation = {
                'total_customers': 0,
                'segments': {}
            }
        
        logger.info(f"👥 Customer segmentation: {len(customer_metrics):,} customers analyzed")
        
        return segmentation
        
    except Exception as e:
        logger.error(f"❌ Customer segmentation failed: {e}")
        raise


def generate_daily_report(**context):
    """Generate comprehensive daily report"""
    metrics = context['task_instance'].xcom_pull(task_ids='process_business_metrics')
    segmentation = context['task_instance'].xcom_pull(task_ids='process_customer_segmentation')
    
    try:
        report_content = f"""
E-COMMERCE DAILY ANALYTICS REPORT (Airflow)
============================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
DAG Run: {context['dag_run'].run_id}
Data Source: {metrics.get('data_source', 'unknown')}

📊 BUSINESS METRICS:
Date: {metrics['date']}
Total Revenue: £{metrics['total_revenue']:,}
Total Orders: {metrics['total_orders']:,}
Unique Customers: {metrics['unique_customers']:,}
Unique Products: {metrics['unique_products']:,}
Avg Order Value: £{metrics['avg_order_value']}
Countries Served: {metrics['countries_served']}
"""
        
        if 'peak_hour' in metrics:
            report_content += f"Peak Hour: {metrics['peak_hour']:02d}:00 (£{metrics['peak_hour_revenue']:,})\n"
        
        report_content += f"""
👥 CUSTOMER SEGMENTATION:
Total Customers: {segmentation['total_customers']:,}
"""
        
        # Add segment details
        for segment, data in segmentation['segments'].items():
            report_content += f"{segment}: {data['customer_count']:,} customers (avg £{data['avg_revenue_per_customer']:.2f})\n"
        
        # Add top performers
        report_content += f"\n🏆 TOP PERFORMERS:\n"
        report_content += f"Top Countries:\n"
        for i, country in enumerate(metrics['top_countries'], 1):
            report_content += f"  {i}. {country['country']}: £{country['revenue']:,}\n"
        
        # Save report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = f"data/processed/daily_report_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        # Print to logs
        logger.info("📋 DAILY REPORT GENERATED:")
        for line in report_content.split('\n')[:20]:  # First 20 lines
            if line.strip():
                logger.info(line)
        
        return {
            'status': 'success',
            'report_file': report_file,
            'metrics_summary': {
                'revenue': metrics['total_revenue'],
                'orders': metrics['total_orders'],
                'customers': metrics['unique_customers'],
                'data_source': metrics.get('data_source', 'unknown')
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Report generation failed: {e}")
        raise
