"""
E-Commerce Analytics Data Pipeline DAG
Apache Airflow 2.8.1 Compatible

This DAG orchestrates the complete data pipeline:
1. Generate fake data
2. Clean the data
3. Batch process data
4. Train ML model
5. Monitor results
"""

import os
import sys
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from simple_littles_law import order_arrives, order_completes, show_metrics

# Add the project root to Python path
import sys
import os
sys.path.append('/opt/airflow')
sys.path.append('/opt/airflow/src')

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Default arguments for the DAG
default_args = {
    'owner': 'data-engineering-team',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Create the DAG
dag = DAG(
    'ecommerce_data_pipeline',
    default_args=default_args,
    description='Daily E-Commerce Analytics Data Pipeline',
    schedule_interval='@daily',  # Run daily
    max_active_runs=1,
    catchup=False,
    tags=['ecommerce', 'analytics', 'ml', 'data-engineering']
)

def generate_fake_data_task(**context):
    """Task 1: Generate fake e-commerce data"""
    try:
        from src.fake_data_generator import EcommerceFakeDataGenerator
        import json
        from pathlib import Path
        
        print("🏭 Starting fake data generation...")

        pipeline_id = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        order_arrives(pipeline_id)
        print(f"📊 Little's Law: Started tracking pipeline {pipeline_id}")
        
        # Initialize generator
        generator = EcommerceFakeDataGenerator()
        
        # Generate training data (larger batch)
        training_orders = generator.generate_batch_orders(
            num_orders=1000,
            order_mix={'random': 0.6, 'high_value': 0.2, 'bulk': 0.1, 'single': 0.1}
        )
        
        # Generate streaming test data
        streaming_orders = generator.generate_batch_orders(
            num_orders=50,
            order_mix={'random': 0.8, 'high_value': 0.2}
        )
        
        # Ensure data directory exists
        Path('/opt/airflow/data').mkdir(exist_ok=True)
        
        # Save generated data
        with open('/opt/airflow/data/training_orders.json', 'w') as f:
            json.dump(training_orders, f, indent=2, default=str)
        
        with open('/opt/airflow/data/streaming_orders.json', 'w') as f:
            json.dump(streaming_orders, f, indent=2, default=str)
        
        # Get summary
        summary = generator.get_data_quality_summary(training_orders)
        
        print(f"✅ Generated {len(training_orders)} training orders")
        print(f"✅ Generated {len(streaming_orders)} streaming orders")
        print(f"📊 Total revenue: £{summary['total_revenue']:.2f}")
        
        return {
            'pipeline_id': pipeline_id,  # ADD THIS LINE
            'training_orders_count': len(training_orders),
            'streaming_orders_count': len(streaming_orders),
            'total_revenue': summary['total_revenue']
        }
        
    except Exception as e:
        print(f"❌ Error in fake data generation: {e}")
        raise


def clean_data_task(**context):
    """Task 2: Clean the generated data - FIXED VERSION"""
    try:
        from src.data_cleaning_utility import EcommerceDataCleaner
        import pandas as pd
        import json
        from pathlib import Path
        
        print("🧹 Starting data cleaning...")

        gen_results = context['task_instance'].xcom_pull(task_ids='generate_fake_data')
        pipeline_id = gen_results['pipeline_id']
        
        # Load generated data
        with open('/opt/airflow/data/training_orders.json', 'r') as f:
            orders = json.load(f)
        
        # Convert to DataFrame for cleaning
        df = pd.DataFrame(orders)
        
        # Basic data cleaning
        print(f"Initial data shape: {df.shape}")
        
        # Remove rows with missing critical values
        critical_cols = ['CustomerID', 'InvoiceDate', 'Quantity', 'UnitPrice']
        df = df.dropna(subset=critical_cols)
        
        # Remove negative quantities and prices
        df = df[df['Quantity'] > 0]
        df = df[df['UnitPrice'] > 0]
        
        # Convert InvoiceDate to datetime
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        
        # Create calculated revenue column
        df['Revenue'] = df['Quantity'] * df['UnitPrice']
        
        # Create time-based features for ML
        df['Hour'] = df['InvoiceDate'].dt.hour
        df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek
        df['Month'] = df['InvoiceDate'].dt.month
        df['Year'] = df['InvoiceDate'].dt.year
        df['Date'] = df['InvoiceDate'].dt.date
        
        # Create derived features that will be needed for ML
        df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
        df['IsBusinessHours'] = ((df['Hour'] >= 9) & (df['Hour'] <= 17)).astype(int)
        
        # Clean and standardize text fields
        if 'Description' in df.columns:
            df['Description'] = df['Description'].fillna('Unknown Product')
            df['Description'] = df['Description'].str.strip().str.upper()
        
        if 'Country' in df.columns:
            df['Country'] = df['Country'].str.strip()
            # Standardize common country names
            country_mapping = {
                'UNITED KINGDOM': 'United Kingdom',
                'UK': 'United Kingdom',
                'USA': 'United States',
                'US': 'United States'
            }
            df['Country'] = df['Country'].replace(country_mapping)
        
        # Create price and quantity categories for ML
        df['PriceCategory'] = pd.cut(df['UnitPrice'], 
                                   bins=[0, 2, 5, 15, float('inf')], 
                                   labels=['Low', 'Medium', 'High', 'Premium'])
        
        df['QuantityCategory'] = pd.cut(df['Quantity'], 
                                      bins=[0, 1, 3, 10, float('inf')],
                                      labels=['Single', 'Few', 'Multiple', 'Bulk'])
        
        print(f"Cleaned data shape: {df.shape}")
        print(f"Columns created: {list(df.columns)}")
        
        # Ensure cleaned directory exists
        Path('/opt/airflow/data/cleaned').mkdir(parents=True, exist_ok=True)
        
        # Save cleaned data
        df.to_csv('/opt/airflow/data/cleaned/ml_ready_data.csv', index=False)
        
        # Generate and save cleaning report
        cleaning_summary = {
            'initial_rows': len(orders),
            'final_rows': len(df),
            'retention_rate': len(df) / len(orders) * 100 if len(orders) > 0 else 0,
            'columns_created': list(df.columns),
            'revenue_range': {
                'min': float(df['Revenue'].min()),
                'max': float(df['Revenue'].max()),
                'mean': float(df['Revenue'].mean())
            },
            'cleaning_steps': [
                'Removed rows with missing critical values',
                'Removed negative quantities and prices',
                'Created time-based features',
                'Created derived ML features',
                'Standardized text fields',
                'Created categorical features'
            ]
        }
        
        with open('/opt/airflow/data/cleaned/cleaning_report.json', 'w') as f:
            json.dump(cleaning_summary, f, indent=2, default=str)
        
        print(f"✅ Cleaned data: {len(df)} rows retained")
        print(f"📁 Saved to: /opt/airflow/data/cleaned/ml_ready_data.csv")
        print(f"📊 Revenue range: £{df['Revenue'].min():.2f} - £{df['Revenue'].max():.2f}")
        
        return {
            'pipeline_id': pipeline_id,
            'cleaned_rows': len(df),
            'retention_rate': len(df) / len(orders) * 100 if len(orders) > 0 else 0,
            'columns_created': list(df.columns)
        }
        
    except Exception as e:
        print(f"❌ Error in data cleaning: {e}")
        import traceback
        traceback.print_exc()
        raise


def batch_process_data_task(**context):
    """Task 3: Batch process the cleaned data"""
    try:
        import pandas as pd
        import numpy as np
        from datetime import datetime
        import json
        from pathlib import Path
        
        print("📊 Starting batch processing...")
        
        # Load cleaned data
        df = pd.read_csv('/opt/airflow/data/cleaned/ml_ready_data.csv')
        
        # Convert InvoiceDate to datetime
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        df['Date'] = df['InvoiceDate'].dt.date
        
        # Process daily metrics for the most recent date
        target_date = df['Date'].max()
        daily_data = df[df['Date'] == target_date]
        
        if daily_data.empty:
            target_date = df['Date'].iloc[0]  # Fallback to first date
            daily_data = df[df['Date'] == target_date]
        
        # Calculate metrics
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
        
        # Top performers
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
        
        # Ensure processed directory exists
        Path('/opt/airflow/data/processed').mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(f'/opt/airflow/data/processed/daily_metrics_{timestamp}.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"✅ Processed {metrics['total_orders']} orders")
        print(f"💰 Total revenue: £{metrics['total_revenue']}")
        
        return metrics
        
    except Exception as e:
        print(f"❌ Error in batch processing: {e}")
        raise


def train_model_task(**context):
    """Task 4: Train the ML model - FIXED VERSION"""
    try:
        import pandas as pd
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        from sklearn.metrics import mean_squared_error, r2_score
        import joblib
        import pickle
        import numpy as np
        from pathlib import Path
        
        print("🤖 Starting ML model training...")
        
        # Load cleaned data
        df = pd.read_csv('/opt/airflow/data/cleaned/ml_ready_data.csv')
        
        # Convert InvoiceDate to datetime and create time-based features
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        df['Hour'] = df['InvoiceDate'].dt.hour
        df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek
        df['Month'] = df['InvoiceDate'].dt.month
        
        # Create the missing features that were causing the error
        df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
        df['IsBusinessHours'] = ((df['Hour'] >= 9) & (df['Hour'] <= 17)).astype(int)
        
        # Create price and quantity categories
        df['PriceCategory'] = pd.cut(df['UnitPrice'], 
                                   bins=[0, 2, 5, 15, float('inf')], 
                                   labels=['Low', 'Medium', 'High', 'Premium'])
        
        df['QuantityCategory'] = pd.cut(df['Quantity'], 
                                      bins=[0, 1, 3, 10, float('inf')],
                                      labels=['Single', 'Few', 'Multiple', 'Bulk'])
        
        # Define numerical features (these should all exist now)
        numerical_features = ['Quantity', 'UnitPrice', 'Hour', 'DayOfWeek', 'Month', 
                             'IsWeekend', 'IsBusinessHours']
        
        # Check which numerical features actually exist in the DataFrame
        available_numerical = [col for col in numerical_features if col in df.columns]
        print(f"Available numerical features: {available_numerical}")
        
        # Use only available numerical features
        X = df[available_numerical].copy()
        
        # Define categorical features
        categorical_features = ['Country', 'PriceCategory', 'QuantityCategory']
        
        # Encode categorical features
        label_encoders = {}
        for cat_feature in categorical_features:
            if cat_feature in df.columns:
                # Convert to string and handle missing values
                cat_data = df[cat_feature].astype(str).fillna('Unknown')
                
                # Encode
                le = LabelEncoder()
                X[f'{cat_feature}_encoded'] = le.fit_transform(cat_data)
                label_encoders[cat_feature] = le
                print(f"Encoded {cat_feature}: {len(le.classes_)} unique values")
        
        # Target variable
        y = df['Revenue']
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target variable shape: {y.shape}")
        print(f"Features: {list(X.columns)}")
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model (set n_jobs=1 to avoid multiprocessing issues in Airflow)
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=1  # Important: Set to 1 for Airflow compatibility
        )
        
        print("Training model...")
        model.fit(X_train, y_train)
        
        # Evaluate model
        test_predictions = model.predict(X_test)
        test_mse = mean_squared_error(y_test, test_predictions)
        test_r2 = r2_score(y_test, test_predictions)
        
        print(f"Model training completed!")
        print(f"R² Score: {test_r2:.4f}")
        print(f"RMSE: £{np.sqrt(test_mse):.2f}")
        
        # Ensure data directory exists for saving models (writable directory)
        Path('/opt/airflow/data/models').mkdir(parents=True, exist_ok=True)
        
        # Save model and artifacts to writable directory
        joblib.dump(model, '/opt/airflow/data/models/revenue_model.pkl')
        print("✅ Model saved to /opt/airflow/data/models/revenue_model.pkl")
        
        with open('/opt/airflow/data/models/label_encoders.pkl', 'wb') as f:
            pickle.dump(label_encoders, f)
        print("✅ Label encoders saved")
        
        with open('/opt/airflow/data/models/feature_names.txt', 'w') as f:
            f.write('\n'.join(X.columns))
        print("✅ Feature names saved")
        
        performance = {
            'r2_score': float(test_r2),
            'rmse': float(np.sqrt(test_mse)),
            'feature_count': len(X.columns),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'features_used': list(X.columns)
        }
        
        print(f"✅ Model trained successfully")
        print(f"📊 R² Score: {test_r2:.4f}")
        print(f"📊 RMSE: £{np.sqrt(test_mse):.2f}")
        print(f"📊 Features used: {len(X.columns)}")
        
        return performance
        
    except Exception as e:
        print(f"❌ Error in model training: {e}")
        import traceback
        traceback.print_exc()
        raise


def monitor_results_task(**context):
    """Task 5: Monitor pipeline results and generate summary"""
    try:
        import json
        import glob
        from pathlib import Path
        from datetime import datetime
        
        print("🖥️ Starting pipeline monitoring...")

        results = context['task_instance'].xcom_pull(task_ids='generate_fake_data')
        pipeline_id = results['pipeline_id']
        order_completes(pipeline_id)
        
        # Collect results from previous tasks
        task_instance = context['task_instance']
        
        # Get results from previous tasks using XCom
        fake_data_results = task_instance.xcom_pull(task_ids='generate_fake_data')
        cleaning_results = task_instance.xcom_pull(task_ids='clean_data')
        batch_results = task_instance.xcom_pull(task_ids='batch_process_data')
        ml_results = task_instance.xcom_pull(task_ids='train_model')
        
        # Check file system for generated files
        data_files = {
            'training_data': Path('/opt/airflow/data/training_orders.json').exists(),
            'cleaned_data': Path('/opt/airflow/data/cleaned/ml_ready_data.csv').exists(),
            'ml_model': Path('/opt/airflow/data/models/revenue_model.pkl').exists(),
            'processed_metrics': len(list(Path('/opt/airflow/data/processed').glob('*.json'))) if Path('/opt/airflow/data/processed').exists() else 0
        }
        
        # Generate monitoring report
        monitoring_report = {
            'pipeline_run_timestamp': datetime.now().isoformat(),
            'task_results': {
                'fake_data_generation': fake_data_results or {'status': 'failed'},
                'data_cleaning': cleaning_results or {'status': 'failed'},
                'batch_processing': batch_results or {'status': 'failed'},
                'ml_training': ml_results or {'status': 'failed'}
            },
            'file_system_check': data_files,
            'pipeline_health': {
                'total_tasks': 5,
                'successful_tasks': sum([1 for result in [fake_data_results, cleaning_results, batch_results, ml_results] if result]),
                'data_files_created': sum([1 for exists in data_files.values() if exists]),
                'overall_status': 'success' if all([fake_data_results, cleaning_results, batch_results, ml_results]) else 'partial_success'
            }
        }
        
        # Ensure monitoring directory exists
        Path('/opt/airflow/data/monitoring').mkdir(parents=True, exist_ok=True)
        
        # Save monitoring report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(f'/opt/airflow/data/monitoring/pipeline_report_{timestamp}.json', 'w') as f:
            json.dump(monitoring_report, f, indent=2)
        
        # Print summary
        print("=" * 60)
        print("🎉 PIPELINE EXECUTION SUMMARY")
        print("=" * 60)
        
        if fake_data_results:
            print(f"✅ Data Generation: {fake_data_results.get('training_orders_count', 0)} orders")
        
        if cleaning_results:
            print(f"✅ Data Cleaning: {cleaning_results.get('retention_rate', 0):.1f}% retention")
        
        if batch_results:
            print(f"✅ Batch Processing: £{batch_results.get('total_revenue', 0):,.2f} revenue")
        
        if ml_results:
            print(f"✅ ML Training: R² = {ml_results.get('r2_score', 0):.4f}")
        
        print(f"📁 Files Created: {sum(data_files.values())}/4")
        print(f"🏁 Overall Status: {monitoring_report['pipeline_health']['overall_status'].upper()}")
        print("\n📊 Little's Law Analysis for Pipeline:")
        show_metrics()
        
        print("=" * 60)

        return {
            'pipeline_id': pipeline_id,
            'completion_status': 'success',
            'monitoring_report': monitoring_report,
            'pipeline_health': monitoring_report['pipeline_health']
        }
        
        
    except Exception as e:
        print(f"❌ Error in monitoring: {e}")
        raise


# Define tasks
generate_fake_data = PythonOperator(
    task_id='generate_fake_data',
    python_callable=generate_fake_data_task,
    dag=dag,
)

clean_data = PythonOperator(
    task_id='clean_data',
    python_callable=clean_data_task,
    dag=dag,
)

batch_process_data = PythonOperator(
    task_id='batch_process_data',
    python_callable=batch_process_data_task,
    dag=dag,
)

train_model = PythonOperator(
    task_id='train_model',
    python_callable=train_model_task,
    dag=dag,
)

monitor_results = PythonOperator(
    task_id='monitor_results',
    python_callable=monitor_results_task,
    dag=dag,
)

# Set task dependencies (sequential pipeline)
generate_fake_data >> clean_data >> batch_process_data >> train_model >> monitor_results