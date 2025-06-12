"""
Simplified E-Commerce Analytics Pipeline
Start with basic functionality and build up
"""

import os
import sys
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

# Add paths for imports
sys.path.append('/opt/airflow')
sys.path.append('/opt/airflow/src')

# Default arguments
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
    'ecommerce_simple_pipeline',
    default_args=default_args,
    description='Simplified E-Commerce Analytics Pipeline',
    schedule_interval='@daily',
    max_active_runs=1,
    catchup=False,
    tags=['ecommerce', 'simple', 'test']
)

def test_imports_task(**context):
    """Test if we can import our modules"""
    try:
        print("🧪 Testing basic imports...")
        
        # Test basic packages
        import pandas as pd
        import numpy as np
        import json
        from datetime import datetime
        from pathlib import Path
        
        print(f"✅ Pandas version: {pd.__version__}")
        print(f"✅ Numpy version: {np.__version__}")
        
        # Test file system access
        Path('/opt/airflow/data').mkdir(exist_ok=True)
        print("✅ Data directory accessible")
        
        # Try to import your modules
        try:
            # Import without the complex dependencies first
            print("🔍 Testing src imports...")
            print(f"Python path: {sys.path}")
            
            # List what's in src
            import os
            if os.path.exists('/opt/airflow/src'):
                src_files = os.listdir('/opt/airflow/src')
                print(f"Files in src: {src_files}")
            else:
                print("❌ /opt/airflow/src directory not found")
            
        except Exception as e:
            print(f"⚠️ Src import issue: {e}")
        
        return "Import test completed"
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        raise

def generate_simple_data_task(**context):
    """Generate simple test data without complex dependencies"""
    try:
        import pandas as pd
        import json
        from datetime import datetime
        from pathlib import Path
        
        print("🏭 Generating simple test data...")
        
        # Create simple fake data
        orders = []
        for i in range(10):
            order = {
                'InvoiceNo': f'INV{i:03d}',
                'StockCode': f'PROD{i % 3}',
                'Description': f'Product {i % 3}',
                'Quantity': i + 1,
                'UnitPrice': 10.0 + (i * 2),
                'InvoiceDate': datetime.now().isoformat(),
                'CustomerID': 1000 + i,
                'Country': ['UK', 'Germany', 'France'][i % 3]
            }
            orders.append(order)
        
        # Save data
        Path('/opt/airflow/data').mkdir(exist_ok=True)
        with open('/opt/airflow/data/simple_orders.json', 'w') as f:
            json.dump(orders, f, indent=2)
        
        print(f"✅ Generated {len(orders)} test orders")
        print(f"📁 Saved to /opt/airflow/data/simple_orders.json")
        
        return {'orders_generated': len(orders)}
        
    except Exception as e:
        print(f"❌ Data generation failed: {e}")
        raise

def process_simple_data_task(**context):
    """Process the simple data"""
    try:
        import pandas as pd
        import json
        
        print("📊 Processing simple data...")
        
        # Load the data
        with open('/opt/airflow/data/simple_orders.json', 'r') as f:
            orders = json.load(f)
        
        # Convert to DataFrame
        df = pd.DataFrame(orders)
        
        # Add calculated fields
        df['Revenue'] = df['Quantity'] * df['UnitPrice']
        
        # Simple metrics
        metrics = {
            'total_orders': len(df),
            'total_revenue': df['Revenue'].sum(),
            'avg_order_value': df['Revenue'].mean(),
            'unique_customers': df['CustomerID'].nunique(),
            'unique_products': df['StockCode'].nunique()
        }
        
        print(f"📈 Processed metrics:")
        for key, value in metrics.items():
            print(f"   {key}: {value}")
        
        # Save processed data
        df.to_csv('/opt/airflow/data/processed_orders.csv', index=False)
        
        with open('/opt/airflow/data/simple_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print("✅ Data processing completed")
        return metrics
        
    except Exception as e:
        print(f"❌ Data processing failed: {e}")
        raise

def simple_model_task(**context):
    """Simple model training"""
    try:
        import pandas as pd
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        import joblib
        from pathlib import Path
        
        print("🤖 Training simple model...")
        
        # Load processed data
        df = pd.read_csv('/opt/airflow/data/processed_orders.csv')
        
        # Simple features and target
        X = df[['Quantity', 'UnitPrice']].values
        y = df['Revenue'].values
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train simple model
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Test score
        score = model.score(X_test, y_test)
        
        # Save model
        Path('/opt/airflow/src/ml').mkdir(parents=True, exist_ok=True)
        joblib.dump(model, '/opt/airflow/src/ml/simple_model.pkl')
        
        print(f"✅ Model trained with R² score: {score:.4f}")
        print("📁 Model saved to /opt/airflow/src/ml/simple_model.pkl")
        
        return {'model_score': score}
        
    except Exception as e:
        print(f"❌ Model training failed: {e}")
        raise

def monitor_simple_results_task(**context):
    """Monitor the simple pipeline results"""
    try:
        import json
        from pathlib import Path
        
        print("🖥️ Monitoring pipeline results...")
        
        # Get results from previous tasks
        task_instance = context['task_instance']
        data_gen_results = task_instance.xcom_pull(task_ids='generate_simple_data')
        processing_results = task_instance.xcom_pull(task_ids='process_simple_data')
        model_results = task_instance.xcom_pull(task_ids='simple_model')
        
        # Check files exist
        files_check = {
            'simple_orders.json': Path('/opt/airflow/data/simple_orders.json').exists(),
            'processed_orders.csv': Path('/opt/airflow/data/processed_orders.csv').exists(),
            'simple_metrics.json': Path('/opt/airflow/data/simple_metrics.json').exists(),
            'simple_model.pkl': Path('/opt/airflow/src/ml/simple_model.pkl').exists()
        }
        
        # Create monitoring report
        report = {
            'pipeline_timestamp': datetime.now().isoformat(),
            'task_results': {
                'data_generation': data_gen_results,
                'data_processing': processing_results,
                'model_training': model_results
            },
            'files_created': files_check,
            'overall_status': 'success' if all(files_check.values()) else 'partial'
        }
        
        # Save report
        with open('/opt/airflow/data/pipeline_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print("=" * 50)
        print("🎉 SIMPLE PIPELINE SUMMARY")
        print("=" * 50)
        
        if data_gen_results:
            print(f"📊 Data Generation: {data_gen_results.get('orders_generated', 0)} orders")
        
        if processing_results:
            print(f"💰 Total Revenue: £{processing_results.get('total_revenue', 0):.2f}")
            print(f"🛍️ Total Orders: {processing_results.get('total_orders', 0)}")
        
        if model_results:
            print(f"🤖 Model Score: {model_results.get('model_score', 0):.4f}")
        
        files_created = sum(files_check.values())
        print(f"📁 Files Created: {files_created}/4")
        print(f"🏁 Status: {report['overall_status'].upper()}")
        
        return report
        
    except Exception as e:
        print(f"❌ Monitoring failed: {e}")
        raise

# Define tasks
test_imports = PythonOperator(
    task_id='test_imports',
    python_callable=test_imports_task,
    dag=dag,
)

generate_simple_data = PythonOperator(
    task_id='generate_simple_data',
    python_callable=generate_simple_data_task,
    dag=dag,
)

process_simple_data = PythonOperator(
    task_id='process_simple_data',
    python_callable=process_simple_data_task,
    dag=dag,
)

simple_model = PythonOperator(
    task_id='simple_model',
    python_callable=simple_model_task,
    dag=dag,
)

monitor_simple_results = PythonOperator(
    task_id='monitor_simple_results',
    python_callable=monitor_simple_results_task,
    dag=dag,
)

# Set task dependencies
test_imports >> generate_simple_data >> process_simple_data >> simple_model >> monitor_simple_results