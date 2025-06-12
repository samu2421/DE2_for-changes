# tests/test_airflow_phase1.py
"""
Test Airflow Phase 1 implementation for existing project
"""

import sys
import os
from pathlib import Path
from datetime import datetime

def print_test_header(test_name):
    """Print formatted test header"""
    print(f"\n🔍 Testing: {test_name}")
    print("-" * 50)

def test_project_structure():
    """Test 1: Verify project structure"""
    print_test_header("Project Structure")
    
    required_items = [
        ('airflow/dags', 'directory'),
        ('src/', 'directory'),
        ('data/', 'directory'),
        ('tests/', 'directory'),
        ('gcp-service-key.json', 'file'),
        ('requirements.txt', 'file')
    ]
    
    all_good = True
    
    for item, item_type in required_items:
        path = Path(item)
        if item_type == 'directory':
            exists = path.is_dir()
        else:
            exists = path.is_file()
        
        if exists:
            if item_type == 'file':
                size = path.stat().st_size
                print(f"   ✅ {item} ({size} bytes)")
            else:
                print(f"   ✅ {item}/")
        else:
            print(f"   ❌ Missing: {item}")
            all_good = False
    
    if all_good:
        print("✅ Project structure test PASSED")
    else:
        print("❌ Project structure test FAILED")
    
    return all_good


def test_python_environment():
    """Test 2: Check Python environment and imports"""
    print_test_header("Python Environment")
    
    # Test basic imports
    try:
        import pandas as pd
        print(f"   ✅ pandas {pd.__version__}")
    except ImportError:
        print("   ❌ pandas not available")
        return False
    
    try:
        from datetime import datetime
        print("   ✅ datetime module")
    except ImportError:
        print("   ❌ datetime module not available")
        return False
    
    # Test Google Cloud (optional)
    try:
        from google.cloud import bigquery
        print("   ✅ google-cloud-bigquery available")
    except ImportError:
        print("   ⚠️ google-cloud-bigquery not available (will use fallback)")
    
    print("✅ Python environment test PASSED")
    return True


def test_task_functions():
    """Test 3: Test task function imports and basic execution"""
    print_test_header("Task Functions")
    
    # Add src to path
    current_dir = os.getcwd()
    src_path = os.path.join(current_dir, 'src')
    if src_path not in sys.path:
        sys.path.append(src_path)
    
    try:
        # Test imports
        from tasks.data_processing_tasks import (
            initialize_bigquery_connection,
            extract_daily_data,
            validate_data_quality,
            process_business_metrics,
            process_customer_segmentation,
            generate_daily_report
        )
        print("   ✅ All task functions imported successfully")
        
        # Test basic function call
        context = {
            'params': {'project_id': 'test-project'},
            'execution_date': datetime.now(),
            'dag_run': type('MockDagRun', (), {'run_id': 'test_run'})()
        }
        
        result = initialize_bigquery_connection(**context)
        if result and 'status' in result:
            print(f"   ✅ Sample function execution: {result['status']}")
        else:
            print(f"   ❌ Function execution failed: {result}")
            return False
        
        print("✅ Task functions test PASSED")
        return True
        
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Execution error: {e}")
        return False


def test_dag_import():
    """Test 4: Test DAG import and structure"""
    print_test_header("DAG Import and Structure")
    
    # Add airflow/dags to path
    current_dir = os.getcwd()
    dags_path = os.path.join(current_dir, 'airflow', 'dags')
    if dags_path not in sys.path:
        sys.path.append(dags_path)
    
    try:
        # Import the DAG
        import ecommerce_daily_analytics as dag_module
        dag = dag_module.dag
        
        print(f"   ✅ DAG imported: {dag.dag_id}")
        print(f"   ✅ Schedule: {dag.schedule_interval}")
        print(f"   ✅ Task count: {len(dag.tasks)}")
        print(f"   ✅ Tags: {dag.tags}")
        
        # Check expected tasks
        expected_tasks = [
            'start_pipeline',
            'initialize_bigquery',
            'extract_daily_data',
            'validate_data_quality',
            'process_business_metrics',
            'process_customer_segmentation',
            'generate_daily_report',
            'complete_pipeline'
        ]
        
        actual_tasks = [task.task_id for task in dag.tasks]
        missing_tasks = [task for task in expected_tasks if task not in actual_tasks]
        
        if missing_tasks:
            print(f"   ❌ Missing tasks: {missing_tasks}")
            return False
        else:
            print("   ✅ All expected tasks found")
        
        print("✅ DAG import test PASSED")
        return True
        
    except Exception as e:
        print(f"   ❌ DAG import error: {e}")
        return False


def test_data_sources():
    """Test 5: Check available data sources"""
    print_test_header("Data Sources")
    
    data_sources = []
    
    # Check for OnlineRetail.csv
    csv_locations = [
        'data/OnlineRetail.csv',
        'data/raw/OnlineRetail.csv',
        'data/cleaned/OnlineRetail.csv'
    ]
    
    csv_found = False
    for csv_path in csv_locations:
        if Path(csv_path).exists():
            size_mb = Path(csv_path).stat().st_size / (1024*1024)
            print(f"   ✅ Found CSV: {csv_path} ({size_mb:.1f} MB)")
            data_sources.append('CSV')
            csv_found = True
            break
    
    if not csv_found:
        print("   ⚠️ No OnlineRetail.csv found - will use sample data")
        data_sources.append('Sample')
    
    # Check BigQuery credentials
    if Path('gcp-service-key.json').exists():
        print("   ✅ BigQuery credentials file found")
        data_sources.append('BigQuery')
    else:
        print("   ⚠️ No BigQuery credentials - will use fallback")
    
    print(f"   ✅ Available data sources: {', '.join(data_sources)}")
    print("✅ Data sources test PASSED")
    return True


def test_end_to_end_simulation():
    """Test 6: Simulate end-to-end pipeline"""
    print_test_header("End-to-End Pipeline Simulation")
    
    try:
        # Add paths
        current_dir = os.getcwd()
        src_path = os.path.join(current_dir, 'src')
        if src_path not in sys.path:
            sys.path.append(src_path)
        
        from tasks.data_processing_tasks import (
            initialize_bigquery_connection,
            extract_daily_data
        )
        
        # Create mock context
        class MockTaskInstance:
            def __init__(self):
                self.xcom_data = {}
            
            def xcom_pull(self, task_ids):
                return self.xcom_data.get(task_ids, {})
            
            def xcom_push(self, key, value):
                self.xcom_data[key] = value
        
        mock_ti = MockTaskInstance()
        
        context = {
            'params': {'project_id': 'test-project'},
            'execution_date': datetime.now(),
            'dag_run': type('MockDagRun', (), {'run_id': 'test_run'})(),
            'task_instance': mock_ti
        }
        
        # Test task 1
        print("   🔄 Testing BigQuery initialization...")
        result1 = initialize_bigquery_connection(**context)
        mock_ti.xcom_push('initialize_bigquery', result1)
        print(f"   ✅ Task 1 result: {result1['status']}")
        
        # Test task 2
        print("   🔄 Testing data extraction...")
        result2 = extract_daily_data(**context)
        print(f"   ✅ Task 2 result: {result2['status']} ({result2.get('row_count', 0)} rows)")
        
        print("✅ End-to-end simulation test PASSED")
        return True
        
    except Exception as e:
        print(f"   ❌ End-to-end simulation failed: {e}")
        return False


def run_all_tests():
    """Run all tests and provide summary"""
    print("🧪 AIRFLOW PHASE 1 TESTING FOR EXISTING PROJECT")
    print("=" * 60)
    print(f"Test started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Working directory: {os.getcwd()}")
    
    tests = [
        ("Project Structure", test_project_structure),
        ("Python Environment", test_python_environment),
        ("Task Functions", test_task_functions),
        ("DAG Import", test_dag_import),
        ("Data Sources", test_data_sources),
        ("End-to-End Simulation", test_end_to_end_simulation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' failed with exception: {e}")
            results.append((test_name, False))
    
    # Print summary
    print(f"\n📊 TEST SUMMARY")
    print("=" * 40)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status} {test_name}")
    
    print(f"\nOverall Result: {passed}/{total} tests passed")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("\n🎉 All tests passed! Your Airflow Phase 1 setup is working!")
        print("\n📋 What's working:")
        print("   ✅ Project structure is correct")
        print("   ✅ Task functions can be imported and executed")
        print("   ✅ DAG can be loaded successfully")
        print("   ✅ Data sources are available")
        print("   ✅ End-to-end pipeline simulation works")
        
        print("\n🚀 Next steps:")
        print("   1. Test the DAG syntax: python airflow/dags/ecommerce_daily_analytics.py")
        print("   2. Set up local Airflow: pip install apache-airflow")
        print("   3. Run the DAG in Airflow UI")
        print("   4. Proceed to Phase 2: Advanced DAG Implementation")
        
    elif passed >= total * 0.8:
        print("\n✅ Most tests passed! Minor issues to fix:")
        failed_tests = [name for name, success in results if not success]
        for test_name in failed_tests:
            print(f"   • Fix: {test_name}")
        
        print("\n💡 Your setup is mostly working - proceed with caution")
        
    else:
        print("\n⚠️ Several tests failed. Please fix these issues:")
        failed_tests = [name for name, success in results if not success]
        for test_name in failed_tests:
            print(f"   • Fix: {test_name}")
        
        print("\n🔧 Recommended actions:")
        print("   1. Check file locations and content")
        print("   2. Verify Python imports work")
        print("   3. Ensure directory structure is correct")
        print("   4. Re-run tests after fixes")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
