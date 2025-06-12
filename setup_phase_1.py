# setup_phase_1.py
"""
Phase 1.4: Setup and Test Phase 1 Airflow Implementation
This script helps you set up the directory structure and test the basic DAG
"""

import os
import shutil
from pathlib import Path
import subprocess
import sys

def create_directory_structure():
    """Create the Airflow project directory structure"""
    print("📁 Creating Airflow directory structure...")
    
    directories = [
        'airflow/dags',
        'airflow/plugins/operators',
        'airflow/plugins/hooks',
        'airflow/config',
        'src/tasks',
        'src/services',
        'src/utils',
        'src/config',
        'data/raw',
        'data/processed',
        'data/ml_ready',
        'data/models',
        'tests/dags',
        'tests/tasks',
        'tests/integration',
        'scripts',
        'docker',
        'docs/airflow'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ Created: {directory}")
    
    print("✅ Directory structure created successfully")

def copy_current_files():
    """Copy existing project files to new structure"""
    print("\n📋 Organizing existing files...")
    
    # File mappings: (source, destination)
    file_mappings = [
        ('src/batch/daily_processor.py', 'src/legacy/daily_processor.py'),
        ('src/ml/train_model.py', 'src/services/ml_training_service.py'),
        ('src/ml/prediction_api.py', 'src/services/ml_prediction_api.py'),
        ('src/streaming/consumer.py', 'src/services/streaming_consumer.py'),
        ('src/streaming/publisher.py', 'src/services/streaming_publisher.py'),
        ('src/monitoring/monitor.py', 'src/services/monitoring_service.py'),
        ('data/OnlineRetail.csv', 'data/raw/OnlineRetail.csv'),
        ('requirements.txt', 'requirements.txt'),
        ('README.md', 'README.md')
    ]
    
    for source, destination in file_mappings:
        if Path(source).exists():
            Path(destination).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            print(f"   ✅ Copied: {source} -> {destination}")
        else:
            print(f"   ⚠️ Not found: {source}")
    
    print("✅ File organization completed")

def create_airflow_config():
    """Create basic Airflow configuration files"""
    print("\n⚙️ Creating Airflow configuration...")
    
    # Create airflow.cfg snippet
    airflow_config = """
# Key Airflow configuration for e-commerce analytics

[core]
dags_folder = ./airflow/dags
base_log_folder = ./logs
logging_level = INFO
executor = LocalExecutor
sql_alchemy_conn = sqlite:///./airflow.db
load_examples = False
max_active_runs_per_dag = 1

[webserver]
web_server_port = 8080
web_server_host = 0.0.0.0
secret_key = ecommerce_analytics_secret_key
expose_config = True

[scheduler]
job_heartbeat_sec = 5
scheduler_heartbeat_sec = 5
min_file_process_interval = 30
catchup_by_default = False

[logging]
remote_logging = False
"""
    
    with open('airflow/config/airflow_snippet.cfg', 'w') as f:
        f.write(airflow_config)
    
    # Create environment variables file
    env_vars = """
# Airflow Environment Variables
export AIRFLOW_HOME=$(pwd)/airflow
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/service-account-key.json"

# E-commerce specific variables
export ECOMMERCE_PROJECT_ID="ecommerce-analytics-462115"
export ECOMMERCE_DATASET_ID="ecommerce_data"
export ECOMMERCE_TABLE_ID="historical_orders"
"""
    
    with open('scripts/setup_env.sh', 'w') as f:
        f.write(env_vars)
    
    os.chmod('scripts/setup_env.sh', 0o755)
    
    print("   ✅ Created airflow/config/airflow_snippet.cfg")
    print("   ✅ Created scripts/setup_env.sh")
    
    print("✅ Airflow configuration created")

def create_docker_setup():
    """Create Docker setup for local development"""
    print("\n🐳 Creating Docker setup...")
    
    dockerfile_content = """
FROM apache/airflow:2.8.1-python3.9

# Install additional dependencies
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Copy project files
COPY airflow/dags /opt/airflow/dags
COPY src /opt/airflow/src
COPY data /opt/airflow/data

# Set environment variables
ENV PYTHONPATH="/opt/airflow/src:${PYTHONPATH}"
ENV AIRFLOW__CORE__LOAD_EXAMPLES=False
ENV AIRFLOW__CORE__MAX_ACTIVE_RUNS_PER_DAG=1
"""
    
    docker_compose_content = """
version: '3.8'

x-airflow-common:
  &airflow-common
  image: ecommerce-airflow:latest
  environment:
    &airflow-common-env
    AIRFLOW__CORE__EXECUTOR: LocalExecutor
    AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@postgres/airflow
    AIRFLOW__CORE__FERNET_KEY: ''
    AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION: 'true'
    AIRFLOW__CORE__LOAD_EXAMPLES: 'false'
    AIRFLOW__API__AUTH_BACKENDS: 'airflow.api.auth.backend.basic_auth'
    PYTHONPATH: '/opt/airflow/src'
  volumes:
    - ./airflow/dags:/opt/airflow/dags
    - ./airflow/logs:/opt/airflow/logs
    - ./airflow/plugins:/opt/airflow/plugins
    - ./src:/opt/airflow/src
    - ./data:/opt/airflow/data
  user: "50000:0"
  depends_on:
    postgres:
      condition: service_healthy

services:
  postgres:
    image: postgres:13
    environment:
      POSTGRES_USER: airflow
      POSTGRES_PASSWORD: airflow
      POSTGRES_DB: airflow
    volumes:
      - postgres-db-volume:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD", "pg_isready", "-U", "airflow"]
      interval: 5s
      retries: 5
    restart: always

  airflow-webserver:
    <<: *airflow-common
    command: webserver
    ports:
      - 8080:8080
    healthcheck:
      test: ["CMD", "curl", "--fail", "http://localhost:8080/health"]
      interval: 10s
      timeout: 10s
      retries: 5
    restart: always

  airflow-scheduler:
    <<: *airflow-common
    command: scheduler
    healthcheck:
      test: ["CMD-SHELL", 'airflow jobs check --job-type SchedulerJob --hostname "$${HOSTNAME}"']
      interval: 10s
      timeout: 10s
      retries: 5
    restart: always

  airflow-init:
    <<: *airflow-common
    entrypoint: /bin/bash
    command:
      - -c
      - |
        function ver() {
          printf "%04d%04d%04d%04d" $${1//./ }
        }
        airflow_version=$$(AIRFLOW__LOGGING__LOGGING_LEVEL=INFO && airflow version)
        airflow_version_comparable=$$(ver $${airflow_version})
        min_airflow_version=2.2.0
        min_airflow_version_comparable=$$(ver $${min_airflow_version})
        if (( airflow_version_comparable < min_airflow_version_comparable )); then
          echo -e "\\033[1;31mERROR!!!: Too old Airflow version $${airflow_version}!\\e[0m"
          exit 1
        fi
        if [[ -z "${AIRFLOW_UID}" ]]; then
          echo -e "\\033[1;33mWARNING!!!: AIRFLOW_UID not set!\\e[0m"
          echo "Setting AIRFLOW_UID to 50000"
          export AIRFLOW_UID=50000
        fi
        airflow db init
        airflow users create \\
          --username admin \\
          --firstname Admin \\
          --lastname User \\
          --role Admin \\
          --email admin@example.com \\
          --password admin
    user: "0:0"

volumes:
  postgres-db-volume:
"""
    
    with open('docker/Dockerfile', 'w') as f:
        f.write(dockerfile_content)
    
    with open('docker/docker-compose.yml', 'w') as f:
        f.write(docker_compose_content)
    
    print("   ✅ Created docker/Dockerfile")
    print("   ✅ Created docker/docker-compose.yml")
    
    print("✅ Docker setup created")

def write_artifacts_to_files():
    """Write the artifact content to actual files"""
    print("\n📝 Creating DAG and task files...")
    
    # This would contain the content from our artifacts
    # For now, we'll create placeholders that the user should replace
    
    dag_placeholder = '''# TODO: Copy the DAG content from the artifacts above
# File: airflow/dags/ecommerce_daily_analytics.py

# Copy the complete DAG code from the "Phase 1.2: Basic Daily Analytics DAG" artifact
'''
    
    tasks_placeholder = '''# TODO: Copy the task functions from the artifacts above
# File: src/tasks/data_processing_tasks.py

# Copy the complete task functions from the "Phase 1.1: Extracted Task Functions" artifact
'''
    
    test_placeholder = '''# TODO: Copy the test code from the artifacts above
# File: tests/test_phase_1_basic_dag.py

# Copy the complete test code from the "Phase 1.3: Test Basic DAG Implementation" artifact
'''
    
    with open('airflow/dags/ecommerce_daily_analytics.py', 'w') as f:
        f.write(dag_placeholder)
    
    with open('src/tasks/data_processing_tasks.py', 'w') as f:
        f.write(tasks_placeholder)
    
    with open('tests/test_phase_1_basic_dag.py', 'w') as f:
        f.write(test_placeholder)
    
    # Create __init__.py files
    init_files = [
        'src/__init__.py',
        'src/tasks/__init__.py',
        'src/services/__init__.py',
        'src/utils/__init__.py',
        'tests/__init__.py'
    ]
    
    for init_file in init_files:
        Path(init_file).touch()
    
    print("   ✅ Created placeholder files")
    print("   ⚠️ You need to copy the actual content from the artifacts above")
    
def run_phase_1_tests():
    """Run Phase 1 tests"""
    print("\n🧪 Running Phase 1 tests...")
    
    try:
        # Check if required files exist
        required_files = [
            'airflow/dags/ecommerce_daily_analytics.py',
            'src/tasks/data_processing_tasks.py'
        ]
        
        missing_files = []
        for file_path in required_files:
            if not Path(file_path).exists() or Path(file_path).stat().st_size < 100:
                missing_files.append(file_path)
        
        if missing_files:
            print("⚠️ Cannot run tests - missing or empty files:")
            for file_path in missing_files:
                print(f"   - {file_path}")
            print("\n💡 Please copy the content from the artifacts above into these files")
            return False
        
        # Run the test
        result = subprocess.run([
            sys.executable, 'tests/test_phase_1_basic_dag.py'
        ], capture_output=True, text=True)
        
        print("Test output:")
        print(result.stdout)
        
        if result.stderr:
            print("Errors:")
            print(result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        return False

def main():
    """Main setup function for Phase 1"""
    print("🚀 PHASE 1: CORE INFRASTRUCTURE SETUP")
    print("=" * 60)
    print("Setting up Airflow migration infrastructure...")
    
    steps = [
        ("Create Directory Structure", create_directory_structure),
        ("Copy Current Files", copy_current_files),
        ("Create Airflow Config", create_airflow_config),
        ("Create Docker Setup", create_docker_setup),
        ("Write Artifact Files", write_artifacts_to_files)
    ]
    
    for step_name, step_func in steps:
        print(f"\n{step_name}...")
        try:
            step_func()
        except Exception as e:
            print(f"❌ {step_name} failed: {e}")
            return False
    
    print("\n📋 PHASE 1 SETUP COMPLETE!")
    print("=" * 40)
    
    print("\n✅ What was created:")
    print("   📁 Airflow directory structure")
    print("   📄 DAG and task file templates")
    print("   ⚙️ Airflow configuration files")
    print("   🐳 Docker setup for local development")
    print("   🧪 Test files")
    
    print("\n⚠️ IMPORTANT: Manual steps required:")
    print("   1. Copy DAG content from 'Phase 1.2' artifact to airflow/dags/ecommerce_daily_analytics.py")
    print("   2. Copy task functions from 'Phase 1.1' artifact to src/tasks/data_processing_tasks.py")
    print("   3. Copy test code from 'Phase 1.3' artifact to tests/test_phase_1_basic_dag.py")
    print("   4. Update GOOGLE_APPLICATION_CREDENTIALS in scripts/setup_env.sh")
    
    print("\n🔄 Next steps:")
    print("   1. Complete the manual steps above")
    print("   2. Run: source scripts/setup_env.sh")
    print("   3. Run: python tests/test_phase_1_basic_dag.py")
    print("   4. If tests pass, proceed to Phase 2")
    
    print("\n🐳 To start with Docker:")
    print("   1. cd docker")
    print("   2. docker-compose up airflow-init")
    print("   3. docker-compose up -d")
    print("   4. Visit http://localhost:8080 (admin/admin)")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Phase 1 setup completed successfully!")
        print("Complete the manual steps and run tests before proceeding to Phase 2.")
    else:
        print("\n❌ Phase 1 setup failed. Please check the errors above.")
    
    sys.exit(0 if success else 1)
