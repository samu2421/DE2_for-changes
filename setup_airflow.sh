#!/bin/bash

# E-Commerce Analytics Airflow Setup Script
# This script prepares the environment for running Airflow with Docker

echo "🚀 Setting up E-Commerce Analytics Airflow Environment"
echo "========================================================"

# Create airflow_dags directory if it doesn't exist
if [ ! -d "airflow_dags" ]; then
    echo "📁 Creating airflow_dags directory..."
    mkdir -p airflow_dags
    echo "✅ airflow_dags directory created"
else
    echo "✅ airflow_dags directory already exists"
fi

# Create data directories
echo "📁 Creating data directories..."
mkdir -p data/{processed,monitoring,cleaned,ml_ready}
echo "✅ Data directories created"

# Create logs directory
echo "📁 Creating logs directory..."
mkdir -p logs
echo "✅ Logs directory created"

# Set up environment variables for Airflow
echo "🔧 Setting up environment variables..."

# Check if .env file exists, create if not
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file..."
    cat > .env << EOF
# Airflow environment variables
AIRFLOW_UID=50000
AIRFLOW_GID=0
_AIRFLOW_WWW_USER_USERNAME=airflow
_AIRFLOW_WWW_USER_PASSWORD=airflow

# Project settings
PYTHONPATH=/opt/airflow/src:/opt/airflow/dags
EOF
    echo "✅ .env file created"
else
    echo "✅ .env file already exists"
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "   Visit: https://docs.docker.com/get-docker/"
    exit 1
else
    echo "✅ Docker is installed"
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    echo "   Visit: https://docs.docker.com/compose/install/"
    exit 1
else
    echo "✅ Docker Compose is installed"
fi

# Check Docker daemon
if ! docker info &> /dev/null; then
    echo "❌ Docker daemon is not running. Please start Docker first."
    exit 1
else
    echo "✅ Docker daemon is running"
fi

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "   1. Build and start the Airflow environment:"
echo "      docker-compose up --build -d"
echo ""
echo "   2. Wait for all services to be healthy (this may take 2-3 minutes)"
echo ""
echo "   3. Access Airflow Web UI:"
echo "      http://localhost:8080"
echo "      Username: airflow"
echo "      Password: airflow"
echo ""
echo "   4. Your DAG 'ecommerce_data_pipeline' should appear in the UI"
echo ""
echo "   5. To stop the environment:"
echo "      docker-compose down"
echo ""
echo "   6. To view logs:"
echo "      docker-compose logs -f"
echo ""
echo "📊 Project Structure:"
echo "   ├── airflow_dags/          # Airflow DAG definitions"
echo "   ├── src/                   # Your source code"
echo "   ├── data/                  # Data files and outputs"
echo "   ├── logs/                  # Airflow logs"
echo "   ├── docker-compose.yaml    # Docker services configuration"
echo "   ├── Dockerfile             # Airflow image definition"
echo "   └── requirements.txt       # Python dependencies"
echo ""
echo "🔍 Troubleshooting:"
echo "   • If services fail to start, check: docker-compose logs"
echo "   • For permission issues on Linux: sudo chown -R \$USER:docker ."
echo "   • To reset everything: docker-compose down -v && docker system prune"
echo ""
echo "Happy data engineering! 🚀"