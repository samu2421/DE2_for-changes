# E-Commerce Analytics Data Pipeline project

A comprehensive, production-ready data analytics platform for e-commerce businesses, featuring real-time streaming, machine learning predictions, and automated data processing pipelines.

## 🚀 Overview

This project implements a complete data engineering solution for e-commerce analytics, combining batch processing, streaming data, machine learning, and system monitoring. Built with Apache Airflow, Docker, and Google Cloud Platform, it provides scalable analytics for online retail businesses.

### Key Capabilities

- **📊 Data Pipeline Orchestration**: Automated daily processing using Apache Airflow
- **🤖 ML-Powered Revenue Prediction**: Real-time revenue forecasting using Random Forest models
- **🌊 Streaming Data Processing**: Real-time order processing with Google Cloud Pub/Sub
- **📈 Business Intelligence**: Automated daily/weekly reports and customer segmentation
- **🔍 Data Quality Management**: Comprehensive data cleaning and validation
- **📡 System Monitoring**: Health checks, performance metrics, and alerting
- **☁️ Cloud Integration**: BigQuery for large-scale data processing

## ✨ Features

### Data Processing
- **Batch Processing**: Daily metrics calculation, customer segmentation, product analysis
- **Streaming Pipeline**: Real-time order processing and ML prediction integration
- **Data Cleaning**: Automated data quality checks and preprocessing
- **Fake Data Generation**: Realistic test data generation for development and testing

### Machine Learning
- **Revenue Prediction**: ML model for predicting order values
- **Feature Engineering**: Automated feature creation from raw e-commerce data
- **Model Training**: Automated retraining with BigQuery integration
- **Prediction API**: REST API for real-time ML predictions

### Infrastructure
- **Docker Containerization**: Complete containerized deployment
- **Apache Airflow**: Workflow orchestration and scheduling
- **Google Cloud Integration**: BigQuery, Pub/Sub, and Cloud Storage
- **Monitoring Dashboard**: System health and performance monitoring

## 🛠️ Prerequisites

### Required Software
- **Docker** (v20.10+) and **Docker Compose** (v2.0+)
- **Python** 3.9+
- **Git**

### Required Accounts/Services
- **Google Cloud Platform** account (for BigQuery and Pub/Sub)
- **Kaggle** account (for dataset download)

### Hardware Requirements
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB free space
- **CPU**: 4+ cores recommended

## 📦 Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd ecommerce-analytics-pipeline
```

### 2. Download Dataset
Download the Online Retail dataset from Kaggle:
```bash
# Visit: https://www.kaggle.com/datasets/vijayuv/onlineretail
# Download OnlineRetail.csv and place it in the data/ directory
mkdir -p data
# Place OnlineRetail.csv in data/OnlineRetail.csv
```

### 3. Set Up Environment
```bash
# Run the setup script
chmod +x setup_airflow.sh
./setup_airflow.sh

# Or set up manually:
cp .env.template .env
mkdir -p data/{processed,monitoring,cleaned,ml_ready}
mkdir -p logs
```

### 4. Configure Google Cloud (Optional)
```bash
# Set up GCP credentials for BigQuery and Pub/Sub
export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/service-account-key.json"

# Update .env file with your GCP project ID
echo "GOOGLE_CLOUD_PROJECT=your-project-id" >> .env
```

### 5. Install Python Dependencies
```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### Option 1: Docker Deployment (Recommended)
```bash
# Build and start all services
docker-compose up --build -d

# Wait for services to initialize (2-3 minutes)
docker-compose logs -f

# Access Airflow Web UI
# Navigate to: http://localhost:8080
# Username: airflow
# Password: airflow
```

### Option 2: Local Development
```bash
# 1. Test the complete system
python test_complete_system.py

# 2. Train ML model
python src/ml/train_model.py

# 3. Start ML API
python src/ml/prediction_api.py

# 4. Run batch processing
python src/batch/daily_processor.py

# 5. Start monitoring
python src/monitoring/monitor.py
```

## 📋 Usage

### Running Data Pipeline

#### Using Airflow (Docker)
1. Access Airflow UI at `http://localhost:8080`
2. Enable the `ecommerce_data_pipeline` DAG
3. Trigger the DAG manually or wait for scheduled execution
4. Monitor progress in the Airflow interface

#### Manual Execution
```bash
# Run individual components
python src/data_analysis.py              # Data analysis
python src/data_cleaning_utility.py     # Data cleaning
python src/batch/daily_processor.py     # Batch processing
python src/ml/train_model.py            # ML training
```

### Testing ML Predictions
```bash
# Start the ML API
python src/ml/prediction_api.py

# Test predictions
python src/ml/test_api.py

# Manual API test
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"quantity": 3, "unit_price": 15.99, "hour": 14, "day_of_week": 2}'
```

### Streaming Data Processing
```bash
# Terminal 1: Start consumer
python src/streaming/consumer.py

# Terminal 2: Start publisher
python src/streaming/publisher.py
```

### System Monitoring
```bash
# Run system health check
python src/monitoring/monitor.py

# Continuous monitoring
python src/monitoring/monitor.py
# Choose option 2 for continuous monitoring
```

## 📁 Project Structure

```
ecommerce-analytics-pipeline/
├── airflow_dags/                 # Airflow DAG definitions
│   ├── data_pipeline_dag.py      # Main data pipeline DAG
│   └── ecommerce_simple.py       # Simplified testing DAG
├── src/                          # Source code
│   ├── batch/                    # Batch processing
│   │   ├── daily_processor.py    # BigQuery-powered batch processor
│   │   └── integrated_processor.py # ML-integrated processor
│   ├── ml/                       # Machine learning
│   │   ├── train_model.py        # Model training
│   │   ├── prediction_api.py     # REST API for predictions
│   │   ├── data_preparation.py   # Feature engineering
│   │   └── test_api.py          # API testing
│   ├── streaming/                # Real-time data processing
│   │   ├── consumer.py           # Pub/Sub consumer
│   │   └── publisher.py          # Pub/Sub publisher
│   ├── monitoring/               # System monitoring
│   │   └── monitor.py            # Health monitoring
│   ├── data_analysis.py          # Data exploration
│   ├── data_cleaning_utility.py  # Data quality management
│   └── fake_data_generator.py    # Test data generation
├── data/                         # Data storage
│   ├── OnlineRetail.csv          # Main dataset (download required)
│   ├── processed/                # Processed outputs
│   ├── cleaned/                  # Cleaned datasets
│   └── monitoring/               # Monitoring logs
├── docs/                         # Documentation and logs
├── docker-compose.yaml           # Docker services configuration
├── Dockerfile                    # Airflow container definition
├── requirements.txt              # Python dependencies
├── setup_airflow.sh             # Setup script
└── test_complete_system.py      # System testing
```

## 🛠️ Technologies Used

### Core Technologies
- **Python 3.9+**: Primary programming language
- **Apache Airflow 2.8.1**: Workflow orchestration
- **Docker & Docker Compose**: Containerization
- **PostgreSQL**: Airflow metadata database

### Data Processing
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **Google Cloud BigQuery**: Large-scale data processing
- **Google Cloud Pub/Sub**: Real-time messaging

### Machine Learning
- **scikit-learn**: ML algorithms and preprocessing
- **joblib**: Model serialization
- **Flask**: ML prediction API

### Monitoring & Utilities
- **psutil**: System monitoring
- **requests**: HTTP client
- **Faker**: Test data generation

## 🔧 Configuration

### Environment Variables
Key environment variables in `.env`:
```bash
# Airflow Configuration
AIRFLOW_UID=50000
_AIRFLOW_WWW_USER_USERNAME=airflow
_AIRFLOW_WWW_USER_PASSWORD=airflow

# Google Cloud Platform
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
GOOGLE_CLOUD_PROJECT=your-project-id

# Database
POSTGRES_USER=airflow
POSTGRES_PASSWORD=airflow
POSTGRES_DB=airflow
```

### API Endpoints
- **Airflow UI**: `http://localhost:8080`
- **ML Prediction API**: `http://localhost:5001`
  - `GET /health` - Health check
  - `POST /predict` - Revenue prediction
  - `POST /predict/simple` - Simplified prediction

## 🧪 Testing

### Run Complete System Test
```bash
python test_complete_system.py
```

### Individual Component Tests
```bash
# Test ML API
python src/ml/test_api.py

# Test data processing
python src/data_analysis.py

# Test fake data generation
python src/fake_data_generator.py
```

## 📊 Monitoring

The system includes comprehensive monitoring:

- **System Health**: CPU, memory, disk usage
- **API Health**: Response times, error rates
- **Data Quality**: File sizes, processing times
- **Business Metrics**: Revenue, order volumes, customer counts

Access monitoring via:
```bash
python src/monitoring/monitor.py
```

## 🚨 Troubleshooting

### Common Issues

#### Docker Services Won't Start
```bash
# Check logs
docker-compose logs

# Reset everything
docker-compose down -v
docker system prune
docker-compose up --build
```

#### ML API Not Working
```bash
# Check if model exists
ls -la src/ml/revenue_model.pkl

# Retrain model
python src/ml/train_model.py

# Check API logs
python src/ml/prediction_api.py
```

#### BigQuery Connection Issues
```bash
# Check credentials
echo $GOOGLE_APPLICATION_CREDENTIALS

# Test connection
python -c "from google.cloud import bigquery; client = bigquery.Client(); print('Connected!')"
```

### Getting Help
1. Check the logs in `docs/` directory
2. Run the system test: `python test_complete_system.py`
3. Check Docker logs: `docker-compose logs -f`

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** and add tests
4. **Run the test suite**: `python test_complete_system.py`
5. **Commit your changes**: `git commit -m 'Add amazing feature'`
6. **Push to the branch**: `git push origin feature/amazing-feature`
7. **Open a Pull Request**

### Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include error handling and logging
- Update documentation for new features
- Test your changes thoroughly

## 🙏 Acknowledgments

- **UCI Machine Learning Repository** for the Online Retail dataset
- **Apache Airflow** community for the orchestration framework
- **Google Cloud Platform** for cloud infrastructure
- **scikit-learn** community for machine learning tools

---

**Built with ❤️ for data engineering and analytics**

For questions or support, please open an issue in the GitHub repository.
