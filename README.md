# E-Commerce Analytics Platform
**Data Engineering 2 - Final Project**  
*MSc Applied Data Science and Analytics*

## 🎯 Project Overview

A comprehensive real-time e-commerce analytics system that demonstrates both batch and streaming data processing capabilities with integrated machine learning. The platform processes transaction data to provide business insights, customer segmentation, and revenue predictions.

### Key Features
- **Real-time Data Streaming** with Google Cloud Pub/Sub
- **Batch Processing Pipeline** for historical analysis and daily/weekly reporting
- **Machine Learning Integration** for revenue prediction
- **Customer Segmentation** and behavior analysis
- **Business Intelligence Dashboards** with key performance metrics
- **Comprehensive Monitoring** and logging

## 🏗️ Architecture

```
📊 Kaggle Dataset → 🔄 Batch Processing → 📈 BigQuery Storage
                            ↓
🌊 Streaming Data → 📡 Pub/Sub → 🤖 ML Predictions → 📊 Real-time Analytics
                            ↓
                    🔍 Monitoring & Alerts
```

### Technology Stack
- **Data Processing**: Python, Pandas, NumPy
- **Cloud Platform**: Google Cloud Platform (BigQuery, Pub/Sub)
- **Machine Learning**: Scikit-learn, Random Forest
- **API Framework**: Flask
- **Monitoring**: Python logging, psutil
- **Data Generation**: Faker library

## 📁 Project Structure

```
ecommerce-analytics-project/
├── data/
│   ├── OnlineRetail.csv           # Kaggle dataset
│   ├── ml_ready/                  # Processed ML datasets
│   └── processed/                 # Batch processing outputs
├── src/
│   ├── batch/
│   │   └── daily_processor.py     # Main batch processing pipeline
│   ├── ml/
│   │   ├── train_model.py         # ML model training
│   │   ├── prediction_api.py      # Flask prediction API
│   │   ├── test_api.py           # API testing script
│   │   └── data_preparation.py    # ML data preprocessing
│   ├── streaming/
│   │   ├── publisher.py          # Pub/Sub message publisher
│   │   └── consumer.py           # Real-time stream processor
│   ├── data_analysis.py          # Exploratory data analysis
│   └── fake_data_generator.py    # Synthetic data generation
├── docs/
│   ├── batch_processing.log      # Processing logs
│   ├── dataset_summary.csv       # Data analysis results
│   └── business_metrics.json     # Key business KPIs
└── requirements.txt              # Python dependencies
```

## 🚀 Quick Start

### Prerequisites

1. **Python Environment** (3.9)
```bash
pip install pandas numpy faker google-cloud-pubsub google-cloud-bigquery 
pip install scikit-learn flask apache-airflow psutil matplotlib seaborn
```

2. **Kaggle Dataset**
   - Download [Online Retail Dataset](https://www.kaggle.com/datasets/vijayuv/onlineretail)
   - Place `OnlineRetail.csv` in the `data/` folder

3. **Google Cloud Setup**
   - Create GCP project with BigQuery and Pub/Sub APIs enabled
   - Download service account JSON key
   - Set `GOOGLE_APPLICATION_CREDENTIALS` environment variable

### Installation & Setup

1. **Clone and Setup**
```bash
git clone <repository-url>
cd ecommerce-analytics-project
pip install -r requirements.txt
```

2. **Initialize Data Analysis**
```bash
python src/data_analysis.py
```

3. **Prepare ML Data**
```bash
python src/ml/data_preparation.py
```

4. **Train ML Model**
```bash
python src/ml/train_model.py
```

## 🔧 Usage Guide

### 1. Batch Processing Pipeline

Process historical data and generate business intelligence reports:

```bash
python src/batch/daily_processor.py
```

**Features:**
- Daily business metrics calculation
- Weekly trend analysis
- Customer segmentation (VIP, High, Medium, Low value)
- Top performers identification
- Revenue and order analytics

**Output Example:**
```
==================================================
DAILY E-COMMERCE ANALYTICS REPORT
==================================================
Date: 2011-12-09
Total Revenue: £184,349.28
Total Orders: 617
Unique Customers: 287
Unique Products: 1,814
Avg Order Value: £298.94
Peak Hour: 12:00 (£28,485.67)
```

### 2. Machine Learning API

Start the revenue prediction service:

```bash
# Terminal 1: Start API
python src/ml/prediction_api.py

# Terminal 2: Test predictions
python src/ml/test_api.py
```

**API Endpoints:**
- `POST /predict` - Predict order revenue
- `GET /health` - Service health check

**Example Request:**
```json
{
    "quantity": 3,
    "unit_price": 15.50,
    "hour": 14,
    "day_of_week": 2
}
```

**Example Response:**
```json
{
    "predicted_revenue": 46.50,
    "status": "success"
}
```

### 3. Real-time Streaming

Process live transaction data:

```bash
# Terminal 1: Start consumer
python src/streaming/consumer.py

# Terminal 2: Publish test data
python src/streaming/publisher.py
```

**Real-time Analytics:**
- Transaction volume monitoring
- High-value order alerts (>£100)
- Revenue tracking
- Customer behavior patterns

### 4. System Monitoring

```bash
python src/monitoring/monitor.py
```

Monitors:
- CPU and memory usage
- API health status
- Processing performance
- Error rates and alerts

## 📊 Business Intelligence Features

### Daily Metrics Dashboard
- **Revenue Analytics**: Total daily/weekly revenue, growth rates
- **Order Intelligence**: Order volumes, average order values
- **Customer Insights**: Unique customers, retention metrics
- **Product Performance**: Best-selling products, inventory trends
- **Geographic Analysis**: Revenue by country/region

### Customer Segmentation
- **VIP Customers**: Top 5% by revenue
- **High Value**: 67th-95th percentile
- **Medium Value**: 33rd-67th percentile  
- **Low Value**: Bottom 33%

### Predictive Analytics
- **Revenue Forecasting**: ML-powered order value prediction
- **Peak Time Analysis**: Optimal business hours identification
- **Seasonal Trends**: Monthly and weekly pattern recognition

## 🔬 Data Pipeline Architecture

### Batch Processing Flow
```
Raw Data → Data Cleaning → Feature Engineering → Analytics → Storage → Reports
```

### Streaming Processing Flow
```
Live Orders → Pub/Sub → Real-time Processing → ML Predictions → Alerts
```

### ML Pipeline
```
Historical Data → Feature Engineering → Model Training → API Deployment → Predictions
```

## 📈 Performance Metrics

- **Data Volume**: 397,884+ clean transactions processed
- **Processing Speed**: ~10,000 records/minute batch processing
- **ML Accuracy**: Random Forest model with optimized MSE
- **Real-time Latency**: <100ms streaming processing
- **API Response Time**: <50ms prediction endpoint

## 🛠️ Development & Deployment

### Running the Complete System

1. **Start Core Services**
```bash
# ML API (Terminal 1)
python src/ml/prediction_api.py

# Monitoring (Terminal 2)  
python src/monitoring/monitor.py
```

2. **Process Historical Data**
```bash
python src/batch/daily_processor.py
```

3. **Enable Real-time Processing**
```bash
# Consumer (Terminal 3)
python src/streaming/consumer.py

# Publisher (Terminal 4)
python src/streaming/publisher.py
```

### Configuration

Update GCP settings in streaming files:
```python
# Replace 'your-project-id' with actual GCP project ID
topic_path = publisher.topic_path('your-project-id', 'orders-stream')
subscription_path = subscriber.subscription_path('your-project-id', 'orders-consumer')
```

## 🎯 Business Value & Use Cases

### Retail Intelligence
- **Revenue Optimization**: Identify peak sales periods and optimize staffing
- **Inventory Management**: Predict demand patterns for better stock planning
- **Customer Retention**: Segment customers for targeted marketing campaigns

### Real-time Operations
- **Fraud Detection**: Monitor unusual transaction patterns
- **Dynamic Pricing**: Adjust prices based on real-time demand
- **Supply Chain**: Real-time inventory alerts and reorder triggers

### Strategic Planning
- **Market Analysis**: Geographic performance insights
- **Product Strategy**: Data-driven product development decisions
- **Financial Forecasting**: Revenue prediction for business planning

## 🚨 Troubleshooting

### Common Issues

**"Module not found" errors:**
```bash
pip install --user [package-name]
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**GCP Authentication errors:**
```bash
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-key.json"
```

**API not responding:**
- Verify Flask app runs on port 5001
- Check firewall settings
- Ensure model file exists: `src/ml/revenue_model.pkl`

**Data file missing:**
- Download OnlineRetail.csv from Kaggle
- Place in `data/` directory
- Run data analysis first: `python src/data_analysis.py`

## 📝 Future Enhancements

- **Advanced ML Models**: Deep learning for complex pattern recognition
- **Real-time Dashboards**: Interactive web-based analytics interface
- **Automated Alerting**: Email/SMS notifications for critical events
- **Data Lake Integration**: Scalable storage for big data analytics
- **A/B Testing Framework**: Experiment management and analysis
- **Advanced Security**: Data encryption and access controls

## 👥 Team Contributions

This project demonstrates expertise in:
- **Data Architecture**: Scalable pipeline design and implementation
- **Machine Learning**: Model development, training, and deployment
- **Cloud Engineering**: GCP services integration and management
- **Real-time Systems**: Streaming data processing and analytics
- **Business Intelligence**: KPI development and reporting automation

## 📊 Demo Script

### 5-Minute System Demonstration

1. **Architecture Overview** (1 min)
   - Show system diagram and data flow
   - Explain batch vs streaming components

2. **Real-time Processing** (2 min)
   - Start monitoring dashboard
   - Demonstrate live data streaming
   - Show real-time alerts and analytics

3. **ML Predictions** (1 min)
   - API call demonstration
   - Revenue prediction examples
   - Model performance metrics

4. **Business Intelligence** (1 min)
   - Daily analytics report
   - Customer segmentation results
   - Key business insights

---

*This project showcases a production-ready data engineering solution combining modern cloud technologies, machine learning, and real-time analytics for comprehensive business intelligence.*
