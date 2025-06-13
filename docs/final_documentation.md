E-Commerce Analytics Project – Final Documentation

Project Title:

Airflow-Orchestrated E-Commerce Analytics System

⸻

Team Members:

Name	Role & Responsibilities
Samiksha Dhere	Data Handling & Batch Processing 
Nitanshu Idnani	Streaming, GCP Setup, ML Training & Deployment 
Aswin Subash	Architecture, Testing, Monitoring, Documentation 


⸻

Project Overview

This project demonstrates a complete, hybrid e-commerce analytics system integrating batch data pipelines, real-time streaming, and a machine learning prediction API, all orchestrated using Apache Airflow DAGs.

We used the Online Retail dataset from Kaggle, simulated real-time order flow, and built a predictive model to estimate revenue.

⸻

Final Architecture

An architecture diagram with DAG integration was created, showing three main Airflow DAGs:
	•	Daily Analytics DAG
	•	ML Training DAG
	•	Data Quality Monitoring DAG

All DAGs were designed and implemented collaboratively by Samiksha, Nitanshu, and Aswin.

⸻

Day-by-Day Breakdown

Day 1: Project Setup & Architecture
	•	Samiksha set up project folder structure.
	•	Nitanshu created the GCP project and enabled BigQuery, Pub/Sub, and Vertex AI APIs.
	•	Aswin created GitHub repo and the architecture diagram and saved it to docs.

Deliverables:
	•	README.md
	•	docs/architecture.png

⸻

Day 2: Data Understanding & Fake Generator
	•	Samiksha analyzed the Kaggle dataset and created a summary CSV.
	•	Nitanshu created a new BigQuery dataset in GCP.
	•	Aswin developed the src/fake_data_generator.py to simulate realistic streaming orders.

Deliverables:
	•	data/OnlineRetail.csv
	•	data/dataset_summary.csv
	•	src/fake_data_generator.py

Example Output:
	•	Sample fake order: {'InvoiceNo': 'F4A7C9', 'StockCode': '85123A', 'Description': 'Handcrafted item', 'Quantity': 3, 'InvoiceDate': '...', 'UnitPrice': 14.99, 'CustomerID': 17634, 'Country': 'United Kingdom'}

⸻

Day 3: Batch Pipeline
	•	Samiksha wrote src/batch/daily_processor.py to compute daily KPIs.
	•	Nitanshu created a service account and configured BigQuery access.
	•	Aswin tested the batch pipeline, validated outputs, and documented the results in docs/daily_log.md.

Deliverables:
	•	src/batch/daily_processor.py
	•	credentials/ecommerce-key.json
	•	data/daily_metrics_2025-06-09.csv

Example Output:
	•	Total Orders: 156
	•	Total Revenue: £2,350.45
	•	Unique Customers: 87

⸻

Day 4: Real-Time Streaming
	•	Nitanshu created src/streaming/publisher.py and src/streaming/consumer.py with GCP Pub/Sub integration.
	•	Samiksha updated the fake data generator.
	•	Aswin tested the full streaming pipeline and logged test outputs.

Deliverables:
	•	src/streaming/publisher.py
	•	src/streaming/consumer.py

Example Output:
	•	Published order: 7D2A3F
	•	Received order: 7D2A3F - 2 x $29.99
	•	HIGH VALUE ORDER: $119.96

⸻

Day 5: ML Model Development
	•	Nitanshu trained a RandomForestRegressor to predict order revenue.
	•	Samiksha assisted in data cleaning and feature engineering.
	•	Aswin documented the ML model in docs/model_info.md.

Deliverables:
	•	src/ml/train_model.py
	•	src/ml/revenue_model.pkl
	•	docs/model_info.md

Example Output:
	•	Model trained! MSE: 2320.45
	•	Features used: Quantity, UnitPrice, Hour, DayOfWeek

⸻

Day 6: ML API & Monitoring
	•	Nitanshu built the Flask-based ML API (src/ml/prediction_api.py) and test script (src/ml/test_api.py).
	•	Aswin created a health check and monitoring script (src/monitoring/monitor.py).
	•	Samiksha tested API endpoints and prediction consistency.

Deliverables:
	•	src/ml/prediction_api.py
	•	src/ml/test_api.py
	•	src/monitoring/monitor.py
	•	docs/monitoring_log.txt

Example Output:
	•	API prediction response: {"predicted_revenue": 47.25, "status": "success"}
	•	Health check response: {"status": "healthy"}

⸻

Day 7: Integration Testing
	•	Samiksha modified batch script to fetch predictions from the ML API.
	•	Nitanshu updated the consumer to call the ML API in real-time.
	•	Aswin ran and validated the full end-to-end integration.

Deliverables:
	•	Updated src/batch/daily_processor.py
	•	Updated src/streaming/consumer.py
	•	Final prediction output with ML integration: data/daily_metrics_2025-06-11.csv

Example Output:
	•	InvoiceNo: A7783T | Actual: 3 x $12.00 | Predicted Revenue: $37.29

⸻

Day 8: Documentation & Demo Prep
	•	Samiksha finalized the README.md file.
	•	Nitanshu created a detailed demo script (docs/demo_script.md).
	•	Aswin compiled presentation slides and finalized documentation including DAG architecture.

Deliverables:
	•	README.md
	•	docs/demo_script.md
	•	docs/daily_log.md
	•	docs/presentation_slides.pdf
	•	docs/PHOTO-2025-06-12-01-23-03.jpg

⸻

Technologies Used

Category	Tools/Services
Orchestration	Apache Airflow (DAGs for batch/ML)
Data Warehouse	Google BigQuery
Streaming	Google Cloud Pub/Sub
ML Model	Random Forest Regressor (scikit-learn)
API Deployment	Flask
Monitoring	psutil, custom logging
Source Control	GitHub
Data Source	Kaggle - Online Retail Dataset


⸻

Final Outcomes
	•	Real-time and batch data pipelines fully operational
	•	ML model predicts order revenue accurately
	•	All components integrated via Airflow DAGs and Flask API
	•	Monitoring and logging systems implemented
	•	Fully demo-ready and production-structured pipeline

⸻

Team Contribution Note

All DAGs and orchestration logic in Apache Airflow were jointly designed and implemented by Samiksha Dhere, Nitanshu Idnani, and Aswin Subash.

⸻

Summary

This project demonstrates how to build a modular, scalable e-commerce analytics system using modern cloud-native tools. By integrating data pipelines, ML prediction, real-time streaming, and orchestration with Airflow, the team built a robust platform in just eight days of focused teamwork.