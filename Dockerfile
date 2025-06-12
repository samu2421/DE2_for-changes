# Use official Apache Airflow 2.8.1 image as base
FROM apache/airflow:2.8.1-python3.9

# Switch to root user to install system packages
USER root

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create necessary directories with proper permissions
RUN mkdir -p /opt/airflow/dags \
    && mkdir -p /opt/airflow/src \
    && mkdir -p /opt/airflow/data \
    && mkdir -p /opt/airflow/data/cleaned \
    && mkdir -p /opt/airflow/data/processed \
    && mkdir -p /opt/airflow/data/monitoring \
    && mkdir -p /opt/airflow/data/ml_ready \
    && mkdir -p /opt/airflow/logs \
    && mkdir -p /opt/airflow/plugins \
    && chown -R airflow:root /opt/airflow

# Switch back to airflow user
USER airflow

# Set environment variables
ENV PYTHONPATH="/opt/airflow/src:/opt/airflow/dags:${PYTHONPATH}"
ENV AIRFLOW__CORE__DAGS_FOLDER="/opt/airflow/dags"
ENV AIRFLOW__CORE__LOAD_EXAMPLES=False
ENV AIRFLOW__CORE__ENABLE_XCOM_PICKLING=True

# Copy and install Python dependencies
COPY requirements.txt /opt/airflow/requirements.txt
RUN pip install --no-cache-dir -r /opt/airflow/requirements.txt

# Copy project source code (these will also be mounted as volumes)
COPY --chown=airflow:root src/ /opt/airflow/src/
COPY --chown=airflow:root airflow_dags/ /opt/airflow/dags/

# Set working directory
WORKDIR /opt/airflow

# Verify Airflow installation
RUN python -c "import airflow; print('Airflow version:', airflow.__version__)"

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD python -c "import airflow; print('OK')" || exit 1