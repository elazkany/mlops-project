# Use a base image with Python and MLflow
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy app and mlruns directory
COPY app/ ./app/
#COPY mlruns/ ./mlruns/
COPY deployment/model_artifacts/ ./model_artifacts/


# Set environment variable for MLflow tracking
#ENV MLFLOW_TRACKING_URI=file:./mlruns

# Expose FastAPI port
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]
