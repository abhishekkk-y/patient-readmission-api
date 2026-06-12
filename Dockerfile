# Start from an official Python image
# "slim" means it's a stripped down version — smaller and faster
FROM python:3.11-slim

# Set the working directory inside the container
# All commands from here on run inside /app
WORKDIR /app

# Copy requirements first and install dependencies
# We do this before copying code so Docker can cache this layer
# If requirements don't change, Docker skips this step on rebuilds
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the API code
COPY app/ ./app/

# Copy the MLflow folder — this contains your trained model
COPY mlruns/ ./mlruns/

# Copy the MLflow database — this contains the model registry,
# including registered model names and aliases like "champion"
COPY mlflow.db ./mlflow.db

# Tell MLflow to use the database backend for tracking/registry
ENV MLFLOW_TRACKING_URI=sqlite:///mlflow.db
# Load the local model artifact path inside the container
ENV READMISSION_MODEL_URI=mlruns/1/models/m-883a2c92d3844206a9d7d90af3f7f57a/artifacts

# Tell Docker this app uses port 8000
EXPOSE 8000

# The command that runs when the container starts
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]