FROM python:3.8

# Install MLflow
RUN pip install mlflow

# Copy the MLflow model
COPY mlruns /mlruns

# Expose the port
EXPOSE 5000

# Serve the MLflow model
CMD mlflow models serve -m /mlruns/0/976a8ca3770647db86a5b00187b76170/artifacts/random_forest_model -p 5000

