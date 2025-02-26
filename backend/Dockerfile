# Use the official Python image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Set the environment variable for model path
ENV MODEL_PATH="/app/ml/models/model_randomForest.pkl"

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["python3", "app.py"]