# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
   gcc \
   && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Build the index before copying the rest of the code
RUN python build_index.py

# Expose the port the app runs on
EXPOSE 8000
# Command to run the application
#CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--root-path", "/mentor/proxy/8000"]
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
