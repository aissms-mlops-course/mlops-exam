# Use official Python image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set working directory inside the container
WORKDIR /webapp/pages

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt into /webapp
COPY requirements.txt /webapp/

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r /webapp/requirements.txt

# Copy app code
COPY webapp/pages/mlops-endsem.py /webapp/pages/
COPY models/trained.h5 /webapp/pages/models/

# Expose Streamlit default port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "mlops-endsem.py", "--server.port=8501", "--server.address=0.0.0.0"]
