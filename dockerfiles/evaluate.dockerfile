FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Install evaluation-specific dependencies
RUN pip install scikit-learn matplotlib

# Copy source code, data, and model
COPY src/ ./src/
COPY conf/ ./conf/
COPY pyproject.toml .

# Install the package
RUN pip install -e .

# Create output directory for reports
RUN mkdir -p reports/figures models data/processed

# Set environment variable
ENV PYTORCH_ENABLE_MPS_FALLBACK=1

# Run evaluation
CMD ["python", "-m", "fashionmnist_classification_mlops.evaluate"]