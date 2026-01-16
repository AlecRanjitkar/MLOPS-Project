FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install evaluation-specific dependencies
RUN pip install scikit-learn matplotlib

# Copy source code, data, and model
COPY src/ ./src/
COPY data/processed/ ./data/processed/
COPY models/model.pth ./models/model.pth
COPY pyproject.toml .

# Install the package
RUN pip install -e .

# Create output directory for reports
RUN mkdir -p reports/figures

# Set environment variable
ENV PYTORCH_ENABLE_MPS_FALLBACK=1

# Run evaluation
CMD ["python", "-m", "fashionmnist_classification_mlops.evaluate"]