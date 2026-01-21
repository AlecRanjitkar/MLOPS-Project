FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt
# Copy source code and data
COPY src/ ./src/
COPY conf/ ./conf/
COPY pyproject.toml .

# Install the package
RUN pip install -e .

# Create directories for outputs
RUN mkdir -p models reports/figures data/processed

# Set environment variable to avoid MPS on non-Mac systems
ENV PYTORCH_ENABLE_MPS_FALLBACK=1

# Default command: train the model
CMD ["python", "-m", "fashionmnist_classification_mlops.train"]