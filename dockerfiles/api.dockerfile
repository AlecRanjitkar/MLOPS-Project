FROM python:3.11-slim

WORKDIR /app

# Copy requirements
COPY requirements-docker.txt .
COPY pyproject.toml .

RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy source code and model
COPY src/ src/

# Install the project
RUN pip install --no-cache-dir -e .

EXPOSE 8080
ENTRYPOINT ["sh", "-c", "uvicorn fashionmnist_classification_mlops.api:app --host 0.0.0.0 --port ${PORT:-8080}"]
