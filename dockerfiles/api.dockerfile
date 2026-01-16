FROM python:3.11-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install FastAPI and dependencies
RUN pip install fastapi uvicorn[standard] python-multipart pillow

# Copy source code and trained model
COPY src/ ./src/
COPY models/model.pth ./models/model.pth
COPY pyproject.toml .

# Install the package
RUN pip install -e .

# Expose port
EXPOSE 8000

# Run the API
CMD ["uvicorn", "fashionmnist_classification_mlops.api:app", "--host", "0.0.0.0", "--port", "8000"]