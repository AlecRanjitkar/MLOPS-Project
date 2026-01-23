
# 👕 Fashion-MNIST Classification — MLOps Project

This repository implements an end-to-end **MLOps pipeline** for classifying Fashion-MNIST images using **PyTorch**, focusing on **reproducibility, monitoring, deployment, and robustness** rather than maximizing accuracy.

The project follows DTU MLOps modules and demonstrates:
- experiment tracking
- containerized training and serving
- CI/testing
- load testing
- monitoring and drift detection

---

## 📊 Dataset & Task

- **Dataset**: Fashion-MNIST
- **Input**: 28×28 grayscale images
- **Output**: 1 of 10 clothing classes
- **Model**: CNN (PyTorch)

---

## 🧱 Project Structure

```text
MLOPS-Project/
├── configs/                     # Hydra configs
├── data/                        # DVC-tracked data
├── dockerfiles/                 # Dockerfiles
├── models/                      # Saved model weights
├── monitoring/
│   ├── monitoring_api.py        # Drift detection API
│   ├── reference_features.csv   # Reference feature distribution
│   └── prediction_log.csv       # Logged predictions
├── src/fashionmnist_classification_mlops/
│   ├── api.py                   # FastAPI inference API
│   ├── train.py                 # Training script
│   ├── evaluate.py              # Evaluation script
│   ├── model.py                 # Model definitions
│   └── sweep_runner.py          # W&B sweep runner
├── tests/
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── locustfile.py                # Load testing
├── README.md
└── pyproject.toml
````

---

## ⚙️ Setup Instructions

### 1️⃣ Clone repository and create virtual environment

```bash
git clone <REPO_URL>
cd MLOPS-Project

python -m venv .venv
source .venv/bin/activate
pip install -U pip
```

---

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
pip install -r requirements_dev.txt
```

---

### 3️⃣ Pull data with DVC

```bash
pip install dvc dvc-gs
dvc pull
```

---

## 🏋️ Model Training & Evaluation

### Local execution

```bash
python -m fashionmnist_classification_mlops.train
python -m fashionmnist_classification_mlops.evaluate
```

---

### Dockerized training (recommended)

```bash
docker-compose run --rm train
docker-compose run --rm evaluate
```

---

## 📈 Experiment Tracking (Weights & Biases)

### Login to W&B

```bash
wandb login
```

---

### Run training with logging

```bash
python -m fashionmnist_classification_mlops.train
```

---

### Run hyperparameter sweeps

```bash
wandb sweep configs/sweep.yaml
wandb agent <ENTITY>/<PROJECT>/<SWEEP_ID>
```

---

## 🚀 FastAPI Inference Service

### Run locally

```bash
uvicorn fashionmnist_classification_mlops.api:app --host 0.0.0.0 --port 8000
```

---

### Run with Docker

#### Build image

```bash
docker build -t fashionmnist-api:latest -f dockerfiles/api.Dockerfile .
```

#### Run container

```bash
docker run --rm -p 8000:8080 \
  -v "$(pwd)/models":/app/models \
  fashionmnist-api:latest
```

---

### Test API

```bash
curl http://localhost:8000/health
curl http://localhost:8000/classes
```

---

### Run prediction

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@bag.png"
```

---

## 🧪 Testing & CI

### Run tests

```bash
pytest
```

---

## 🔥 Load Testing (Locust)

### Start API

```bash
uvicorn fashionmnist_classification_mlops.api:app --port 8000
```

---

### Run Locust

```bash
locust -f locustfile.py --host http://localhost:8000
```

Open browser:

```
http://localhost:8089
```

---

## 📡 Monitoring & Drift Detection (Module M27)

---

### ✅ Task 1: Model robustness to data drift


Robustness was evaluated by comparing prediction behavior under distribution shifts using statistical feature summaries.

---

### ✅ Task 2: Input–output data collection

The FastAPI `/predict` endpoint logs **input features and predictions** automatically into:

```text
monitoring/prediction_log.csv
```

Each row includes:

* timestamp
* feature statistics (brightness, contrast)
* predicted class
* confidence

Example:

```bash
head monitoring/prediction_log.csv
```

---

### ✅ Task 3: Drift Detection API (Local Deployment)

#### 1️⃣ Go to monitoring directory

```bash
cd monitoring
```

---

#### 2️⃣ Build Docker image

```bash
docker build -t drift-monitor .
```

---

#### 3️⃣ Run drift API

```bash
docker run --rm -p 8080:8080 \
  -v "$(pwd)":/app \
  drift-monitor
```

---

#### 4️⃣ Generate drift report

```bash
curl http://localhost:8080/drift-report > drift_report.html
open drift_report.html
```

The report is generated using **Evidently** and compares:

* `reference_features.csv`
* `prediction_log.csv`

---

### ☁️ Cloud Deployment (Conceptual)

To deploy the drift API to the cloud:

1. Upload logs to cloud storage (e.g. GCS)
2. Modify `monitoring_api.py` to pull latest logs from cloud
3. Deploy container to Cloud Run
4. Access `/drift-report` via public endpoint

---

## 🧹 Git Hygiene

Recommended `.gitignore` entries:

```txt
wandb/
outputs/
monitoring/prediction_log.csv
monitoring/*.html
```
---

## ⚙️ Advanced MLOps Topics (Modules M28–M31)

Due to the project being carried out by a **3-person group**, not all advanced MLOps modules could be fully implemented within the given timeframe. However, the following section documents **what was implemented**, **what was partially implemented**, and **how remaining components would be completed in a production setting**, demonstrating understanding of the concepts.

---

## 📡 M28 — System Metrics & Cloud Monitoring

### Instrument API with system metrics (M28)

The inference API was instrumented to expose **basic runtime and operational metrics**, including:
- request count
- average inference latency
- model load status

Metrics are available via the `/metrics` endpoint:

```bash
curl http://localhost:8000/metrics
````

Example metrics exposed:

* total requests
* average prediction time
* model availability

In a production setting, this endpoint would be scraped by **Prometheus** or a similar monitoring system.

---

### Cloud monitoring setup (M28)

While full cloud monitoring was not deployed, the project architecture supports:

* containerized services (Docker)
* stateless APIs
* metric endpoints compatible with Google Cloud Monitoring

In a cloud deployment, metrics would be collected using:

* **Google Cloud Monitoring agents**
* or **Prometheus → Cloud Monitoring integration**

---

### Alerting systems in GCP (M28)

Due to time constraints, alert policies were not created in GCP.
However, the following alerts would be configured in production:

| Metric         | Alert Condition                 |
| -------------- | ------------------------------- |
| API health     | `/health` returns non-200       |
| Latency        | Avg latency > threshold         |
| Error rate     | 5xx responses exceed limit      |
| Drift detected | Dataset drift score > threshold |

Alerts would notify operators via:

* email
* Slack
* PagerDuty

---

## ⚡ M29 — Distributed Data Loading

Distributed data loading was **not required** for this project due to:

* small dataset size (Fashion-MNIST)
* single-node training sufficiency

However, the training pipeline uses PyTorch `DataLoader`, which can be extended with:

* `num_workers > 0`
* distributed samplers (`torch.utils.data.distributed.DistributedSampler`)

This allows straightforward scaling if training on larger datasets.

---

## 🧠 M30 — Distributed Training

Distributed training was not implemented because:

* the dataset and model are lightweight
* training completes quickly on a single CPU/GPU

If scaled to larger workloads, the training script can be extended using:

* PyTorch `DistributedDataParallel (DDP)`
* multi-GPU or multi-node setups

The current architecture supports this extension with minimal changes.

---

## ⚡ M31 — Model Optimization (Quantization, Compilation, Pruning)

Inference optimization techniques were **investigated conceptually**, but not fully implemented due to time constraints.

Potential optimizations include:

* **Quantization**

  * reduce model precision (FP32 → INT8)
  * faster inference on CPU
* **Torch compilation**

  * `torch.compile()` to optimize computation graph
* **Pruning**

  * remove low-importance weights

These techniques would significantly improve inference speed in production and are compatible with the current PyTorch model.

---

### Rationale for Partial Coverage

This project prioritizes:

* correctness
* reproducibility
* monitoring
* deployment

Given limited group size and time, the team focused on implementing **core MLOps principles end-to-end**, while documenting advanced modules with clear reasoning and extension paths.

---



## 🏁 Summary

This project demonstrates a complete MLOps workflow including:

* Data versioning (DVC)
* Experiment tracking (W&B)
* CI/testing
* Dockerized training and inference
* Load testing
* Monitoring and drift detection

The focus is on **engineering robustness and reproducibility**, not just model performance.
