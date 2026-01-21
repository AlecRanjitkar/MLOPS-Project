# 👗 Fashion-MNIST Classification — MLOps Project

## 🚀 Project Overview

This project implements a **reproducible end-to-end machine learning pipeline** for image classification, with a strong emphasis on **MLOps best practices** rather than achieving state-of-the-art accuracy.

The goal is to demonstrate how a real-world machine learning system can be:
- cleanly structured  
- configuration-driven  
- fully reproducible  
- experiment-tracked  
- containerized and deployable  

We use the **Fashion-MNIST** dataset to classify grayscale images of clothing items. Each image is `28×28` pixels and belongs to one of **10 clothing categories** (e.g. T-shirt/top, trouser, sneaker, coat).

📦 **Dataset size:** 70,000 images  
🧵 **Classes:** 10 clothing types  
📁 **Versioning:** DVC  

🔗 Dataset link:  
https://www.kaggle.com/datasets/zalando-research/fashionmnist

---

## 🧠 What This Project Demonstrates

### 📝 Application Logging (M14)
- Structured logging using **Loguru**
- Logs important runtime events:
  - device selection (CPU / MPS / CUDA)
  - training start and completion
  - batch- and epoch-level progress
  - evaluation metrics
  - model checkpoint saving
- Makes debugging and monitoring easier and more transparent

---

### 📊 Experiment Tracking with Weights & Biases (M14)
- Logs:
  - training loss and accuracy
  - test accuracy per epoch
  - training curves as plots
  - trained models as **W&B artifacts**
- Stores full experiment configuration automatically
- Enables easy comparison and reproducibility via the W&B dashboard

---

### 🔍 Hyperparameter Optimization (M14)
- Hyperparameter sweeps using **Weights & Biases Sweeps**
- Tuned parameters:
  - learning rate
  - batch size
  - number of epochs
  - model type (MLP vs CNN)
- Optimization metric:
  - `epoch/test_acc`
- Multiple experiments launched automatically using sweep agents

---

## 🗂 Project Structure

```txt
├── .github/                  # CI and GitHub actions
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Hydra & W&B configuration files
│   └── sweep.yaml
├── data/                     # Versioned data (DVC)
│   ├── processed/
│   └── raw/
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── models/                   # Trained model checkpoints
├── notebooks/                # Exploration notebooks
├── reports/
│   └── figures/              # Training plots
├── src/
│   └── fashionmnist_classification_mlops/
│       ├── __init__.py
│       ├── api.py
│       ├── data.py
│       ├── evaluate.py
│       ├── logging_utils.py
│       ├── model.py
│       ├── sweep_runner.py
│       ├── train.py
│       └── visualize.py
├── tests/                    # Unit tests
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml
├── README.md
├── requirements.txt
└── requirements_dev.txt
```



# ⚙️ Setup Instructions (From Scratch)
## 1️⃣ Clone the Repository

```bash
git clone <repository-url>
cd MLOPS-Project
```

## 2️⃣ Create and Activate Virtual Environment
````bash
python -m venv .venv
source .venv/bin/activate
````

## 3️⃣ Install Dependencies
````bash
pip install -r requirements.txt
pip install -r requirements_dev.txt
````

## 4️⃣ Install Project as Editable Package
````bash
pip install -e .
````

## 5️⃣ Add DVC Configuration for Google Drive Remote
We are using Google Drive as the remote storage for DVC, ensure the .dvc/config file is correctly set up. Add the following configuration:

```
[core]
    remote = gdrive_remote

[remote "gdrive_remote"]
    url = gdrive://<your-drive-folder-id>
    gdrive_client_id = "<your-client-id>"
    gdrive_client_secret = "<your-client-secret>"
```
# 📦 Data Version Control (DVC)

> Prerequisite: Docker installed and running

Install DVC and remote storage support:

````bash
pip install dvc dvc-gdrive
````
Pull the versioned dataset:

````bash
dvc pull
````

This populates:

- data/raw/

- data/processed/

# 🏋️ Training the Model (Local Python)
Default training:

````bash
python -m fashionmnist_classification_mlops.train

Custom hyperparameters:
python -m fashionmnist_classification_mlops.train \
  hyperparameters.learning_rate=0.001 \
  hyperparameters.batch_size=64 \
  hyperparameters.epochs=5 \
 ````

# 📌 Training automatically logs metrics, plots, and models.

## 🔁 Hyperparameter Sweeps (Weights & Biases)
### 1️⃣ Login to W&B
````bash
wandb login
````

### 2️⃣ Create a Sweep
````bash
wandb sweep configs/sweep.yaml
````

This command returns a sweep ID.

### 3️⃣ Run the Sweep Agent
```bash
wandb agent <entity>/<project>/<sweep_id>
```

The agent automatically:

- launches multiple training runs

- explores different hyperparameter combinations

- logs everything to the W&B dashboard

# 🐳 Dockerized Execution
Train the Model in Docker:

```bash
docker build -f dockerfiles/train.Dockerfile -t fashionmnist-train .
docker run --rm fashionmnist-train
```

Run the Inference API:
```bash
docker build -f dockerfiles/api.Dockerfile -t fashionmnist-api .
docker run -p 8000:8000 fashionmnist-api
```
🚀 The API exposes an endpoint for Fashion-MNIST predictions.

- ♻️ Reproducibility

Every experiment can be reproduced using:

- Git commit hash

- Hydra configuration files

- DVC-tracked data versions

- Logged hyperparameters

- W&B artifacts and run metadata

This ensures full traceability from raw data to trained model.