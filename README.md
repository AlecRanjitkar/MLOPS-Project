# Project Description: Fashion-MNIST Classification

The goal of this project is to build a reproducible end to- nd machine learning pipeline for image classification, with the main focus on applying MLOps principles rather than achieving the highest possible model accuracy. The project aims to show how machine learning projects can be structured in a clean and systematic way, where data, code, configurations, and experiments are properly organized, version-controlled, and easy to reproduce. The pipeline will cover the full workflow, including data handling, model training and evaluation, logging, and containerized execution, similar to how machine learning systems are developed and maintained in real world applications.

We will be using Fashion-MNIST dataset to classify images of clothing items. Each input consists of a 28×28 grayscale image, and the model must predict one of ten clothing categories, such as T-shirt/top, trouser, sneaker, or coat. The project will initially use the Fashion-MNIST dataset, which contains 70,000 labeled images split into training and test sets. With a total size of around 60 MB. The data will be loaded using the Torchvision library and version controlled with DVC, making it possible to track changes to the dataset or preprocessing steps. 

For the modeling part, the project will begin with simple baseline models, such as a multilayer perceptron (MLP), to establish a reference performance. After that, a small convolutional neural network (CNN) will be implemented to better capture the spatial structure of the image data. All models will be implemented in PyTorch, and key hyperparameters like learning rate, batch size, and number of training epochs will be specified through configuration files instead of being hardcoded. Training and evaluation metrics will be logged using an experiment tracking tool, which makes it easier to compare different model setups and data versions throughout the project.

Dataset link: https://www.kaggle.com/datasets/zalando-research/fashionmnist



## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```




# Setup Project

First clone project from github, the run these:

```bash
pip install dvc # Remember to have docker installed and ready

pip install dvc-gdrive

dvc pull # Pulls data
```

Then train model:

```bash
docker-compose run --rm train                                               
docker-compose run --rm evaluate
docker-compose up api
```
