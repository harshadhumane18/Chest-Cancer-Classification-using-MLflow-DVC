# 🫁 Chest Cancer Classification using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13.0-orange.svg)](https://www.tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)](https://fastapi.tiangolo.com/)
[![DVC](https://img.shields.io/badge/DVC-Enabled-blueviolet.svg)](https://dvc.org/)
[![MLFlow](https://img.shields.io/badge/MLFlow-2.2.2-0194E2.svg)](https://mlflow.org/)

An end-to-end **Machine Learning Operations (MLOps)** project for binary classification of chest CT scan images to detect **Adenocarcinoma Cancer** using **VGG16 Transfer Learning**. This project demonstrates production-ready ML pipeline implementation with experiment tracking, version control, and a web-based inference API.

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [MLOps Pipeline](#-mlops-pipeline)
- [API Documentation](#-api-documentation)
- [Configuration](#-configuration)
- [Results](#-results)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [License](#-license)

## 🎯 Overview

This project implements a complete MLOps workflow for medical image classification, specifically designed to classify chest CT scan images into two categories:
- **Normal** - No signs of cancer
- **Adenocarcinoma Cancer** - Detected signs of lung adenocarcinoma

The solution leverages **Transfer Learning** with VGG16 architecture, ensuring efficient training with limited medical imaging data while maintaining high accuracy.

### Key Highlights

- ✅ **Production-Ready MLOps Pipeline** with DVC orchestration
- ✅ **Experiment Tracking** using MLFlow
- ✅ **RESTful API** built with FastAPI
- ✅ **Modern Web Interface** for real-time predictions
- ✅ **Modular & Scalable Architecture** following best practices
- ✅ **Comprehensive Logging** and error handling
- ✅ **Data Version Control** for reproducibility

## ✨ Features

### 1. **Automated ML Pipeline**
   - Automated data ingestion from Google Drive
   - Base model preparation with transfer learning
   - Training with data augmentation
   - Model evaluation and metrics tracking

### 2. **MLOps Integration**
   - **DVC (Data Version Control)**: Pipeline orchestration and dependency management
   - **MLFlow**: Experiment tracking, model versioning, and metrics logging
   - **YAML Configuration**: Centralized configuration management

### 3. **Web Application**
   - Interactive web interface for image upload
   - Real-time cancer detection predictions
   - Drag-and-drop image support
   - Training trigger from web interface

### 4. **Model Architecture**
   - **Base Model**: VGG16 (ImageNet weights)
   - **Transfer Learning**: Fine-tuned for medical imaging
   - **Image Preprocessing**: 224x224x3 normalization
   - **Data Augmentation**: Enabled for improved generalization

## 🛠 Tech Stack

### Deep Learning & ML
- **TensorFlow/Keras** (2.13.0) - Model development and training
- **VGG16** - Pre-trained CNN architecture
- **NumPy, Pandas** - Data manipulation

### MLOps & DevOps
- **DVC** - Data version control and pipeline orchestration
- **MLFlow** (2.2.2) - Experiment tracking and model registry
- **Python-Box** - Configuration management

### Web Framework
- **FastAPI** (0.104.1) - High-performance REST API
- **Uvicorn** - ASGI server
- **HTML/CSS/JavaScript** - Frontend interface

### Utilities
- **gdown** - Google Drive data download
- **PyYAML** - Configuration file parsing
- **tqdm** - Progress bars


## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git
- DVC (will be installed via requirements)

### Step 1: Clone the Repository

git clone https://github.com/your-username/DL_Image_Classification_MLFlow_DVC.git
cd DL_Image_Classification_MLFlow_DVC### Step 2: Create Virtual Environment

# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate### Step 3: Install Dependencies

pip install -r requirements.txt### Step 4: Install Package in Development Mode
h
pip install -e .### Step 5: Initialize DVC (Optional)

If you want to use DVC for pipeline execution:

dvc init
dvc remote add -d storage <your-remote-storage>## 💻 Usage

### Training the Model

#### Option 1: Using Python Script
ash
python main.pyThis will execute all pipeline stages sequentially:
1. Data Ingestion
2. Base Model Preparation
3. Model Training
4. Model Evaluation

#### Option 2: Using DVC Pipeline
sh
# Run entire pipeline
dvc repro

# Run specific stage
dvc repro prepare_base_model

# Run with updated parameters
dvc repro --force
### Running the Web Application
ash
python app.pyOr using uvicorn directly:
h
uvicorn app:app --host 0.0.0.0 --port 8080Access the web interface at: `http://localhost:8080`

### Making Predictions via API

#### Using cURL

curl -X POST "http://localhost:8080/predict" \
  -H "Content-Type: application/json" \
  -d '{"image": "<base64_encoded_image>"}'#### Using Python

import requests
import base64

# Read and encode image
with open("path/to/image.jpg", "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

# Make prediction
response = requests.post(
    "http://localhost:8080/predict",
    json={"image": encoded_image}
)

result = response.json()
print(result)## 🔄 MLOps Pipeline

### DVC Pipeline Stages

The pipeline is defined in `dvc.yaml` with the following stages:

1. **data_ingestion**
   - Downloads dataset from Google Drive
   - Extracts zip file
   - Output: `artifacts/data_ingestion/Chest-CT-Scan-data`

2. **prepare_base_model**
   - Loads VGG16 with ImageNet weights
   - Adds custom classification head
   - Output: `artifacts/prepare_base_model/`

3. **training**
   - Trains model with data augmentation
   - Saves trained model
   - Output: `artifacts/training/model.keras`

4. **evaluation**
   - Evaluates model on test set
   - Logs metrics to MLFlow
   - Output: `scores.json`

### MLFlow Integration

MLFlow tracks:
- **Parameters**: Image size, batch size, epochs, learning rate, etc.
- **Metrics**: Loss, accuracy
- **Artifacts**: Trained model files
- **Experiments**: Multiple runs for comparison

To view MLFlow UI:

mlflow uiAccess at: `http://localhost:5000`

## 📡 API Documentation

### Endpoints

#### 1. **GET /** - Home Page
Returns the web interface HTML.

**Response**: HTML content

#### 2. **POST /predict** - Image Prediction
Predicts cancer from uploaded image.

**Request Body**:n
{
  "image": "base64_encoded_image_string"
}**Response**:son
[
  {
    "image": "Normal"  // or "Adenocarcinoma Cancer"
  }
]#### 3. **POST /train** - Trigger Training
Initiates model training pipeline.

**Response**:n
{
  "message": "Training completed successfully!",
  "output": "training logs..."
}### Interactive API Docs

FastAPI provides automatic interactive documentation:
- Swagger UI: `http://localhost:8080/docs`
- ReDoc: `http://localhost:8080/redoc`

## ⚙️ Configuration

### Hyperparameters (`params.yaml`)

AUGMENTATION: True
IMAGE_SIZE: [224, 224, 3]
BATCH_SIZE: 16
INCLUDE_TOP: False
EPOCHS: 1
CLASSES: 2
WEIGHTS: imagenet
LEARNING_RATE: 0.01### Configuration File (`config/config.yaml`)

artifacts_root: artifacts

data_ingestion:
  root_dir: artifacts/data_ingestion
  source_URL: <google_drive_url>
  local_data_file: artifacts/data_ingestion/data.zip
  unzip_dir: artifacts/data_ingestion

prepare_base_model:
  root_dir: artifacts/prepare_base_model
  base_model_path: artifacts/prepare_base_model/base_model.keras
  updated_base_model_path: artifacts/prepare_base_model/base_model_updated.keras

training:
  root_dir: artifacts/training
  trained_model_path: artifacts/training/model.keras## 📊 Results


> **Note**: These metrics can be improved by:
> - Increasing training epochs
> - Adjusting learning rate
> - Using more data augmentation
> - Fine-tuning hyperparameters

### Model Architecture

- **Base Model**: VGG16 (ImageNet pre-trained)
- **Input Shape**: 224 × 224 × 3
- **Output**: Binary classification (2 classes)
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy

## 🔮 Future Improvements

- [ ] Improve model accuracy through hyperparameter tuning
- [ ] Add more data augmentation techniques
- [ ] Implement model versioning with MLFlow Model Registry
- [ ] Add unit tests and integration tests
- [ ] Deploy model using Docker containers
- [ ] Add CI/CD pipeline with GitHub Actions
- [ ] Implement model monitoring and drift detection
- [ ] Add support for batch prediction
- [ ] Create comprehensive test suite
- [ ] Add model explainability (Grad-CAM, SHAP)
- [ ] Support for multiple cancer types
- [ ] Add authentication and authorization to API

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Add docstrings to all functions and classes
- Write unit tests for new features
- Update documentation as needed

## 🙏 Acknowledgments

- VGG16 architecture by Visual Geometry Group, University of Oxford
- TensorFlow/Keras team for the excellent deep learning framework
- FastAPI for the modern web framework
- DVC and MLFlow communities for MLOps tools

## 📚 References

- [VGG16 Paper](https://arxiv.org/abs/1409.1556)
- [Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [DVC Documentation](https://dvc.org/doc)
- [MLFlow Documentation](https://www.mlflow.org/docs/latest/index.html)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

---

