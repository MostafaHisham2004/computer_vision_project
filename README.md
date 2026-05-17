# Multi-Model Vision AI Pipeline

A production-ready computer vision application combining Deep Learning models into a unified FastAPI backend and a high-performance Streamlit user interface. This repository deploys three distinct computer vision pipelines: Object Detection, Semantic Segmentation, and Image Classification.

---

## Interactive Jupyter Notebooks

These are the core research and training notebooks for the models used in this project. Click on any model below to open its corresponding training and development notebook directly in GitHub:

| Model Pipeline | Task Description | Link to Notebook |
| :--- | :--- | :--- |
| **YOLOv8** | Vehicle Detection (Cars, Trucks, Motorbikes, etc.) | [**`notebooks/Yolo/detection.ipynb`**](notebooks/Yolo/detection.ipynb) |
| **UNet / DeepLabV3** | Cityscapes Semantic Segmentation (Roads, Sidewalks, Sky, etc.) | [**`notebooks/UNet/segmentation-project-model.ipynb`**](notebooks/UNet/segmentation-project-model.ipynb) |
| **MobileNetV2 CNN** | Plant Disease Classification (38 disease categories) | [**`notebooks/CNN/cnn-model.ipynb`**](notebooks/CNN/cnn-model.ipynb) |

---

## File and Project Architecture

The workspace is structured to maintain a clear separation between research/development (notebooks) and production/serving (deployment):

```
computer_vision_project/
├── notebooks/                              # Deep learning model development & training
│   ├── Yolo/                               # YOLO Object Detection
│   │   ├── detection.ipynb                 # Jupyter Notebook: Dataset prep, training, and evaluation
│   │   ├── best.pt                         # Trained YOLO model weights
│   │   ├── classes.txt                     # List of class names for the YOLO detector
│   │   ├── data.yaml                       # YOLO configuration file for dataset paths
│   │   ├── eda.py                          # Exploratory Data Analysis helper script
│   │   └── prepare_dataset.py              # Script to structure images for training
│   ├── UNet/                               # UNet / DeepLabV3 Segmentation
│   │   ├── segmentation-project-model.ipynb # Jupyter Notebook: Image preprocessing and model training
│   │   └── best_deeplabv3_cityscapes.pth   # Pre-trained PyTorch DeepLabV3 weights
│   └── CNN/                                # CNN Plant Disease Classifier
│       ├── cnn-model.ipynb                 # Jupyter Notebook: CNN architecture and training loop
│       └── plant_disease_model.h5          # Saved Keras classification model weights
│
├── deployment/                             # Model serving and client application
│   ├── app/                                # FastAPI backend service
│   │   ├── __init__.py
│   │   ├── main.py                         # FastAPI application running inference endpoints
│   │   └── models.py                       # Model wrapper classes (Ultralytics, PyTorch, TensorFlow)
│   ├── dashboard.py                        # Streamlit web dashboard client
│   └── requirements.txt                    # Project package dependencies
│
└── README.md                               # Project documentation (this file)
```

---

## Installation and Setup

Follow these steps to configure the environment and run the multi-model pipeline locally.

### 1. Clone the Repository
```bash
git clone https://github.com/MostafaOsmanFathi/computer_vision_project.git
cd computer_vision_project
```

### 2. Set Up Virtual Environment
Create a virtual environment to ensure Python package isolation and prevent library conflicts:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
Install all required libraries, including FastAPI, Streamlit, PyTorch, TensorFlow, and Ultralytics:
```bash
pip install -r deployment/requirements.txt
```

### 4. Configure Model Weights
Ensure your pre-trained weight files (`best.pt`, `best_deeplabv3_cityscapes.pth`, `plant_disease_model.h5`) are placed under their respective directory paths inside the `notebooks` folder as structured in the [File Architecture](#file-and-project-architecture) section.

> [!NOTE]  
> The backend searches for weights in the specified directories. If a weight file is missing at startup, the system will log a warning and let the remaining active services run (with the exception of critical UNet components).

---

## Running the Project

The application operates in a client-server architecture: a high-performance FastAPI backend running model inference, and a Streamlit dashboard serving as the frontend user interface.

### Step 1: Start the FastAPI Backend Server
Start the backend API using `uvicorn`. The server handles image upload requests and processes model inference.

```bash
# Navigate to the deployment folder and launch the backend
cd deployment
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
*The interactive FastAPI Swagger documentation is accessible at: http://localhost:8000/docs*

### Step 2: Start the Streamlit Dashboard
Open a new terminal window, activate the virtual environment, and launch the Streamlit frontend client.

```bash
# Run the dashboard script from the deployment directory
cd deployment
streamlit run dashboard.py
```
*The web interface will automatically open in your default browser at: http://localhost:8501*

---

## Model Pipeline Details

### YOLOv8 Vehicle Detection
- **Task:** Detect and count vehicles in input imagery.
- **Classes:** `car`, `threewheel`, `bus`, `truck`, `motorbike`, `van`.
- **API Endpoint:** `POST /predict/yolo`
- **Notebook:** [**`notebooks/Yolo/detection.ipynb`**](notebooks/Yolo/detection.ipynb)

### DeepLabV3 Semantic Segmentation
- **Task:** Pixel-level road and urban semantic segmentation.
- **Classes:** 19 classes based on Cityscapes (Road, Sidewalk, Vegetation, Sky, Building, etc.).
- **API Endpoint:** `POST /predict/unet`
- **Notebook:** [**`notebooks/UNet/segmentation-project-model.ipynb`**](notebooks/UNet/segmentation-project-model.ipynb)

### MobileNetV2 CNN Plant Disease Classifier
- **Task:** Classify leaf health and specific crop diseases.
- **Classes:** 38 distinct crop-disease combinations.
- **API Endpoint:** `POST /predict/cnn`
- **Notebook:** [**`notebooks/CNN/cnn-model.ipynb`**](notebooks/CNN/cnn-model.ipynb)

> [!IMPORTANT]  
> **Resource Management Optimization:** The backend implements isolated TensorFlow memory allocation patterns (`set_memory_growth`) alongside optimized PyTorch evaluation states to run PyTorch, TensorFlow, and Ultralytics models concurrently without resource contention or GPU Out-of-Memory (OOM) failures.

---
*Created and maintained by Mostafa Osman.*