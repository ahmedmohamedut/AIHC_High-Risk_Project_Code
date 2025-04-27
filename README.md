# AIHC High-Risk Project Code Repository


## Model Architecture

## Clone repository
``
git clone https://github.com/ahmedmohamedut/AIHC_High-Risk_Project_Code.git
``

## Virtual environment
### Create a virtual environment
``
python3 -m venv .venv
``
### Activate virtual environment
#### Linux/MacOS
``
source .venv/bin/activate
``
#### Windows PowerShell
``
.\.venv\bin\activate.ps1
``


## Install prerequisites 
### Install pyTorch (with CUDA 12.8 Support)
``
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
``
### Install necessary libraries
``
pip install scikit-learn tensorboard opencv-python matplotlib seaborn pandas kaggle grad_cam
``

## Download dataset
``
python3 download_dataset.py
``


## Run training 
``
python3 run_training.py
``

## Run explainability methods
``
python3 run_explainability.py
``

## Install and run Vision LM
### Install Ollama
``
curl -fsSL https://ollama.com/install.sh | sh
``
### Run LlAVA-Med Vision Language Model
``
ollama run z-uo/llava-med-v1.5-mistral-7b_q8_0
``

## Run in-context reasoning
``
python3 in-context-tumor-classifier.py
``
