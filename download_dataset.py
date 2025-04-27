import kaggle
import os
dataset_name = "masoudnickparvar/brain-tumor-mri-dataset"
download_path = "./brain-tumor-mri-dataset"

os.makedirs(download_path, exist_ok=True)
kaggle.api.dataset_download_files(dataset_name, path=download_path, unzip=True, quiet=False)

print(f"Dataset downloaded to: {download_path}")