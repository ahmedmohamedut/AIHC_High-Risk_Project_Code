import requests
import base64
import os
import csv
from pathlib import Path


def encode_image_base64(image_path):
    """Reads an image from disk and returns base64-encoded string."""
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    return base64.b64encode(image_bytes).decode('utf-8')


def build_advanced_few_shot_prompt(examples, test_question):
    """
    Build an advanced prompt with detailed instructions and examples

    examples: List of tuples [(image_description, features, label), ...]
    test_question: String for the final question
    """
    system_instruction = """You are a highly specialized medical imaging AI expert focused on brain MRI analysis. 
Your task is to classify brain MRI scans into four categories:
1. Glioma: Appears as irregular, infiltrative masses with surrounding edema, often with heterogeneous enhancement, may cross midline.
2. Meningioma: Well-circumscribed, extra-axial masses that displace brain tissue, typically homogeneous, often with dural attachment.
3. Pituitary Tumor: Located in the sellar/suprasellar region, often with homogeneous enhancement, can cause compression of optic chiasm.
4. No Tumor: Normal brain anatomy with no abnormal enhancement or mass effect.

For each image, carefully analyze the scan features including location, shape, enhancement pattern, and effect on surrounding tissue.
Provide a confident diagnosis based solely on the imaging characteristics visible in the scan."""

    prompt = f"{system_instruction}\n\n"

    # Add detailed examples
    for idx, (description, features, label) in enumerate(examples, start=1):
        prompt += f"[MRI Scan {idx}]: {description}\n"
        prompt += f"Key Features: {features}\n"
        prompt += f"Diagnosis: {label}\n\n"

    # Add the test question with analysis request
    prompt += f"[MRI Scan {len(examples) + 1}]: {test_question}\n"
    prompt += "First, describe the key features you observe in this scan. "
    prompt += "Then provide your diagnosis from the four categories (Glioma, Meningioma, Pituitary Tumor, or No Tumor).\n"
    prompt += "Provide a confidence score (1-10). \n"
    prompt += "Explain your reasoning for the classification, specifically identifying visual features that influenced your decision \n"
    prompt += "Diagnosis:"

    return prompt


def send_request_to_ollama(model_name, prompt, encoded_images, temperature=0.1):
    """
    Sends the images + prompt to Ollama API and returns the response.
    Lower temperature for more confident predictions.
    """
    url = 'http://localhost:11434/api/generate'
    payload = {
        "model": model_name,
        "prompt": prompt,
        "images": encoded_images,
        "stream": False,
        "temperature": temperature,  # Lower temperature for medical diagnosis
        "num_predict": 512  # Ensure enough tokens for detailed reasoning
    }
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()["response"]


def classify_mri_scan(model_name, test_image_path, example_image_paths, examples, temperature=0.1):
    """
    Classifies a single MRI scan image
    """
    # Encode all images
    all_image_paths = example_image_paths + [test_image_path]
    encoded_images = [encode_image_base64(p) for p in all_image_paths]

    # Build the prompt with advanced instructions
    test_question = """This is a brain MRI scan. Analyze the image carefully and determine if this shows 
    a Glioma, Meningioma, Pituitary Tumor, or No Tumor (normal brain)."""

    prompt = build_advanced_few_shot_prompt(examples, test_question)

    # Send to Ollama with lower temperature for more confident prediction
    model_response = send_request_to_ollama(model_name, prompt, encoded_images, temperature)

    # Extract the final diagnosis
    final_diagnosis = None
    for category in ["Glioma", "Meningioma", "Pituitary Tumor", "No Tumor"]:
        if category.lower() in model_response.lower():
            final_diagnosis = category
            break

    return model_response, final_diagnosis


def main():
    # === Settings ===
    model_name = "rohithbojja/llava-med-v1.6:latest"  # Medical-specific LLaVA model
    csv_file_path = "misclassified_images.csv"  # Path to the CSV file

    # Define paths - update these to your actual file paths
    data_dir = Path("./mri_data")  # Base directory for MRI images

    # === Example images and labels with detailed descriptions ===
    examples = [
        (
            "Axial brain MRI scan showing an irregular mass in the left temporal lobe.",
            "Irregular borders, surrounding edema, heterogeneous enhancement, mass effect with midline shift.",
            "Glioma"
        ),
        (
            "Axial brain MRI scan showing a well-defined extra-axial mass attached to the dura.",
            "Well-circumscribed, homogeneous enhancement, broad dural base, displacement of adjacent brain tissue without invasion.",
            "Meningioma"
        ),
        (
            "Axial brain MRI scan showing a mass in the sellar/suprasellar region.",
            "Homogeneous enhancement, located in the pituitary fossa, possible superior extension, compression of surrounding structures.",
            "Pituitary Tumor"
        ),
        (
            "Axial brain MRI scan showing normal brain anatomy.",
            "Normal ventricle size, no midline shift, no abnormal enhancement, symmetric structures, normal gray-white matter differentiation.",
            "No Tumor"
        ),
    ]

    # === Paths to example images ===
    example_image_paths = [
        data_dir / "train/Tr-gl_0046.jpg",
        data_dir / "train/Tr-me_0508.jpg",
        data_dir / "train/Tr-pi_0121.jpg",
        data_dir / "train/Tr-no_0132.jpg",
    ]

    # Ensure example paths exist
    for path in example_image_paths:
        if not os.path.exists(path):
            print(f"Error: Example image path does not exist: {path}")
            return

    # Read test paths from CSV and perform classification
    results = []

    try:
        with open(csv_file_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)

            if 'file_path' not in reader.fieldnames:
                print(f"Error: CSV file must contain 'file_path' column. Found columns: {reader.fieldnames}")
                return

            for row in reader:
                file_path = row['file_path']
                test_image_path = Path(file_path)

                print(f"\n=== Processing: {file_path} ===")

                if not os.path.exists(test_image_path):
                    print(f"Warning: Image path does not exist: {test_image_path}")
                    results.append({
                        'file_path': file_path,
                        'diagnosis': 'File not found',
                        'details': 'Image file does not exist'
                    })
                    continue

                print("Sending request to Ollama...")
                model_response, diagnosis = classify_mri_scan(
                    model_name,
                    test_image_path,
                    example_image_paths,
                    examples
                )

                print("\n=== Classification Result ===")
                print(f"File: {file_path}")
                print(f"Diagnosis: {diagnosis if diagnosis else 'Uncertain'}")
                print("\n=== Model's Full Response ===")
                print(model_response)

                # Store the results
                results.append({
                    'file_path': file_path,
                    'diagnosis': diagnosis if diagnosis else 'Uncertain',
                    'details': model_response
                })

        # Write results to output CSV
        output_file = "mri_classification_results.csv"
        with open(output_file, 'w', newline='') as csvfile:
            fieldnames = ['file_path', 'diagnosis', 'details']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for result in results:
                writer.writerow(result)

        print(f"\n=== All results saved to {output_file} ===")

    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_file_path}")
    except Exception as e:
        print(f"Error during processing: {str(e)}")


if __name__ == "__main__":
    main()