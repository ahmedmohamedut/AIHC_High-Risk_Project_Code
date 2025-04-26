import base64
import json
import requests
import os
import random
from typing import List, Dict, Tuple
import time

class BrainTumorClassifier:
    def __init__(self, model_name="llava-med", host="http://localhost:11434"):
        """
        Initialize the brain tumor classifier using the LLaVA-Med model via Ollama.
        
        Args:
            model_name: Name of the model in Ollama
            host: URL of the Ollama API
        """
        self.model_name = model_name
        self.api_url = f"{host}/api/generate"
        self.tumor_types = ["glioma", "meningioma", "pituitary", "notumor"]
        
    def encode_image(self, image_path: str) -> str:
        """
        Encode an image to base64 string.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Base64 encoded string of the image
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def load_examples(self, examples_dir: str, tumor_type: str = None) -> List[Dict]:
        """
        Load example images with their classifications.
        
        Args:
            examples_dir: Directory containing example images
            tumor_type: If specified, only load examples of this tumor type
            
        Returns:
            List of dictionaries containing image paths and their classifications
        """
        examples = []
        
        # Expected directory structure:
        # examples_dir/
        #   glioma/
        #   meningioma/
        #   pituitary/
        #   notumor/
        
        for type_dir in os.listdir(examples_dir):
            if tumor_type and type_dir != tumor_type:
                continue
                
            if type_dir in self.tumor_types:
                type_path = os.path.join(examples_dir, type_dir)
                if os.path.isdir(type_path):
                    for img_file in os.listdir(type_path):
                        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            examples.append({
                                "path": os.path.join(type_path, img_file),
                                "classification": type_dir,
                                "filename": img_file
                            })
        
        return examples
    
    def get_explanation_for_type(self, tumor_type: str) -> str:
        """
        Get a standardized explanation for each tumor type.
        
        Args:
            tumor_type: Type of tumor
            
        Returns:
            Explanation text for the tumor type
        """
        explanations = {
            "glioma": "Glioma tumors originate in glial cells and typically show infiltrative growth patterns. They often appear as irregular, heterogeneous masses with varying enhancement patterns.",
            
            "meningioma": "Meningioma tumors arise from meningeal cells and typically appear as well-defined, extra-axial masses that show homogeneous enhancement. They often have a broad dural base and may show a dural tail sign.",
            
            "pituitary": "Pituitary tumors are centered in the sella turcica and can extend superiorly into the suprasellar cistern or laterally into the cavernous sinuses. They typically show homogeneous enhancement.",
            
            "notumor": "No tumor is present in this brain MRI scan. The brain tissues appear normal without evidence of mass effect, abnormal enhancement, or infiltrative growth."
        }
        
        return explanations.get(tumor_type, "No specific explanation available for this classification.")
    
    def create_few_shot_prompt(self, examples: List[Dict], num_shots: int) -> Tuple[str, List[str]]:
        """
        Create a prompt with few-shot examples.
        
        Args:
            examples: List of example images with classifications
            num_shots: Number of examples to include (2, 4, or 8)
            
        Returns:
            Tuple of (prompt text, list of base64 encoded images)
        """
        if num_shots > len(examples):
            raise ValueError(f"Requested {num_shots} examples but only {len(examples)} are available")
        
        # Randomly select examples while ensuring we have at least one of each type if possible
        selected_examples = []
        remaining_examples = examples.copy()
        
        # Try to get at least one example of each tumor type
        for tumor_type in self.tumor_types:
            type_examples = [ex for ex in remaining_examples if ex["classification"] == tumor_type]
            if type_examples:
                selected = random.choice(type_examples)
                selected_examples.append(selected)
                remaining_examples.remove(selected)
                
                # Break if we've selected enough examples
                if len(selected_examples) >= num_shots:
                    selected_examples = selected_examples[:num_shots]
                    break
        
        # If we still need more examples, randomly select from remaining
        while len(selected_examples) < num_shots and remaining_examples:
            selected = random.choice(remaining_examples)
            selected_examples.append(selected)
            remaining_examples.remove(selected)
            
        # Shuffle the examples
        random.shuffle(selected_examples)
        
        # Create the prompt
        prompt = "I'll show you some examples of brain MRI scans with their classifications.\n\n"
        
        encoded_images = []
        for i, example in enumerate(selected_examples, 1):
            encoded_image = self.encode_image(example["path"])
            encoded_images.append(encoded_image)
            
            classification = example["classification"]
            explanation = self.get_explanation_for_type(classification)
            
            prompt += f"Example {i}:\n"
            prompt += f"Classification: {classification}\n"
            prompt += f"Explanation: {explanation}\n\n"
        
        prompt += "Now I'll show you a new brain MRI scan. Please classify it as one of: glioma, meningioma, pituitary, or notumor. Then provide a detailed explanation for your decision, including relevant imaging features that support your classification."
        
        return prompt, encoded_images
    
    def classify_image(self, image_path: str, examples: List[Dict], num_shots: int = 4) -> Dict:
        """
        Classify a brain tumor image using few-shot learning.
        
        Args:
            image_path: Path to the image to be classified
            examples: List of example images with classifications
            num_shots: Number of examples to include (2, 4, or 8)
            
        Returns:
            Dictionary with classification results
        """
        # Create the few-shot prompt
        prompt, example_images = self.create_few_shot_prompt(examples, num_shots)
        
        # Encode the target image
        target_image = self.encode_image(image_path)
        
        # Combine all images for the request
        all_images = example_images + [target_image]
        
        # Prepare the API request
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "images": all_images,
            "stream": True
        }
        
        # Send the request
        print(f"Sending request to classify image: {os.path.basename(image_path)}")
        try:
            response = requests.post(self.api_url, json=payload, stream=True)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            return {"error": f"API request failed: {str(e)}"}
        
        # Process the streaming response
        full_response = ""
        for line in response.iter_lines():
            if line:
                json_line = json.loads(line)
                if 'response' in json_line:
                    full_response += json_line['response']
                    print(json_line['response'], end='', flush=True)
                    
        print("\n" + "-" * 50)
        
        # Extract classification result
        classification_result = {
            "full_response": full_response,
            "image_path": image_path,
            "num_shots": num_shots
        }
        
        # Try to extract the tumor type from the response
        for tumor_type in self.tumor_types:
            if tumor_type.lower() in full_response.lower():
                classification_result["detected_tumor_type"] = tumor_type
                break
                
        return classification_result

def main():
    """
    Run the brain tumor classifier with few-shot learning.
    """
    # model_name = "z-uo/llava-med-v1.5-mistral-7b_q8_0:latest"
    model_name = "rohithbojja/llava-med-v1.6:latest"
    # Initialize the classifier
    classifier = BrainTumorClassifier(model_name=model_name)
    
    # Set the paths
    examples_dir = "./brain-tumor-mri-dataset/Training"  # Directory with categorized examples
    test_image_path = "./brain-tumor-mri-dataset/Testing/pituitary/Te-pi_0276.jpg"  # Image to classify
    
    # Load example images
    all_examples = classifier.load_examples(examples_dir)
    print(f"Loaded {len(all_examples)} examples for few-shot learning")
    
    # Test with different numbers of shots
    for num_shots in [2, 4, 8]:
        print(f"\nTesting with {num_shots}-shot learning:")
        result = classifier.classify_image(
            image_path=test_image_path,
            examples=all_examples,
            num_shots=num_shots
        )
        
        print(f"Results for {num_shots}-shot learning:")
        print(f"Image: {os.path.basename(test_image_path)}")
        if "detected_tumor_type" in result:
            print(f"Detected tumor type: {result['detected_tumor_type']}")
        else:
            print("Could not determine the tumor type from the response")
        print("-" * 50)
        
        # Add a delay between requests to avoid overwhelming the API
        if num_shots < 8:
            print("Waiting before next request...")
            time.sleep(5)

if __name__ == "__main__":
    main()
