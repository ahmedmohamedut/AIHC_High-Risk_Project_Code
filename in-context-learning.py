import torch
from PIL import Image
from transformers import LlavaForConditionalGeneration, LlavaProcessor

# Load model in 8-bit to fit 3060 GPU
model = LlavaForConditionalGeneration.from_pretrained(
    "microsoft/llava-med-v1.5-mistral-7b",
    device_map="auto",
    load_in_8bit=True  # Use 4-bit if 8-bit doesn't fit
)

processor = LlavaProcessor.from_pretrained("microsoft/llava-med-v1.5-mistral-7b")

# Replace with paths to your MRI image files
image_paths = [
    ("tumor_mri_1.jpg", "Tumor"),
    ("tumor_mri_2.jpg", "Tumor"),
    ("normal_mri_1.jpg", "No Tumor"),
]

# Prepare few-shot prompt
few_shot_prompt = ""
for idx, (img_path, label) in enumerate(image_paths):
    few_shot_prompt += f"Image {idx+1}: This is an MRI scan. Does it show a brain tumor?\nAnswer: {label}\n"

# New image to classify
test_image_path = "test_mri.jpg"
image = Image.open(test_image_path).convert("RGB")

# Add the query for the test image
prompt = few_shot_prompt + f"Image {len(image_paths)+1}: This is an MRI scan. Does it show a brain tumor?\nAnswer:"

# Preprocess image + prompt
inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")

# Generate output
generate_ids = model.generate(**inputs, max_new_tokens=20)
output = processor.batch_decode(generate_ids, skip_special_tokens=True)[0]

print("\n🧠 Prediction:", output.strip())
