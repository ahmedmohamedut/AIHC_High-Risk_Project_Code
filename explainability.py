import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from pytorch_grad_cam import GradCAMPlusPlus, EigenGradCAM, AblationCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


class ExplainabilityAnalyzer:
    """
    Class for applying various explainability methods to CNN models.

    Currently supports:
    - GradCAM++
    - EigenGradCAM
    - AblationCAM
    """

    def __init__(self, model, device, target_layer=None):
        """
        Initialize the explainability analyzer.

        Args:
            model: PyTorch model
            device: Device to run inference on ('cuda' or 'cpu')
            target_layer: Model layer to analyze (if None, will use last conv layer)
        """
        self.model = model
        self.device = device

        # If no target layer is provided, try to automatically find the last conv layer
        if target_layer is None:
            # For our BrainTumorCNN model, the last conv layer is conv5
            if hasattr(model, 'conv5'):
                self.target_layer = [model.conv5]
            else:
                # Find the last convolutional layer
                for name, module in reversed(list(model.named_modules())):
                    if isinstance(module, torch.nn.Conv2d):
                        self.target_layer = [module]
                        print(f"Automatically selected target layer: {name}")
                        break
                else:
                    raise ValueError("Could not automatically find a convolutional layer. Please specify target_layer.")
        else:
            self.target_layer = [target_layer]

    def preprocess_image(self, image_path, image_size=(224, 224), normalize=True):
        """
        Preprocess an image for model input.

        Args:
            image_path: Path to the image file
            image_size: Target size for resizing
            normalize: Whether to apply ImageNet normalization

        Returns:
            preprocessed_img: Tensor for model input
            orig_img: Original image as numpy array (for visualization)
        """
        img = Image.open(image_path).convert('RGB')

        # Save original image for visualization
        orig_img = np.array(img.resize(image_size)) / 255.0

        # Define preprocessing transforms
        if normalize:
            preprocess = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            preprocess = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor()
            ])

        # Apply preprocessing
        preprocessed_img = preprocess(img).unsqueeze(0).to(self.device)

        return preprocessed_img, orig_img

    def explain(self, image_path, method="gradcam++", true_label=None, pred_label=None,
                image_size=(224, 224), output_path=None, show=False):
        """
        Apply explainability method to analyze model's prediction.

        Args:
            image_path: Path to the image file
            method: Explainability method to use ('gradcam++', 'eigencam', or 'ablationcam')
            true_label: True class index (optional)
            pred_label: Predicted class index (optional)
            image_size: Image size for model input
            output_path: Path to save visualization (optional)
            show: Whether to display the visualization

        Returns:
            cam_image: Visualization with CAM overlay
            raw_cam: Raw activation map
        """
        # Preprocess image
        input_tensor, orig_img = self.preprocess_image(image_path, image_size)

        # Create CAM object based on method
        if method.lower() == "gradcam++":
            cam = GradCAMPlusPlus(model=self.model, target_layers=self.target_layer)
        elif method.lower() == "eigencam":
            cam = EigenGradCAM(model=self.model, target_layers=self.target_layer)
        elif method.lower() == "ablationcam":
            cam = AblationCAM(model=self.model, target_layers=self.target_layer)
        else:
            raise ValueError(f"Unsupported method: {method}. Use 'gradcam++', 'eigencam', or 'ablationcam'.")

        # Run the model to get predictions if not provided
        if pred_label is None:
            with torch.no_grad():
                outputs = self.model(input_tensor)
                _, pred_label = torch.max(outputs.data, 1)
                pred_label = pred_label.item()

        # Set target
        targets = [ClassifierOutputTarget(pred_label)]

        # Generate CAM
        raw_cam = cam(input_tensor=input_tensor, targets=targets)
        raw_cam = raw_cam[0, :]  # Take first image in batch

        # Create CAM visualization
        cam_image = show_cam_on_image(orig_img, raw_cam, use_rgb=True)

        # Create figure for visualization if output_path or show is True
        if output_path or show:
            plt.figure(figsize=(12, 4))

            # Original image
            plt.subplot(1, 3, 1)
            plt.imshow(orig_img)
            plt.title("Original Image")
            plt.axis('off')

            # CAM overlay
            plt.subplot(1, 3, 2)
            plt.imshow(cam_image)
            title = f"{method} Visualization"
            plt.title(title)
            plt.axis('off')

            # Raw heatmap
            plt.subplot(1, 3, 3)
            plt.imshow(raw_cam, cmap='jet')
            plt.title("Raw Activation Map")
            plt.axis('off')

            # Add title with labels info
            if true_label is not None and pred_label is not None:
                plt.suptitle(f"True: {true_label}, Predicted: {pred_label}", fontsize=14)

            plt.tight_layout()

            if output_path:
                plt.savefig(output_path, bbox_inches='tight')
                print(f"Visualization saved to {output_path}")

            if show:
                plt.show()
            else:
                plt.close()

        return cam_image, raw_cam

    def explain_batch(self, misclassified_df, method="gradcam++", output_dir="explainability_results",
                      image_size=(224, 224), label_map=None):
        """
        Apply explainability method to a batch of misclassified images.

        Args:
            misclassified_df: DataFrame with columns 'file_path', 'true_class', 'predicted_class'
            method: Explainability method to use
            output_dir: Directory to save visualizations
            image_size: Image size for model input
            label_map: Dictionary mapping class names to indices
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        # Create subdirectory for this method
        method_dir = os.path.join(output_dir, method)
        os.makedirs(method_dir, exist_ok=True)

        # If we have a label map, create inverse map for class indices
        inv_label_map = None
        if label_map:
            inv_label_map = {v: k for k, v in label_map.items()}

        print(f"Applying {method} to {len(misclassified_df)} misclassified images...")

        for i, row in misclassified_df.iterrows():
            file_path = row['file_path']
            true_class = row['true_class']
            pred_class = row['predicted_class']

            # Get numerical labels if label map is provided
            true_label = label_map[true_class] if label_map else None
            pred_label = label_map[pred_class] if label_map else None

            # Create output path
            img_filename = os.path.basename(file_path)
            base_name = os.path.splitext(img_filename)[0]
            output_path = os.path.join(method_dir, f"{base_name}_{true_class}_as_{pred_class}.png")

            # Apply explainability method
            try:
                self.explain(
                    image_path=file_path,
                    method=method,
                    true_label=true_label,
                    pred_label=pred_label,
                    image_size=image_size,
                    output_path=output_path,
                    show=False
                )

                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(misclassified_df)} images")

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

        print(f"Completed {method} analysis. Results saved to {method_dir}")