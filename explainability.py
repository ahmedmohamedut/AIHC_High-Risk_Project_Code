import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
import cv2
from captum.attr import IntegratedGradients, GuidedGradCam, LayerGradCam, LayerAttribution
from captum.attr import visualization as viz
import shap

class GradCAM:
    """
    Grad-CAM implementation for CNN visualization
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.hooks = []
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
        
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
            
        # Register the hooks
        forward_handle = self.target_layer.register_forward_hook(forward_hook)
        backward_handle = self.target_layer.register_backward_hook(backward_hook)
        
        self.hooks = [forward_handle, backward_handle]
        
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
            
    def __call__(self, input_tensor, target_class=None):
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)
        
        # If target class is None, use the predicted class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # One-hot encode the target class
        target = output.clone()
        target[:] = 0
        target[:, target_class] = 1
        
        # Backward pass
        self.model.zero_grad()
        output.backward(gradient=target, retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients
        activations = self.activations
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        
        # ReLU and normalization
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        
        # Normalize
        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        
        return cam.squeeze().cpu().numpy()

class GradCAMPlusPlus:
    """
    Grad-CAM++ implementation for CNN visualization
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.hooks = []
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
        
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
            
        # Register the hooks
        forward_handle = self.target_layer.register_forward_hook(forward_hook)
        backward_handle = self.target_layer.register_backward_hook(backward_hook)
        
        self.hooks = [forward_handle, backward_handle]
        
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
            
    def __call__(self, input_tensor, target_class=None):
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)
        
        # If target class is None, use the predicted class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # One-hot encode the target class
        target = output.clone()
        target[:] = 0
        target[:, target_class] = 1
        
        # Backward pass
        self.model.zero_grad()
        output.backward(gradient=target, retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients
        activations = self.activations
        
        # Grad-CAM++ weights calculation
        alphas = torch.zeros_like(gradients)
        for i in range(gradients.shape[0]):
            # First derivative
            numerator = gradients[i].pow(2)
            # Second derivative
            denominator = 2 * gradients[i].pow(2)
            denominator += (activations[i] * gradients[i].pow(3)).sum(dim=[1, 2], keepdim=True)
            denominator = torch.clamp(denominator, min=1e-8)
            
            alpha = numerator / denominator
            alpha_norm = alpha / alpha.sum(dim=[1, 2], keepdim=True)
            alphas[i] = alpha_norm
        
        # Weighted combination of activation maps
        weights = torch.sum(alphas * F.relu(gradients), dim=[2, 3], keepdim=True)
        
        # Apply weights to activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        
        # ReLU and normalization
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        
        # Normalize
        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        
        return cam.squeeze().cpu().numpy()

def preprocess_image(img_path, transform=None):
    """
    Preprocess an image for model input
    """
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    return img_tensor, img

def overlay_heatmap(heatmap, img, colormap=cv2.COLORMAP_JET, alpha=0.5):
    """
    Overlay a heatmap on an image
    """
    # Convert the heatmap to RGB using specified colormap
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    
    # Convert original image to RGB if it's not
    if isinstance(img, Image.Image):
        img = np.array(img)
    
    # Ensure img is RGB (3 channels)
    if len(img.shape) == 2:  # Grayscale
        img = np.stack([img, img, img], axis=2)
    
    # Resize heatmap if needed
    if img.shape[:2] != heatmap.shape[:2]:
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Convert to same datatype (uint8)
    img = np.uint8(img)
    
    # Overlay heatmap on original image
    overlaid = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    
    return overlaid

def run_captum_integrated_gradients(model, input_tensor, target_class, n_steps=50):
    """
    Run Integrated Gradients from Captum
    """
    ig = IntegratedGradients(model)
    attributions = ig.attribute(input_tensor, target=target_class, n_steps=n_steps)
    return attributions.squeeze().cpu().detach().numpy(), None

def run_captum_guided_gradcam(model, input_tensor, target_class, layer=None):
    """
    Run Guided GradCAM from Captum
    """
    if layer is None:
        # Use the last convolutional layer by default
        # Assuming the model is our BrainTumorCNN
        layer = model.conv5
    
    guided_gc = GuidedGradCam(model, layer)
    attributions = guided_gc.attribute(input_tensor, target=target_class)
    return attributions.squeeze().cpu().detach().numpy(), None

def run_shap_deep_explainer(model, input_tensor, background_tensors, target_class=None):
    """
    Run SHAP DeepExplainer
    """
    # Connect model output to softmax
    def predict(x):
        output = model(x)
        return output
    
    # Create explainer with background data
    explainer = shap.DeepExplainer(predict, background_tensors)
    
    # Get SHAP values
    shap_values = explainer.shap_values(input_tensor)
    
    # If target_class is None, return all class explanations
    if target_class is None:
        return shap_values, explainer
    else:
        return shap_values[target_class], explainer

def plot_explanation(img, explanation, method_name, true_class, pred_class, class_names, save_path=None):
    """
    Plot the explanation and original image side by side
    """
    # Convert explanation to appropriate format based on method
    if method_name == 'Grad-CAM' or method_name == 'Grad-CAM++':
        # Explanation is already a heatmap
        explanation_viz = overlay_heatmap(explanation, img)
    elif method_name == 'Integrated Gradients':
        # Sum over channels for visualization
        attr_sum = np.sum(np.abs(explanation), axis=0)
        # Normalize for visualization
        attr_sum = (attr_sum - attr_sum.min()) / (attr_sum.max() - attr_sum.min() + 1e-8)
        explanation_viz = overlay_heatmap(attr_sum, img)
    elif method_name == 'SHAP':
        # Sum over channels for visualization
        attr_sum = np.sum(np.abs(explanation), axis=0)
        # Normalize for visualization
        attr_sum = (attr_sum - attr_sum.min()) / (attr_sum.max() - attr_sum.min() + 1e-8)
        explanation_viz = overlay_heatmap(attr_sum, img)
    else:
        explanation_viz = img  # Fallback
    
    # Create plot
    plt.figure(figsize=(12, 5))
    
    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title(f"Original\nTrue: {class_names[true_class]}, Pred: {class_names[pred_class]}")
    plt.axis('off')
    
    # Explanation
    plt.subplot(1, 2, 2)
    if isinstance(explanation_viz, np.ndarray):
        plt.imshow(explanation_viz)
    else:
        plt.imshow(explanation_viz)
    plt.title(f"{method_name} Explanation")
    plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=200)
        print(f"Saved explanation to {save_path}")
    
    plt.show()

def analyze_misclassified(model, test_loader, device, class_names, num_samples=10, save_dir='explanations'):
    """
    Analyze misclassified images using multiple explainability techniques
    """
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Set model to evaluation mode
    model.eval()
    
    # Collect misclassified samples
    misclassified = []
    
    # Get test transforms for preprocessing
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Get a few random background samples for SHAP
    background_samples = []
    for images, _ in test_loader:
        background_samples.append(images[:5].to(device))
        if len(background_samples) >= 2:  # Get 10 background samples
            break
    background_tensors = torch.cat(background_samples)
    
    print("Collecting misclassified samples...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            # Find misclassified samples
            incorrect_mask = (predicted != labels)
            if incorrect_mask.any():
                for i in range(len(labels)):
                    if incorrect_mask[i]:
                        # Get image path from dataset
                        img_path = test_loader.dataset.dataframe.iloc[i, 0]
                        true_class = labels[i].item()
                        pred_class = predicted[i].item()
                        
                        misclassified.append({
                            'img_path': img_path,
                            'true_class': true_class,
                            'pred_class': pred_class
                        })
            
            if len(misclassified) >= num_samples:
                break
    
    if not misclassified:
        print("No misclassified samples found!")
        return
    
    print(f"Found {len(misclassified)} misclassified samples. Analyzing...")
    
    # Set up explainers
    # Use the last convolutional layer for Grad-CAM
    target_layer = model.conv5
    gradcam = GradCAM(model, target_layer)
    gradcam_pp = GradCAMPlusPlus(model, target_layer)
    
    # Analyze each misclassified sample
    for i, sample in enumerate(misclassified[:num_samples]):
        img_path = sample['img_path']
        true_class = sample['true_class']
        pred_class = sample['pred_class']
        
        print(f"\nAnalyzing sample {i+1}/{min(num_samples, len(misclassified))}")
        print(f"Image: {os.path.basename(img_path)}")
        print(f"True class: {class_names[true_class]}, Predicted class: {class_names[pred_class]}")
        
        # Preprocess image
        input_tensor, original_img = preprocess_image(img_path, transform=test_transform)
        input_tensor = input_tensor.to(device)
        
        # Create subfolder for this sample
        sample_dir = os.path.join(save_dir, f"sample_{i+1}")
        os.makedirs(sample_dir, exist_ok=True)
        
        # Run Grad-CAM
        print("Running Grad-CAM...")
        cam = gradcam(input_tensor, target_class=pred_class)
        plot_explanation(
            original_img, 
            cam, 
            'Grad-CAM', 
            true_class, 
            pred_class, 
            class_names,
            save_path=os.path.join(sample_dir, 'gradcam.png')
        )
        
        # Run Grad-CAM++
        print("Running Grad-CAM++...")
        cam_pp = gradcam_pp(input_tensor, target_class=pred_class)
        plot_explanation(
            original_img, 
            cam_pp, 
            'Grad-CAM++', 
            true_class, 
            pred_class, 
            class_names,
            save_path=os.path.join(sample_dir, 'gradcam_pp.png')
        )
        
        # Run Integrated Gradients
        print("Running Integrated Gradients...")
        try:
            ig_attr, _ = run_captum_integrated_gradients(model, input_tensor, target_class=pred_class)
            plot_explanation(
                original_img, 
                ig_attr.transpose(1, 2, 0), 
                'Integrated Gradients', 
                true_class, 
                pred_class, 
                class_names,
                save_path=os.path.join(sample_dir, 'integrated_gradients.png')
            )
        except Exception as e:
            print(f"Error running Integrated Gradients: {e}")
        
        # Run SHAP
        print("Running SHAP...")
        try:
            shap_values, _ = run_shap_deep_explainer(
                model, 
                input_tensor, 
                background_tensors, 
                target_class=pred_class
            )
            plot_explanation(
                original_img, 
                shap_values.transpose(1, 2, 0), 
                'SHAP', 
                true_class, 
                pred_class, 
                class_names,
                save_path=os.path.join(sample_dir, 'shap.png')
            )
        except Exception as e:
            print(f"Error running SHAP: {e}")
    
    # Clean up
    try:
        gradcam.remove_hooks()
        gradcam_pp.remove_hooks()
    except:
        pass
    
    print(f"\nExplanation analysis complete. Results saved to {save_dir}")
