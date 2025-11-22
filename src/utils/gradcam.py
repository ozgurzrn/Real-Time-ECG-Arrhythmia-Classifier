import torch
import torch.nn.functional as F
import numpy as np

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hooks
        self.target_layer.register_forward_hook(self.save_activations)
        self.target_layer.register_full_backward_hook(self.save_gradients)

    def save_activations(self, module, input, output):
        self.activations = output

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        self.model.eval()
        
        # Forward pass
        output = self.model(x)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
            
        # Backward pass
        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()
        
        # Generate heatmap
        gradients = self.gradients.data.cpu().numpy()[0] # (channels, length)
        activations = self.activations.data.cpu().numpy()[0] # (channels, length)
        
        weights = np.mean(gradients, axis=1) # Global Average Pooling over time
        
        cam = np.zeros(activations.shape[1], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        cam = np.maximum(cam, 0) # ReLU
        
        # Resize to input size (linear interpolation)
        # Input x shape: (1, 1, length) or (1, length)
        input_len = x.shape[-1]
        cam = np.interp(np.linspace(0, len(cam), input_len), np.arange(len(cam)), cam)
        
        # Normalize
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
        
        return cam, class_idx

if __name__ == "__main__":
    print("Grad-CAM utility ready.")
