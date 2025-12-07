import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt

class GrbasGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, input_tensor, item_idx=0, class_idx=None):

        logits = self.model(input_tensor)

        target_logits = logits[0, item_idx]
        
        if class_idx is None:
            class_idx = torch.argmax(target_logits).item()

        self.model.zero_grad()
        target_logits[class_idx].backward(retain_graph=True)

        gradients = self.gradients
        activations = self.activations
        weights = torch.mean(gradients, dim=(2, 3))[0]
        cam = torch.zeros(activations.shape[2:], dtype=torch.float32, device=input_tensor.device)
        
        for i, w in enumerate(weights):
            cam += w * activations[0, i, :, :]

        cam = cam.cpu().detach().numpy()

        if np.max(cam) - np.min(cam) > 1e-8:
            cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
        else:
            cam = np.zeros_like(cam)

        target_h, target_w = input_tensor.shape[2], input_tensor.shape[3]
        cam = cv2.resize(cam, (target_w, target_h))

        return cam, class_idx