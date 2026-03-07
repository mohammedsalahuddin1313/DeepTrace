from typing import Dict, Optional

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image


class GradCAM:
    """
    Minimal Grad-CAM implementation for the spatial branch of the model.
    Targets the last convolutional feature map.
    """

    def __init__(self, model, target_module):
        self.model = model
        self.target_module = target_module
        self.gradients = None
        self.activations = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_module.register_forward_hook(forward_hook)
        self.target_module.register_full_backward_hook(backward_hook)

    def generate(
        self,
        spatial_tensor: torch.Tensor,
        freq_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for a single input image tensor (1, C, H, W).
        Returns a numpy array in [0,1] with shape (H, W).
        """
        self.model.eval()
        self.model.zero_grad()

        logits, _ = self.model(spatial_tensor, freq_tensor)
        if logits.ndim == 0:
            logits = logits.unsqueeze(0)

        if target_class is None:
            target = logits[0]
        else:
            raise NotImplementedError("Binary setting uses scalar logit; target_class not required.")

        target.backward()

        grads = self.gradients  # (N, C, H, W)
        activations = self.activations

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        return cam


def overlay_heatmap_on_image(heatmap, image, alpha=0.5):
    import numpy as np
    import matplotlib.cm as cm
    from PIL import Image

    # convert heatmap to numpy
    if not isinstance(heatmap, np.ndarray):
        try:
            heatmap = heatmap.detach().cpu().numpy()
        except:
            heatmap = np.array(heatmap)

    heatmap = np.squeeze(heatmap)

    # apply color map
    heatmap_color = cm.jet(heatmap)

    # ensure numpy array
    heatmap_color = np.array(heatmap_color)

    # keep RGB only
    heatmap_color = heatmap_color[:, :, :3]

    # convert original image
    image = np.array(image) / 255.0

    # resize heatmap if needed
    if heatmap_color.shape[:2] != image.shape[:2]:
        heatmap_color = np.array(
            Image.fromarray((heatmap_color * 255).astype(np.uint8)).resize(
                (image.shape[1], image.shape[0])
            )
        ) / 255.0

    overlay = alpha * heatmap_color + (1 - alpha) * image
    overlay = np.clip(overlay, 0, 1)

    return overlay