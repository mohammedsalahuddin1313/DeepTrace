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

    heatmap = np.asarray(heatmap)
    heatmap = np.squeeze(heatmap)

    # Normalize/reshape heatmap into (H, W)
    if heatmap.ndim == 0:
        heatmap = heatmap.reshape(1, 1)
    elif heatmap.ndim == 1:
        # Prefer reshaping to the image size if it matches exactly
        if isinstance(image, Image.Image):
            w, h = image.size
            if heatmap.size == h * w:
                heatmap = heatmap.reshape(h, w)
            else:
                side = int(round(np.sqrt(heatmap.size)))
                if side * side == heatmap.size:
                    heatmap = heatmap.reshape(side, side)
                else:
                    raise ValueError(
                        f"Heatmap is 1D with length {heatmap.size}, cannot reshape "
                        f"to image size ({h}x{w})."
                    )
        else:
            side = int(round(np.sqrt(heatmap.size)))
            if side * side == heatmap.size:
                heatmap = heatmap.reshape(side, side)
            else:
                raise ValueError(
                    f"Heatmap is 1D with length {heatmap.size}, and no PIL image "
                    "was provided to infer (H, W)."
                )
    elif heatmap.ndim != 2:
        raise ValueError(f"Expected heatmap with 2 dims (H, W), got shape {heatmap.shape}.")

    # clamp/normalize to [0, 1] for colormap stability
    heatmap = heatmap.astype(np.float32, copy=False)
    if np.isfinite(heatmap).any():
        hmin = np.nanmin(heatmap)
        hmax = np.nanmax(heatmap)
        if hmax > hmin:
            heatmap = (heatmap - hmin) / (hmax - hmin)
        else:
            heatmap = np.zeros_like(heatmap, dtype=np.float32)
    else:
        heatmap = np.zeros_like(heatmap, dtype=np.float32)

    # apply color map
    heatmap_color = cm.jet(heatmap)

    # ensure numpy array
    heatmap_color = np.array(heatmap_color)

    # keep RGB only
    if heatmap_color.ndim != 3 or heatmap_color.shape[2] < 3:
        raise ValueError(f"Unexpected colormap output shape: {heatmap_color.shape}")
    heatmap_color = heatmap_color[:, :, :3]

    # convert original image
    if isinstance(image, Image.Image):
        image = image.convert("RGB")
        image = np.asarray(image, dtype=np.float32) / 255.0
    else:
        image = np.asarray(image, dtype=np.float32)
        if image.max() > 1.0:
            image = image / 255.0
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = image[:, :, :3]

    # resize heatmap if needed
    if heatmap_color.shape[:2] != image.shape[:2]:
        heatmap_color = np.array(
            Image.fromarray((heatmap_color * 255).astype(np.uint8)).resize(
                (image.shape[1], image.shape[0])
            )
        ) / 255.0

    overlay = alpha * heatmap_color + (1 - alpha) * image
    overlay = np.clip(overlay, 0, 1)

    return Image.fromarray((overlay * 255).astype(np.uint8))