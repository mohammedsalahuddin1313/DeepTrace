from typing import Tuple

import numpy as np
from PIL import Image


def compute_fft_image(img: Image.Image, size: int = 224) -> Image.Image:
    """
    Compute log-magnitude spectrum of 2D FFT and return as 3-channel PIL image.
    """
    img = img.convert("L")
    img = img.resize((size, size))
    arr = np.array(img).astype(np.float32)

    fft = np.fft.fft2(arr)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    magnitude = np.log1p(magnitude)

    magnitude -= magnitude.min()
    magnitude /= (magnitude.max() + 1e-8)
    magnitude_uint8 = (magnitude * 255).astype(np.uint8)

    mag_img = Image.fromarray(magnitude_uint8)
    mag_img = mag_img.convert("RGB")
    return mag_img

