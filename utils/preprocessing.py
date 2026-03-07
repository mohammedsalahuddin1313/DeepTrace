import os
from typing import Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from config import cfg
from .fft_utils import compute_fft_image


class DeepfakeDataset(Dataset):
    """
    Dataset that returns both spatial RGB tensor and frequency-domain tensor
    for each image, along with binary label (0=real, 1=fake).
    """

    def __init__(self, root_dir: str, split: str = "train", transform=None):
        assert split in {"train", "val", "test"}

        self.root_dir = root_dir
        self.split = split

        self.real_dir = os.path.join(root_dir, "real")
        self.fake_dir = os.path.join(root_dir, "fake")

        self.samples = []
        for label, cls_dir in [(0, self.real_dir), (1, self.fake_dir)]:
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")):
                    self.samples.append(
                        (os.path.join(cls_dir, fname), label)
                    )

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in {self.real_dir} or {self.fake_dir}")

        # deterministic split into train / val / test
        g = torch.Generator().manual_seed(cfg.SEED)
        indices = torch.randperm(len(self.samples), generator=g).tolist()
        n_total = len(indices)
        n_train = int(n_total * cfg.TRAIN_RATIO)
        n_val = int(n_total * cfg.VAL_RATIO)
        # ensure all samples are used
        n_test = n_total - n_train - n_val

        train_indices = indices[:n_train]
        val_indices = indices[n_train : n_train + n_val]
        test_indices = indices[n_train + n_val :]

        if split == "train":
            self.indices = train_indices
        elif split == "val":
            self.indices = val_indices
        else:
            self.indices = test_indices

        if transform is None:
            if split == "train":
                self.transform = transforms.Compose(
                    [
                        transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                    ]
                )
            else:
                self.transform = transforms.Compose(
                    [
                        transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
                        transforms.ToTensor(),
                    ]
                )
        else:
            self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        real_idx = self.indices[idx]
        path, label = self.samples[real_idx]

        img = Image.open(path).convert("RGB")
        spatial = self.transform(img)

        freq_img = compute_fft_image(img, size=cfg.IMG_SIZE)
        freq_tensor = transforms.ToTensor()(freq_img)

        return spatial, freq_tensor, torch.tensor(label, dtype=torch.float32)

