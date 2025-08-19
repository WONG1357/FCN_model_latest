import torch
from torch.utils.data import Dataset
import numpy as np

class CustomNumpyDataset(Dataset):
    """Dataset for loading .npy files, handles grayscale images and squeezes masks."""
    def __init__(self, images_path, masks_path):
        print(f"Loading data from: {images_path}")
        self.images = np.load(images_path)
        print(f"Loading masks from: {masks_path}")
        self.masks = np.load(masks_path)

        print(f"-> Loaded images shape: {self.images.shape}")
        print(f"-> Loaded masks shape: {self.masks.shape}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load raw data
        image = self.images[idx].astype(np.float32)
        mask = self.masks[idx]

        # --- ESSENTIAL STRUCTURAL TRANSFORMATIONS ---
        # Squeeze mask from (H, W, 1) to (H, W)
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = np.squeeze(mask, axis=-1)

        # Set mask data type for PyTorch loss function
        mask = mask.astype(np.int64)

        # Handle image channels
        if image.ndim == 2:  # Grayscale (H, W) -> (3, H, W)
            image = np.stack([image] * 3, axis=0)
        elif image.ndim == 3 and image.shape[-1] == 1:  # Grayscale (H, W, 1) -> (3, H, W)
            image = image.transpose(2, 0, 1)
            image = np.concatenate([image] * 3, axis=0)
        elif image.ndim == 3 and image.shape[-1] == 3:  # RGB (H, W, 3) -> (3, H, W)
            image = image.transpose(2, 0, 1)

        # Convert to PyTorch tensors
        image_tensor = torch.from_numpy(image.copy())
        mask_tensor = torch.from_numpy(mask.copy())

        return image_tensor, mask_tensor