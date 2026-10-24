"""CelebA 48x48 sprite sheet data loader."""

import os
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import requests
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

IMAGE_URL = "https://zsc.github.io/widgets/celeba/48x48.png"
NUM_IMAGES = 30000
IMAGE_SIZE = 48
CACHE_DIR = Path("cache")
CACHE_FILE = CACHE_DIR / "celeba_48x48_sprites_v2.npy"


def download_sprite_sheet(url: str = IMAGE_URL) -> Image.Image:
    """Download the sprite sheet from URL."""
    print(f"Downloading sprite sheet from {url}...")
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    return Image.open(BytesIO(response.content))


def parse_sprite_sheet(sheet: Image.Image) -> np.ndarray:
    """Parse sprite sheet into individual images.
    
    Returns:
        Array of shape (NUM_IMAGES, 48, 48, 3) with values in [0, 255]
    """
    sheet_array = np.asarray(sheet.convert("RGB"))
    h, w = sheet_array.shape[:2]
    if h % IMAGE_SIZE or w % IMAGE_SIZE:
        raise ValueError("Sprite dimensions must be multiples of IMAGE_SIZE")
    rows, cols = h // IMAGE_SIZE, w // IMAGE_SIZE
    # Infer columns from the file: the CelebA source is 200 columns, not 100.
    images = sheet_array.reshape(rows, IMAGE_SIZE, cols, IMAGE_SIZE, 3)
    return images.transpose(0, 2, 1, 3, 4).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3)[:NUM_IMAGES].copy()


def ensure_cache() -> Path:
    """Ensure the cache file exists, download and parse if needed.
    
    Returns:
        Path to the cache file.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    if CACHE_FILE.exists():
        print(f"Cache file found: {CACHE_FILE}")
        return CACHE_FILE
    
    print("Cache file not found, downloading and parsing...")
    sheet = download_sprite_sheet()
    images = parse_sprite_sheet(sheet)
    
    print(f"Saving {len(images)} images to cache...")
    np.save(CACHE_FILE, images)
    print(f"Cache saved to {CACHE_FILE}")
    
    return CACHE_FILE


def load_cached_images() -> np.ndarray:
    """Load cached images as numpy array.
    
    Returns:
        Array of shape (NUM_IMAGES, 48, 48, 3) with values in [0, 255]
    """
    ensure_cache()
    return np.load(CACHE_FILE, mmap_mode="r")


class CelebASpritesDataset(Dataset):
    """CelebA 48x48 sprites dataset."""
    
    def __init__(self, data_limit: Optional[int] = None, transform=None):
        """
        Args:
            data_limit: Limit number of images (for quick testing)
            transform: Optional transform to apply
        """
        self.images = load_cached_images()
        if data_limit is not None:
            self.images = self.images[:min(data_limit, len(self.images))]
        if len(self.images) == 0:
            raise ValueError("data_limit must select at least one image")
        self.transform = transform
    
    def _default_transform(self):
        """Default transform: normalize to [-1, 1]."""
        return transforms.Compose([
            transforms.ToTensor(),  # Converts to [0, 1]
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # To [-1, 1]
        ])
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        img = self.images[idx]
        if self.transform is None:
            return torch.from_numpy(np.array(img, copy=True)).permute(2, 0, 1).float().div_(127.5).sub_(1)
        img_pil = Image.fromarray(img)
        return self.transform(img_pil)
    
    def get_pil_image(self, idx: int) -> Image.Image:
        """Get image as PIL Image without transform."""
        return Image.fromarray(self.images[idx])


def get_dataloader(
    batch_size: int = 128,
    data_limit: Optional[int] = None,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    generator=None,
) -> DataLoader:
    """Create DataLoader for CelebA sprites.
    
    Args:
        batch_size: Batch size
        data_limit: Limit number of images
        shuffle: Whether to shuffle
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for GPU transfer
    
    Returns:
        DataLoader
    """
    dataset = CelebASpritesDataset(data_limit=data_limit)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        generator=generator,
    )


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Denormalize from [-1, 1] to [0, 1]."""
    return (tensor + 1) / 2


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert tensor to PIL Image.
    
    Args:
        tensor: Tensor in [-1, 1] or [0, 1], shape (C, H, W) or (B, C, H, W)
    """
    if tensor.dim() == 4:
        # Batch: make grid
        tensor = tensor[:16]  # Limit to 16 for display
        grid = make_grid(tensor, nrow=4)
        tensor = grid
    
    if tensor.min() < 0:
        tensor = denormalize(tensor)
    
    tensor = torch.clamp(tensor, 0, 1)
    array = (tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(array)


def make_grid(tensor: torch.Tensor, nrow: int = 8) -> torch.Tensor:
    """Make a grid of images from a batch tensor.
    
    Args:
        tensor: (B, C, H, W)
        nrow: Number of images per row
    
    Returns:
        Grid tensor (C, H_total, W_total)
    """
    B, C, H, W = tensor.shape
    nrows = (B + nrow - 1) // nrow
    
    # Pad if needed
    pad = nrow * nrows - B
    if pad > 0:
        tensor = torch.cat([tensor, torch.zeros(pad, C, H, W, device=tensor.device)], dim=0)
    
    tensor = tensor.view(nrows, nrow, C, H, W)
    tensor = tensor.permute(2, 0, 3, 1, 4).contiguous()  # (C, nrows, H, nrow, W)
    tensor = tensor.view(C, nrows * H, nrow * W)
    
    return tensor
