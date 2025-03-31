import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms
from torchvision.transforms import Compose, RandomRotation, RandomHorizontalFlip, Resize, CenterCrop, ToTensor, Normalize, GaussianBlur, ColorJitter
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, ConcatDataset, Subset
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class SkinLesionDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = os.path.expanduser(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_id = row['image_id']
        label = int(row['label'])

        # Construct full image path
        img_path = os.path.join(self.img_dir, f"{image_id}.jpg")
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

def data_transformation_pipeline(
    image_size,
    center_crop,
    normalize,
    rotate_angle = None,
    horizontal_flip_prob = None,
    gaussian_blur = None,
    brightness_contrast_range = None,
    is_train=False
):
    """
    Dynamically creates a data transformation pipeline.

    Args:
        image_size (int): Target size for resizing and cropping.
        rotate_angle (int, optional): Maximum rotation angle in degrees. Default is None (no rotation).
        horizontal_flip_prob (float, optional): Probability of horizontal flip. Default is None (no flipping).
        gaussian_blur (int, optional): Kernel size for Gaussian blur. Default is None (no blur).
        normalize (bool): Whether to include normalization. Default is False.
        is_train (bool): If True, include training-specific transformations. Default is False.

    Returns:
        torchvision.transforms.Compose: A composed transformation pipeline.
    """
    transform_steps = []

    # Basic resizing and cropping
    transform_steps.append(Resize(image_size))
    transform_steps.append(CenterCrop(center_crop))

    # Training-specific augmentations
    if is_train:
        if rotate_angle is not None:
            transform_steps.append(RandomRotation(degrees=rotate_angle))
        if horizontal_flip_prob is not None:
            transform_steps.append(RandomHorizontalFlip(p=horizontal_flip_prob))
        if gaussian_blur is not None:
            kernel_size, sigma = gaussian_blur 
            transform_steps.append(GaussianBlur(kernel_size=kernel_size,
                                                sigma=sigma))
        if brightness_contrast_range is not None:
            brightness_min, brightness_max, contrast_min, contrast_max = brightness_contrast_range
            transform_steps.append(ColorJitter(brightness=(brightness_min, brightness_max), contrast=(contrast_min, contrast_max)))

    # Convert to tensor
    transform_steps.append(ToTensor())

    # Normalization
    if normalize:
        transform_steps.append(Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    return Compose(transform_steps)


