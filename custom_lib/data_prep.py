import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms
from torchvision.transforms import Compose, RandomRotation, RandomHorizontalFlip, Resize, CenterCrop, ToTensor, Normalize, GaussianBlur, ColorJitter
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, ConcatDataset, Subset
from sklearn.model_selection import train_test_split

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


def data_loader(data_path, train_transform, val_transform, train_prop, batch_size, seed):
    """
    Function to load the train, validation, and test datasets with appropriate transformations
    and return DataLoader objects for each.

    Parameters:
    - data_path (str): Path to the dataset directory containing the images organized in subdirectories by class.
    - train_transform (torchvision.transforms.Compose): Transformations to be applied to the training data (e.g., augmentation).
    - val_transform (torchvision.transforms.Compose): Transformations to be applied to the validation and test data.
    - train_prop (float): Proportion of the dataset to be used for training (e.g., 0.7 for 70% training data).
    - batch_size (int): The batch size to be used in the DataLoader for loading data in mini-batches.

    Returns:
    - train_loader (DataLoader): DataLoader object for the training set.
    - val_loader (DataLoader): DataLoader object for the validation set.
    - test_loader (DataLoader): DataLoader object for the test set.

    The DataLoader objects are created using `SubsetRandomSampler` based on the stratified splits
    of the dataset to ensure balanced class distribution across train, validation, and test sets.
    """

    torch.manual_seed(seed)  # For PyTorch

    # Load full dataset without transform (we'll assign it later)
    full_dataset = datasets.ImageFolder(root=data_path)
    
    # Create indices to represent all the samples in the dataset
    num_samples = len(full_dataset)
    indices = torch.arange(num_samples)

    # Extract class labels for stratification to ensure balanced splits
    class_labels = [full_dataset.targets[i] for i in indices]

    # Split indices into train, validation, and test sets (stratified sampling)
    # Train set will be of size `train_prop` (e.g., 0.7)
    train_indices, temp_indices = train_test_split(indices, train_size=train_prop, random_state=seed, stratify=class_labels)
    
    # The remaining `temp_indices` are split into validation and test sets with 50% for each
    val_indices, test_indices = train_test_split(temp_indices, train_size=0.5, random_state=seed, stratify=[class_labels[i] for i in temp_indices])

    # Create separate datasets with appropriate transformations for each set
    train_dataset = datasets.ImageFolder(root=data_path, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=data_path, transform=val_transform)
    test_dataset = datasets.ImageFolder(root=data_path, transform=val_transform)

    # Create DataLoaders using `SubsetRandomSampler`
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_indices))
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, sampler=SubsetRandomSampler(val_indices))
    test_loader = DataLoader(test_dataset, batch_size=batch_size * 2, sampler=SubsetRandomSampler(test_indices))

    # Print the sizes of each dataset
    print(f"Train size {len(train_indices)}. Val size {len(val_indices)}. Test size {len(test_indices)}.")

    return train_loader, val_loader, test_loader
