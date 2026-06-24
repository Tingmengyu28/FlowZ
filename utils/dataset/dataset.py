import torch
import numpy as np
import cv2
from torch.utils.data import Dataset


class MicroscopyDeepZDataset(Dataset):
    """
    Dataset class for reading paired microscopy images from txt files.
    Each line in the txt file contains: image1_path	image2_path	dpm_value
    Returns: (input_image, target_image, dpm_tensor)
    """
    
    def __init__(self, pairs_file_path, image_size=None, transform=None):
        """
        Initialize the dataset.
        
        Args:
            pairs_file_path (str): Path to the txt file containing image pairs and dpm values
            image_size (tuple, optional): Size to resize images to (height, width)
            transform (callable, optional): Optional transform to be applied on images
        """
        self.pairs_file_path = pairs_file_path
        self.image_size = image_size
        self.transform = transform
        
        # Read the pairs file
        self.pairs_data = []
        with open(pairs_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split('\t')
                    if len(parts) == 3:
                        img1_path, img2_path, dpm_value = parts
                        self.pairs_data.append((img1_path, img2_path, float(dpm_value)))
            
    def __len__(self):
        """Return the number of image pairs."""
        return len(self.pairs_data)
    
    def __getitem__(self, idx):
        """
        Get a sample of paired images and depth parameter.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            tuple: (input_image, target_image, dpm_tensor)
                - input_image: torch.Tensor (C, H, W)
                - target_image: torch.Tensor (C, H, W)
                - dpm_tensor: torch.Tensor (1, H, W) - same spatial size as images
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img1_path, img2_path, dpm_value = self.pairs_data[idx]
        
        # Load input image as grayscale
        input_image = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        if input_image is None:
            raise FileNotFoundError(f"Input image not found: {img1_path}")
        
        # Load target image as grayscale
        target_image = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
        if target_image is None:
            raise FileNotFoundError(f"Target image not found: {img2_path}")
        
        # Resize images if needed
        if self.image_size:
            input_image = cv2.resize(input_image, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_LINEAR)
            target_image = cv2.resize(target_image, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_LINEAR)
        
        # Apply transforms
        if self.transform:
            # For grayscale, we need to add a channel dimension since transforms expect 3D tensors (C, H, W)
            input_image = np.expand_dims(input_image, axis=0)  # Shape becomes (1, H, W)
            target_image = np.expand_dims(target_image, axis=0)  # Shape becomes (1, H, W)
            # Convert to tensor and normalize
            input_image = torch.from_numpy(input_image).float() / 255.0
            target_image = torch.from_numpy(target_image).float() / 255.0
            # Apply transforms
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)
        else:
            # For grayscale, we need to add a channel dimension since tensors expect 3D tensors (C, H, W)
            input_image = np.expand_dims(input_image, axis=0)  # Shape becomes (1, H, W)
            target_image = np.expand_dims(target_image, axis=0)  # Shape becomes (1, H, W)
            # Convert to tensor and normalize to [0, 1] range
            input_image = torch.from_numpy(input_image).float() / 255.0
            target_image = torch.from_numpy(target_image).float() / 255.0
        
        # Ensure images are in [0, 1] range
        input_image = torch.clamp(input_image, 0, 1)
        target_image = torch.clamp(target_image, 0, 1)
        
        # Set requires_grad to False for images
        input_image.requires_grad = False
        target_image.requires_grad = False
        
        # Create dpm tensor with same spatial size as images
        _, height, width = input_image.shape
        # Create a tensor filled with the dpm_value that has the same spatial dimensions as the input
        dpm_tensor = torch.full((1, height, width), float(dpm_value), dtype=input_image.dtype, device=input_image.device)
        
        return input_image, target_image, dpm_tensor


if __name__ == "__main__":
    # Test the new MicroscopyDeepZDataset
    pairs_file = "/data1/azt/cv/recoverZ/data/pairs/train/ch2_train_pairs.txt"
    
    try:
        dataset = MicroscopyDeepZDataset(pairs_file, image_size=(256, 256))
        
        # Test getting a sample
        sample = dataset[0]
        input_img, target_img, dpm_tensor = sample
        
        print(f"Input image shape: {input_img.shape}")
        print(f"Target image shape: {target_img.shape}")
        print(f"DPM tensor shape: {dpm_tensor.shape}")
        print(f"DPM value: {dpm_tensor[0, 0, 0].item()}")
        print(f"Input image range: [{input_img.min().item():.3f}, {input_img.max().item():.3f}]")
        print(f"Target image range: [{target_img.min().item():.3f}, {target_img.max().item():.3f}]")
        
        # Test multiple samples
        print("\nTesting first 3 samples:")
        for i in range(min(3, len(dataset))):
            input_img, target_img, dpm_tensor = dataset[i]
            print(f"Sample {i}: Input {input_img.shape}, Target {target_img.shape}, DPM {dpm_tensor.shape}")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure the pairs file exists and the images are accessible.")
    except Exception as e:
        print(f"Unexpected error: {e}")
