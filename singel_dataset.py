import os
import re
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader

def extract_number(filename):
    match = re.match(r"(\d+)", filename)
    return int(match.group(1)) if match else float('inf')

class SingleFolderImageDataset(Dataset):
    def __init__(self, dataset_path, image_size=512, transform=None):
        """
        Args:
            dataset_path (str): The directory path containing images.
            image_size (int): Target image size for resizing.
            transform (callable, optional): Transform function for image preprocessing.
        """
        all_files = os.listdir(dataset_path)
        image_files = [f for f in all_files if f.lower().endswith((".png", ".jpg"))]
        image_paths = [os.path.join(dataset_path, image_file) for image_file in image_files]
        self.image_paths = sorted(
        image_paths,
        key=lambda path: extract_number(os.path.basename(path))
        )
        self.prompt = os.path.basename(os.path.normpath(dataset_path))
        
        if transform is None:
            self.transform = T.Compose([
                T.Resize((image_size, image_size),interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize([0.5], [0.5]),
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return {"image": image, "prompt": self.prompt}
