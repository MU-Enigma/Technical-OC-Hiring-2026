from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from data.degradation import degrade_image
import os

class SuperResolutionDataset(Dataset):
    def __init__(self, image_dir, config, train = True):
        """
        Args:
            image_dir (str): Directory containing hr images
            config (dict): project config
            train (bool): is dataset being used for training or not
        """

        self.image_dir = image_dir
        self.config = config
        self.train = train

        self.image_files = sorted([
            file
            for file in os.listdir(image_dir)
            if file.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

        patch_size = config["dataset"]["hr_patch_size"]

        if self.train:
            self.crop = transforms.RandomCrop(patch_size)
        else:
            self.crop = transforms.CenterCrop(patch_size)

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """
        Returns:
            tuple:
                (lr_tensor, hr_tensor)
        """

        image_name = self.image_files[idx]

        image_path = os.path.join( self.image_dir, image_name)

        try:
            hr_image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to load image: {image_path}")from e

        hr_image = self.crop(hr_image)

        lr_image = degrade_image(
            hr_image,
            self.config
        )

        hr_tensor = self.to_tensor(hr_image)
        lr_tensor = self.to_tensor(lr_image)

        return lr_tensor, hr_tensor