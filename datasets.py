"""
This module contains custom PyTorch Dataset classes for loading and transforming images from directories.
"""
import os
from torch.utils.data import Dataset
from PIL import Image
import config

class Content2PaintingDataset(Dataset):
    """
    Content2PaintingDataset is a custom PyTorch Dataset for loading and transforming images
    from two directories: one containing content images and the other containing painting images. 
    It supports optional transformations for training and testing phases.
    """
    def __init__(self, content_dir, painting_dir, is_test=False, transform=True):
        self.transform = transform
        self.content_data = os.listdir(content_dir)
        self.paintings_data = os.listdir(painting_dir)
        self.content_dir = content_dir
        self.painting_dir = painting_dir
        self.is_test = is_test

    def read_image(self, class_name, idx):
        """
        Reads an image from the specified class and index, applies transformations if necessary, and returns the image.
        """
        images_classes = {
            'content': (self.content_data, self.content_dir),
            'painting': (self.paintings_data, self.painting_dir)
        }

        files_class, images_dir = images_classes.get(class_name)
        image_name = files_class[idx]
        image_path = os.path.join(images_dir, image_name)

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = config.IMG_TRANSFORMS(image) if not self.is_test else config.TEST_TRANSFORMS(image)

        return image

    def __len__(self):
        return min(len(self.content_data), len(self.paintings_data))

    def __getitem__(self, index):
        photo_img = self.read_image('content', index)
        painting_img = self.read_image('painting', index)

        return photo_img, painting_img

class ImageFolderDataset(Dataset):
    """
    A custom dataset class for loading images from a directory.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_paths = [
            os.path.join(root_dir, img_name) for img_name in os.listdir(root_dir)
            if img_name.lower().endswith(('png', 'jpg', 'jpeg'))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            with Image.open(img_path).convert("RGB") as img:
                img = self.transform(img)
            return img
        except (FileNotFoundError, OSError, IOError) as e:
            print(f"Error al cargar la imagen {img_path}: {e}")
            return None
