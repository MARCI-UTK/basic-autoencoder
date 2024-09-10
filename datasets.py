import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class ImgDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        # Format of annotations file is rows of [img_name, label]
        self.annotations = pd.read_csv(csv_path)
        # Path to data directory storing the images
        self.img_dir = img_dir
        # transforms object containing all the transforms to be performed on the img
        self.transform = transform

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Returns the full path to the image using the img name stored in the csv file at idx row, col 0
        img_name = os.path.join(self.img_dir, self.annotations.iloc[idx, 0])

        img = Image.open(img_name)

        # Returns the label to the specific image stored in the csv file at idx row, col 1
        label = int(self.annotations.iloc[idx, 1])

        # Transforms img, if any. Usually there are basic transforms to convert from PIL Image to normalized tensor
        if self.transform:
            img = self.transform(img)

        return img, label
    
class TextDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        # Format of annotations file is rows of [text, label]
        self.annotations = pd.read_csv(csv_path)
        # transforms object containing all the transforms to be performed on the img
        self.transform = transform
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Ensuring the text and label are correct, expected data types
        text = str(self.annotations.iloc[idx, 0])
        label = int(self.annotations.iloc[idx, 1])

        if self.transform:
            text = self.transform(text)

        return text, label
    