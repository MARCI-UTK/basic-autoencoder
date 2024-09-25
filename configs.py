import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from transformers import AutoProcessor

from models import ConvEncoder, ConvDecoder, HFImageEncoder, HFImageDecoder
from datasets import ImgDataset, HFImageDataset

'''
    Configuration for autoencoder training with regular images
'''
class ImgAEConfig(object):
    def __init__(self):

        self.img_size = 256
        self.channels = 3

        self.epochs = 100
        self.lr = 0.0001
        self.batch_size = 64
        self.num_workers = 2
        
        self.emb_dim = 1024

        # Setting up train and validation dataloaders
        self.train_csv_path = "data/test.csv"
        self.train_img_dir = "data/raw_data/"

        self.valid_csv_path = "data/test.csv"
        self.valid_img_dir = "data/raw_data/"

        self.transform = transforms.Compose([
            transforms.Resize((self.img_size,self.img_size)),
            transforms.ToTensor(),
        ])

        self.train_set = ImgDataset(csv_path=self.train_csv_path, img_dir=self.train_img_dir, transform=self.transform)
        self.valid_set = ImgDataset(csv_path=self.valid_csv_path, img_dir=self.valid_img_dir, transform=self.transform)

        self.train_dataloader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True,
                                           num_workers=self.num_workers)
        self.valid_dataloader = DataLoader(self.valid_set, batch_size=self.batch_size, shuffle=False,
                                           num_workers=self.num_workers)

        # Initialize image encoder        
        self.encoder = ConvEncoder(img_size=self.img_size, channels=self.channels, emb_dim=self.emb_dim)

        # Initialize and pass dummy data through encoder to set shape_before_flatten variable
        dummy_data = torch.zeros(1, self.channels, self.img_size, self.img_size)
        dummy_output = self.encoder(dummy_data)
        shape_before_flat = self.encoder.shape_before_flatten

        # Initializing image decoder
        self.decoder = ConvDecoder(emb_dim=self.emb_dim, shape_before_flattening=shape_before_flat, channels=self.channels)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.lr)

class HFImageConfig(object):
    def __init__(self):

        self.epochs = 100
        self.lr = 0.0001
        self.batch_size = 64
        self.num_workers = 2
        
        self.emb_dim = 768

        # Setting up train and validation paths
        self.train_csv_path = "data/test.csv"
        self.train_img_dir = "data/raw_data/"

        self.valid_csv_path = "data/test.csv"
        self.valid_img_dir = "data/raw_data/"

        # Encoder and preprocessor
        self.encoder_name = "google/vit-base-patch16-224-in21k"
        self.encoder_is_trainable = True
        self.encoder = HFImageEncoder(model_name=self.encoder_name,
                                      is_trainable=self.encoder_is_trainable)
        self.preprocessor = AutoProcessor.from_pretrained(self.encoder_name, use_fast=True)

        # Decoder
        # This is purely for logging
        self.decoder_name = "HFImageDecoder"
        self.decoder = HFImageDecoder(emb_dim=self.emb_dim)

        # Train and validation datasets
        self.train_set = HFImageDataset(csv_path=self.train_csv_path, img_dir=self.train_img_dir,
                                        preprocess=self.preprocessor)
        self.valid_set = HFImageDataset(csv_path=self.valid_csv_path, img_dir=self.valid_img_dir,
                                        preprocess=self.preprocessor)

        # Train and validation dataloaders
        self.train_dataloader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True,
                                           num_workers=self.num_workers)
        self.valid_dataloader = DataLoader(self.valid_set, batch_size=self.batch_size, shuffle=False,
                                           num_workers=self.num_workers)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.lr)