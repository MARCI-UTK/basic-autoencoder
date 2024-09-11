import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from models import ConvEncoder, ConvDecoder
from datasets import ImgDataset


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

        self.train_csv_path = "data/test.csv"
        self.train_img_dir = "data/raw_data/"

        self.test_csv_path = "data/test.csv"
        self.test_img_dir = "data/raw_data/"

        self.transform = transforms.Compose([
            transforms.Resize((self.img_size,self.img_size)),
            transforms.ToTensor(),
        ])

        self.train_set = ImgDataset(csv_path=self.train_csv_path, img_dir=self.train_img_dir, transform=self.transform)
        self.test_set = ImgDataset(csv_path=self.test_csv_path, img_dir=self.test_img_dir, transform=self.transform)

        self.emb_dim = 1024
        self.encoder = ConvEncoder(img_size=self.img_size, channels=self.channels, emb_dim=self.emb_dim)

        # Initialize and pass dummy data through encoder to set shape_before_flatten variable
        dummy_data = torch.zeros(1, self.channels, self.img_size, self.img_size)
        dummy_output = self.encoder(dummy_data)
        shape_before_flat = self.encoder.shape_before_flatten

        self.decoder = ConvDecoder(emb_dim=self.emb_dim, shape_before_flattening=shape_before_flat, channels=self.channels)

        self.loss_function = nn.MSELoss()
        self.optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.lr)
