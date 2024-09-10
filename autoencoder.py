import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets import ImgDataset
from models import ConvEncoder, ConvDecoder

class Pipeline():
    def __init__(self, img_size=256, channels=3, emb_dim=1024, epoch=100, lr=0.0001, batch_size=32, encoder=None, decoder=None, csv_path=None, img_dir=None):

        # Basic training hyperparameters
        self.epochs = epoch
        self.lr = lr
        self.batch_size = batch_size

        # Changing device to most powerful available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Setting encoder and decoder
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)

        # Setting up loss and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.lr)

        # Loading train dataset/loader
        transform = transforms.Compose([
            transforms.Resize((img_size,img_size)),
            transforms.ToTensor(),
            # transforms.Normalize(0.5, 0.5)
        ])

        self.csv_path = csv_path
        self.img_dir = img_dir
        
        self.train_set = ImgDataset(csv_path=self.csv_path, img_dir=self.img_dir, transform=transform)
        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=2)

        # Setting up Tensorboard 
        self.writer = SummaryWriter()

    def train(self):

        # Training Loop
        for ep in (pbar := tqdm(range(self.epochs))): 
            
            # Set NNs to training mode
            self.encoder.train()
            self.decoder.train()

            # Initialize loss
            running_loss = 0.0

            # Not using labels given its unsupervised learning
            for bi, (imgs, labels) in enumerate(self.train_loader):
                imgs = imgs.to(self.device)

                self.optimizer.zero_grad()  

                # Pass the imgs through the AE and get back the reconstructed imgs
                z = self.encoder(imgs) 
                x_hats = self.decoder(z) 

                # Calculate loss of reconstructed imgs vs. original imgs
                loss = self.criterion(x_hats, imgs)

                # Train the network
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
            
            train_loss = running_loss / len(self.train_loader)
            self.writer.add_scalar("Loss/train", train_loss, ep+1)
            pbar.set_description(f"EPOCH {ep+1} TRAIN LOSS: {train_loss:.4f}")

if __name__ == "__main__":

    img_size = 256
    img_channels = 3
    emb_dim = 1024

    # Initializing the encoder
    # Separating the encoder and decoder so we can more easily use them separately after training
    encoder = ConvEncoder(img_size=img_size, channels=img_channels, emb_dim=emb_dim)

    # Initialize and pass dummy data through encoder to set shape_before_flatten variable
    dummy_data = torch.zeros(1, img_channels, img_size, img_size)
    dummy_output = encoder(dummy_data)
    shape_before_flat = encoder.shape_before_flatten

    # Initialize the decoder
    decoder = ConvDecoder(emb_dim=emb_dim, shape_before_flattening=shape_before_flat, channels=img_channels)

    # Initialize pipeline; Don't need to list all arguments, doing so for example
    pipeline = Pipeline(
        img_size=img_size,
        channels=img_channels,
        epoch=100,
        lr=0.0001,
        batch_size=32,
        emb_dim=emb_dim,
        encoder=encoder,
        decoder=decoder,
        # Will have to create this, expected rows: [img_name, label] (can be a dummy label)
        csv_path="data/img.csv", 
        # Points to the directory with imgs
        img_dir="data/imgs/", 
    )

    # Train AE
    pipeline.train()

