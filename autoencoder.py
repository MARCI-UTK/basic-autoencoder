from clearml import Task 
task = Task.init(project_name="autoencoder", task_name="ae_training") 

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime

from configs import ImgAEConfig
from util import count_parameters

class Pipeline():
    def __init__(self, img_size, channels, epoch, lr, batch_size, encoder, decoder, 
                 loss_function, optimizer, train_set, test_set):
        
        # Using time for unique save directory name
        self.save_dir = os.path.join("./runs/", datetime.now().strftime("%Y%b%d_%H:%M:%S"))
        self.make_dirs(self.save_dir)

        # Setting up basics of pipeline
        self.img_size = img_size
        self.channels = channels
        self.epochs = epoch
        self.lr = lr
        self.batch_size = batch_size

        # Changing device to most powerful available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Setting encoder and decoder
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)

        # Setting up loss and optimizer
        self.criterion = loss_function
        self.optimizer = optimizer

        # Loading train/test dataloaders
        self.train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=2)
        self.test_loader = DataLoader(test_set, batch_size=1, shuffle=True, num_workers=2)

        # Setting up Tensorboard 
        self.writer = SummaryWriter(self.save_dir)

        print(f"Param. #:\t {count_parameters(self.encoder) + count_parameters(self.decoder)}")

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
            
            # Calculate and record training loss
            train_loss = running_loss / len(self.train_loader)
            self.writer.add_scalar("Loss/train", train_loss, ep+1)
            # Update description bar
            pbar.set_description(f"EPOCH {ep+1} TRAIN LOSS: {train_loss:.4f}")
    
            # Save models every epoch
            torch.save(self.encoder.state_dict(), f"{self.save_dir}/encoder_{ep}")

    def make_dirs(self, save_dir):
        if not os.path.exists("./runs/"):
            os.mkdir("./runs/")
        
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_type", type=str)

    args = parser.parse_args()

    config_type = args.config_type
    if config_type == "img":
        config = ImgAEConfig()

    task.connect(config, name="config")

    # Initialize pipeline    
    pipeline = Pipeline(
        img_size=config.img_size,
        channels=config.channels,
        epoch=config.epochs,
        lr=config.lr,
        batch_size=config.batch_size,
        encoder=config.encoder,
        decoder=config.decoder,
        loss_function=config.loss_function,
        optimizer=config.optimizer,
        train_set=config.train_set,
        test_set=config.test_set,
    )

    # Train AE
    pipeline.train()


