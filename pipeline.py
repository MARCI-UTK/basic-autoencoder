from clearml import Task 
task = Task.init(project_name="autoencoder", task_name="ae_training") 

import os
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime

from configs import ImgAEConfig, HFImageConfig
from util import count_parameters

class Pipeline():
    def __init__(self, epoch, encoder, decoder, criterion, optimizer, 
                 train_dataloder, valid_dataloader):
        
        # Using time for unique save directory name
        self.save_dir = os.path.join("./runs/", datetime.now().strftime("%Y%b%d_%H:%M:%S"))
        self.make_dirs(self.save_dir)

        # Changing device to most powerful available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.epochs = epoch

        # Setting encoder and decoder
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)

        # Setting up loss and optimizer
        self.criterion = criterion
        self.optimizer = optimizer

        # Loading train/test dataloaders
        self.train_dataloader = train_dataloder
        self.valid_dataloader = valid_dataloader

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
            for bi, (imgs, labels) in enumerate(self.train_dataloader):
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

            valid_loss = self.valid()
            self.writer.add_scalar("Loss/valid", valid_loss, ep+1)

            # Update description bar
            pbar.set_description(f"EPOCH {ep+1} TRAIN LOSS: {train_loss:.4f}")
    
            # Save models every epoch
            torch.save(self.encoder.state_dict(), f"{self.save_dir}/encoder_{ep}")

    def valid(self):
        self.encoder.eval()
        self.decoder.eval()

        # Initialize loss
        running_loss = 0.0

        with torch.no_grad():
            # Not using labels given its unsupervised learning
            for bi, (imgs, labels) in enumerate(self.valid_dataloader):
                imgs = imgs.to(self.device)

                # Pass the imgs through the AE and get back the reconstructed imgs
                z = self.encoder(imgs) 
                x_hats = self.decoder(z) 

                # Calculate loss of reconstructed imgs vs. original imgs
                loss = self.criterion(x_hats, imgs)

                running_loss += loss.item()
            
            # Calculate and record training loss
            valid_loss = running_loss / len(self.valid_dataloader)

        return valid_loss

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
    elif config_type == "hf":
        config = HFImageConfig()

    task.connect(config, name="config")

    # Initialize pipeline    
    pipeline = Pipeline(
        epoch=config.epochs,
        encoder=config.encoder,
        decoder=config.decoder,
        criterion=config.criterion,
        optimizer=config.optimizer,
        train_dataloder=config.train_dataloader,
        valid_dataloader=config.valid_dataloader,
    )

    # Train AE
    pipeline.train()


