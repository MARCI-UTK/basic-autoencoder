import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
    
class ConvEncoder(nn.Module):
    def __init__(self, img_size, channels, emb_dim):
        super().__init__()

        # Store parameters
        self.img_size = img_size
        self.channels = channels
        self.emb_dim = emb_dim

        # Convolutional Layers
        self.conv1 = nn.Conv2d(self.channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)

        # Stores shape of tensor before flattening to be used in the Decoder
        self.shape_before_flatten = None

        # Flattened shape size; // 8 as we have a stride=2 3 times and * 128 as there are 128 channels at end
        flattened_shape = (self.img_size // 8) * (self.img_size // 8) * 128

        # FC layer for embedding space
        self.fc = nn.Linear(flattened_shape, self.emb_dim)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Storing shape before flattening
        self.shape_before_flatten = x.shape[1:]

        # Flatten x to (batch_size, product_of_dims)
        x = x.view(x.size(0), -1)

        # Typically, the embedding space in an AE is referred to by z
        z = self.fc(x)

        return z
    
class ConvDecoder(nn.Module):
    def __init__(self, emb_dim, shape_before_flattening, channels):
        super().__init__()

        # Store parameters
        self.emb_dim = emb_dim
        self.shape_before_flattening = shape_before_flattening
        self.channels = channels

        # FC for unflattening
        self.fc = nn.Linear(self.emb_dim, np.prod(self.shape_before_flattening))

        # Transpose Conv layers
        self.deconv1 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)

        # Conv layer to get channels to correct size, but retain other dimensions
        self.conv1 = nn.Conv2d(32, self.channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        
        # Expanding the dimensions to where we can unflatten them
        x = self.fc(x)
        x = x.view(x.size(0), *self.shape_before_flattening)

        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))

        # Printed shape is [batch, 32, 256, 256] which confirms idea the conv layer is to get channels right
        # print(f"x shape before final conv layer in Decoder: {x.shape}")

        x = F.sigmoid(self.conv1(x))

        return x
    