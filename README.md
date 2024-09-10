# Repository for a Basic Autoencoder

Sets up a simple autoencoder comprised of 2 PyTorch models: an encoder and a decoder.

NOTE: Will have to bring your own data. The pipeline is expecting two things regarding data: 1) path to a csv containing (img_name, label) pairs. The labels can be dummy labels as autoencoder training is unsupervised, and 2) path to the directory holding the actual images.

## File Purposes

models.py: Contains the convolutional encoder and decoder for autoencoder pipeline.
datasets.py: Contains a custom dataset for images for training and testing.
autoencoder.py: Contains a Pipeline object for initializing the autoencoder and training function for training. Also setups Tensorboard for logging.