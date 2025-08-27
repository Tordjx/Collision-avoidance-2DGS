import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from autoencoder import AutoEncoder
if __name__ == "__main__":


    images = torch.load("images_test.pt")
    depths = torch.load("depths_test.pt")

    dataset = torch.utils.data.TensorDataset(images, depths)  # <-- Corrected

    # Initialize AutoEncoder
    autoencoder = AutoEncoder((3, 128, 128), 32)
    autoencoder.load_state_dict(torch.load("autoencoder.pth"))
    autoencoder.eval()

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create DataLoader
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=2048, shuffle=True, num_workers=20
    )
    autoencoder.to(device)
    from tqdm import tqdm
    errors = 0
    for x, y in dataloader: 
        y_hat = autoencoder(x.to(device))
        error = F.mse_loss(y_hat, torch.log(torch.clamp(y.to(device),1e-2,10)), reduction='none')
        errors += error.sum().item()
    print(errors / len(dataset))

    images = torch.load("images.pt")
    depths = torch.load("depths.pt")

    dataset = torch.utils.data.TensorDataset(images, depths)  # <-- Corrected
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create DataLoader
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=2048, shuffle=True, num_workers=20
    )
    autoencoder.to(device)
    from tqdm import tqdm
    errors = 0
    for x, y in dataloader: 
        y_hat = autoencoder(x.to(device))
        error = F.mse_loss(y_hat, torch.log(torch.clamp(y.to(device),1e-2,10)), reduction='none')
        errors += error.sum().item()
    print(errors / len(dataset))

