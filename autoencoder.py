import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoEncoder(nn.Module):
    def __init__(self, input_shape, z_size):
        super(AutoEncoder, self).__init__()
        self.z_size = z_size

        self.encoder = nn.Sequential(
            nn.Conv2d(input_shape[0], 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        # Compute shape before flattening
        with torch.no_grad():
            self.shape_before_flatten = self.encoder(
                torch.ones((1,) + input_shape)
            ).shape[1:]
        flatten_size = int(np.prod(self.shape_before_flatten))

        self.encode_linear = nn.Linear(flatten_size, self.z_size)
        self.decode_linear = nn.Linear(self.z_size, flatten_size)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
            # nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon

    def encode(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        x = self.encoder(x)
        x = x.flatten(1)
        z = self.encode_linear(x)
        z = F.sigmoid(z)
        # print(z)
        return z

    def decode(self, z):

        x = self.decode_linear(z)
        x = x.view(x.size(0), *self.shape_before_flatten)
        x_recon = self.decoder(x)
        return x_recon


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process dataset parameters.")
    parser.add_argument(
        "--epochs", type=int, help="Number of epochs of training", default=100
    )
    parser.add_argument("--batch_size", type=int, help="Batch size", default=256)
    parser.add_argument("--skip_train", type=bool, help="Skip training", default=False)
    args = parser.parse_args()
    # Create dataset correctly

    images = torch.load("images.pt")
    depths = torch.load("depths.pt")

    dataset = torch.utils.data.TensorDataset(images, depths)  # <-- Corrected

    # Initialize AutoEncoder
    autoencoder = AutoEncoder((3, 128, 128), 32)

    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create DataLoader
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=20
    )
    autoencoder.to(device)
    from tqdm import tqdm

    def noiser(x):
        noise = torch.randn_like(x) * 0.01
        x = torch.clamp(x + noise, 0, 1)
        return x

    if not args.skip_train:
        for epoch in tqdm(range(args.epochs)):
            losses = 0
            for i, data in enumerate(dataloader):
                x, y = data
                print(x.shape, y.shape)
                # print(y)
                x = x.to(device)
                y = torch.log(y).to(device).unsqueeze(1)
                optimizer.zero_grad()
                outputs = autoencoder(noiser(x))
                loss = criterion(outputs, y)
                loss.backward()
                losses += loss.item()
                optimizer.step()
            losses = losses / len(dataloader)
        torch.save(autoencoder.state_dict(), "autoencoder.pth")
    autoencoder.load_state_dict(torch.load("autoencoder.pth"))
    autoencoder.eval()

    # Get a batch of images
    (
        x,
        y,
    ) = next(iter(dataloader))
    x = x.to(device)
    y = torch.log(y)
    y = y.to(device)
    # Get reconstructions
    with torch.no_grad():
        reconstructions = autoencoder(x)

    # Plot original and reconstructed images
    num_images = 5  # Reduced number of images for a cleaner illustration
    fig, axs = plt.subplots(3, num_images, figsize=(12, 6))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)  # Less space between images

    for i in range(num_images):
        axs[0, i].imshow(x[i].cpu().permute(1, 2, 0))
        axs[0, i].set_xticks([])
        axs[0, i].set_yticks([])

        axs[1, i].imshow(reconstructions[i].cpu().detach().permute(1, 2, 0))
        axs[1, i].set_xticks([])
        axs[1, i].set_yticks([])

        axs[2, i].imshow(y[i].cpu().detach().permute(1, 2, 0))
        axs[2, i].set_xticks([])
        axs[2, i].set_yticks([])

    axs[0, 0].set_ylabel("Original", fontsize=14)
    axs[1, 0].set_ylabel("Reconstruction", fontsize=14)
    axs[2, 0].set_ylabel("Ground truth", fontsize=14)
    plt.tight_layout()
    plt.savefig("Vision training.png", dpi=300, bbox_inches="tight")
