import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from autoencoder import AutoEncoder
from tqdm import tqdm


def PSNR(y,yhat) : 
    dmax = 10 
    y= torch.exp(y)
    yhat = torch.exp(yhat)
    mse = F.mse_loss(y,yhat, reduction = "none") #B
    eps = 0
    mse = torch.where(mse<eps, eps, mse).mean((1,2,3))
    psnr = 10* torch.log10((dmax**2)/mse) #B 
    return psnr


def evaluate(autoencoder, images, depths, device, name="dataset"):
    dataset = torch.utils.data.TensorDataset(images, depths)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=2048, shuffle=False, num_workers=20
    )

    all_errors = []
    all_psnr = []
    for x, y in tqdm(dataloader, desc=f"Evaluating {name}"):
        with torch.no_grad():
            y_hat = autoencoder(x.to(device))
            y = torch.log(torch.clamp(y.to(device), 1e-2, 10))
            # reduction='none' → keep per-sample loss
            error = F.mse_loss(
                y_hat, y, reduction="none"
            )
            # flatten per-pixel errors, then mean per sample
            sample_errors = error.view(error.size(0), -1).mean(dim=1).cpu().numpy()
            all_errors.append(sample_errors)
            all_psnr.append(PSNR(y_hat, y ).cpu().numpy())
    all_errors = np.concatenate(all_errors)
    mean = all_errors.mean()
    std = all_errors.std()
    all_psnr = np.concatenate(all_psnr)
    print(all_psnr)
    mean_psnr = all_psnr.mean()
    std_psnr = all_psnr.std()
    return mean, std, mean_psnr, std_psnr 


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    autoencoder = AutoEncoder((3, 128, 128), 32)
    autoencoder.load_state_dict(torch.load("autoencoder.pth"))
    autoencoder.eval().to(device)

    # Test set
    images_test = torch.load("images_test.pt")
    depths_test = torch.load("depths_test.pt")
    mean_test, std_test, mean_test_psnr, std_test_psnr = evaluate(autoencoder, images_test, depths_test, device, name="test")

    # Train set
    images_train = torch.load("images.pt")
    depths_train = torch.load("depths.pt")
    mean_train, std_train , mean_train_psnr, std_train_psnr= evaluate(autoencoder, images_train, depths_train, device, name="train")

    print(f"test set: mean={mean_test:.2f}, std={std_test:.2f}, mean psnr={mean_test_psnr:.2f}, std psnr = {std_test_psnr:.2f}")
    print(f"Train set: mean={mean_train:.2f}, std={std_train:.2f}, mean psnr={mean_train_psnr:.2f}, std psnr = {std_train_psnr:.2f}")
