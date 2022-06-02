import torch
from torch import nn
import torch.utils.data as data
from data.dataset import ColorHintDataset
from model import *
from solver import *


def main():
    # Set mode
    mode = "train" #train, test

    # Set file root
    root_path = "/home/ksh/Desktop/CV/cv_project/"
    test_path = root_path + "test/" #test folder has hint folder / mask folder
    save_pth_path = "checkpoints/"
    save_val_img_path = "output_val/"
    save_test_img_path = "output_test/"
    weight_path = "HSCU-Net_ep-338_losses-0.01282.pth"

    # Depend on runtime setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set hyper-param
    batch_size = 32
    n_workers = 4
    epochs = 1000
    lr = 5e-8 #0 - 0.00005 -> 186 - 0.000005 -> 286 - 5e-6 -> 324 - 5e-8

    # Load dataset and dataloader
    train_dataset = ColorHintDataset(root_path, 256, "train")
    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers)

    valid_dataset = ColorHintDataset(root_path, 256, "val")
    valid_dataloader = data.DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=n_workers)

    real_test_dataset = ColorHintDataset(test_path, 256, "test")
    real_test_dataloader = data.DataLoader(real_test_dataset, batch_size= 1, shuffle=False, num_workers=n_workers)



    # Set Network
    netG = HINT_CSATUNet(n_channels=4, n_classes=2).to(device)

    # Set pre-trained weight
    netG.load_state_dict(torch.load(weight_path, map_location=device))

    # L1 Loss function
    criterion = nn.L1Loss().to(device)

    # Optimizer
    optimG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.5,0.999))

    # train & test
    if mode == "train":
        train(netG, train_dataloader, valid_dataloader, optimG, criterion, epochs, device, save_pth_path, save_val_img_path)

    elif mode == "test":
        test(netG, real_test_dataloader, device, save_test_img_path)

if __name__ == "__main__":
    main()