import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from PIL import Image

def load_data(batch_size, workers, train_data_path, test_data_path, transform):
    # orig_transform = transforms.Compose([
    #     transforms.Resize([256, 256]),
    #     transforms.ToTensor()
    # ])
    org_transform = transform

    dataset = ImageFolder(root=train_data_path, transform=org_transform)

    test_set = ImageFolder(root=test_data_path, transform=org_transform)
    train_dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=workers,
    shuffle=True,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        num_workers=workers,
        shuffle=False,
    )

    return train_dataloader, test_dataloader