import torch
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

def load_data(batch_size, workers, train_data_path, test_data_path, orig_transform):
    transform = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor()
    ])

    train_set = ImageFolder(root=train_data_path, transform=transform)

    test_set = ImageFolder(root=test_data_path, transform=transform)

    train_dataloader = DataLoader(
    train_set ,
    batch_size=batch_size,
    num_workers=workers,
    shuffle=True,
    )

    test_dataloader = DataLoader(
        test_set,
        batch_size=batch_size,
        num_workers=workers,
        shuffle=False,
    )

    return train_dataloader, test_dataloader