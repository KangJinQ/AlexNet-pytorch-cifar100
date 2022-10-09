import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data


def load_datasets(name, root, batch_size):
    if name == "cifar100":
        train_dataset = datasets.CIFAR100(root=root,
                                          download=False,
                                          train=True,
                                          transform=transforms.Compose([
                                              transforms.Resize(32),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                          ]))
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                       shuffle=True, num_workers=2)
        test_dataset = datasets.CIFAR100(root=root,
                                         download=True,
                                         train=False,
                                         transform=transforms.Compose([
                                             transforms.Resize(32),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                         ]))

        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                      shuffle=False, num_workers=2)
    return train_dataloader, test_dataloader
