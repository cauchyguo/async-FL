from torchvision import datasets, transforms

from dataset.BaseDataset import BaseDataset
import os

class TinyImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        # ImageFolder normally returns
        original_tuple = super(TinyImageFolder, self).get__item__(index)
        # 图像路径
        path = self.imgs[index][0]
        # 构造一个新的tuple使其包括origin和图像路径
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


class TinyImage(BaseDataset):
    def __init__(self, clients, iid_config, params):
        BaseDataset.__init__(self, iid_config)
        transformer = transforms.Compose([
            # 将图片转化为Tensor格式
            transforms.ToTensor(),
        ])

        normalize = transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
        transform_train = transforms.Compose(
            [transforms.RandomResizedCrop(32), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
            normalize, ])
        transform_test = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), normalize, ])

        # # 获取数据集
        # self.train_dataset = datasets.FashionMNIST(root=self.path, train=True,
        #                                            transform=transformer, download=True)
        # self.test_dataset = datasets.FashionMNIST(root=self.path, train=False,
        #                                           transform=transformer, download=True)
        # self.init(clients, self.train_dataset, self.test_dataset)

        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/tiny-imagenet-200/')
        self.train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform_train)
        self.test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'val_new'), transform=transform_test)
        self.init(clients, self.train_dataset, self.test_dataset)


# train_dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=data_transforms['train'])
# val_dataset = torchvision.datasets.ImageFolder(root='tiny-imagenet-200/val', transform=data_transforms['val'])
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
# val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2)

# import torch
# import torchvision

# data_transforms = {
#     'train': torchvision.transforms.Compose([
#         torchvision.transforms.RandomResizedCrop(64),
#         torchvision.transforms.RandomHorizontalFlip(),
#         torchvision.transforms.ToTensor(),
#         torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
#     'val': torchvision.transforms.Compose([
#         torchvision.transforms.Resize(64),
#         torchvision.transforms.CenterCrop(64),
#         torchvision.transforms.ToTensor(),
#         torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
# }

# train_dataset = torchvision.datasets.ImageFolder(root='tiny-imagenet-200/train', transform=data_transforms['train'])
# val_dataset = torchvision.datasets.ImageFolder(root='tiny-imagenet-200/val', transform=data_transforms['val'])
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
# val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2)