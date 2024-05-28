import os
import tqdm
import numpy as np
from torchsummary import summary

import torch
import torchvision
from torchvision import transforms
from torch import nn

from nets.nn import resnet50
from utils.loss import yoloLoss
from utils.dataset import Dataset

import argparse
import re


class E_ELAN(nn.Module):
    def __init__(self, in_channels, out_channels, groups=4):
        super(E_ELAN, self).__init__()
        self.groups = groups
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=1, stride=1, padding=0, groups=groups)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1, groups=groups)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1, groups=groups)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.conv4 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1, groups=groups)
        self.bn4 = nn.BatchNorm2d(out_channels)
        self.conv5 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1, groups=groups)
        self.bn5 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.relu(self.bn2(self.conv2(x1)))
        x3 = self.relu(self.bn3(self.conv3(x2)))
        x4 = self.relu(self.bn4(self.conv4(x3)))
        x5 = self.relu(self.bn5(self.conv5(x4)))
        out = torch.cat([x1, x2, x3, x4, x5], dim=1)
        return out


class ResNet50_E_ELAN(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet50_E_ELAN, self).__init__()
        self.resnet = resnet50()

        # Replace certain layers with E-ELAN
        self.resnet.layer2 = self._make_e_elan_layer(256, 512, 3)
        self.resnet.layer3 = self._make_e_elan_layer(512, 1024, 4)
        self.resnet.layer4 = self._make_e_elan_layer(1024, 2048, 6)

        self.fc = nn.Linear(2048, num_classes)

    def _make_e_elan_layer(self, in_channels, out_channels, blocks):
        layers = []
        for _ in range(blocks):
            layers.append(E_ELAN(in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'mps'

    root = args.data_dir
    num_epochs = args.epoch
    batch_size = args.batch_size
    learning_rate = args.lr

    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    net = resnet50()

    if args.pre_weights:
        pattern = 'yolov1_([0-9]+)'
        strs = args.pre_weights.split('.')[-2]
        f_name = strs.split('/')[-1]
        epoch_str = re.search(pattern, f_name).group(1)
        epoch_start = int(epoch_str) + 1
        net.load_state_dict(
            torch.load(f'./weights/{args.pre_weights}')['state_dict'])
    else:
        epoch_start = 1
        resnet = torchvision.models.resnet50(pretrained=True)
        new_state_dict = resnet.state_dict()

        net_dict = net.state_dict()
        for k in new_state_dict.keys():
            if k in net_dict.keys() and not k.startswith('fc'):
                net_dict[k] = new_state_dict[k]
        net.load_state_dict(net_dict)

    print('NUMBER OF CUDA DEVICES:', torch.cuda.device_count())

    criterion = yoloLoss().to(device)
    net = net.to(device)

    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)

    # summary(net, input_size=(3, 448, 448))
    # 다양한 학습률

    net.train()

    params = []
    params_dict = dict(net.named_parameters())
    for key, value in params_dict.items():
        if key.startswith('features'):
            params += [{'params': [value], 'lr': learning_rate * 10}]
        else:
            params += [{'params': [value], 'lr': learning_rate}]

    optimizer = torch.optim.SGD(
        params, lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    with open('./Dataset/train.txt') as f:
        train_names = f.readlines()
    train_dataset = Dataset(root, train_names, train=True,
                            transform=[transforms.ToTensor(),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ColorJitter(
                                           brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                                       transforms.RandomResizedCrop(args.img_size, scale=(0.8, 1.0))])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=os.cpu_count())

    with open('./Dataset/test.txt') as f:
        test_names = f.readlines()
    test_dataset = Dataset(root, test_names, train=False,
                           transform=[transforms.ToTensor()])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size // 2, shuffle=False,
                                              num_workers=os.cpu_count())

    print(f'NUMBER OF DATA SAMPLES: {len(train_dataset)}')
    print(f'BATCH SIZE: {batch_size}')

    for epoch in range(epoch_start, num_epochs):
        net.train()

        if epoch == 30:
            learning_rate = 0.0001
        if epoch == 40:
            learning_rate = 0.00001
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

        # training
        total_loss = 0.
        print(('\n' + '%10s' * 3) % ('epoch', 'loss', 'gpu'))
        progress_bar = tqdm.tqdm(
            enumerate(train_loader), total=len(train_loader))
        for i, (images, target) in progress_bar:
            images = images.to(device)
            target = target.to(device)

            pred = net(images)

            optimizer.zero_grad()
            loss = criterion(pred, target.float())

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            mem = '%.3gG' % (torch.cuda.memory_reserved() /
                             1E9 if torch.cuda.is_available() else torch.mps.current_allocated_memory() / 1E9)
            s = ('%10s' + '%10.4g' + '%10s') % ('%g/%g' %
                                                (epoch, num_epochs), total_loss / (i + 1), mem)
            progress_bar.set_description(s)

        # validation
        validation_loss = 0.0
        net.eval()
        with torch.no_grad():
            progress_bar = tqdm.tqdm(
                enumerate(test_loader), total=len(test_loader))
            for i, (images, target) in progress_bar:
                images = images.to(device)
                target = target.to(device)

                prediction = net(images)
                loss = criterion(prediction, target)
                validation_loss += loss.data

        validation_loss /= len(test_loader)
        print(f'Validation_Loss:{validation_loss:07.3}')

        if epoch % 5 == 0:
            save = {'state_dict': net.state_dict()}
            torch.save(save, f'./weights/yolov1_{epoch+1:04d}.pth')

    save = {'state_dict': net.state_dict()}
    torch.save(save, './weights/yolov1_final.pth')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--data_dir", type=str, default='./Dataset')
    parser.add_argument("--pre_weights", type=str, help="pretrained weight")
    parser.add_argument("--save_dir", type=str, default="./weights")
    parser.add_argument("--img_size", type=int, default=448)
    args = parser.parse_args()

    # args.pre_weights = 'yolov1_0010.pth'
    main(args)
