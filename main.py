import os
import tqdm
import numpy as np
from torchsummary import summary

import torch
import torchvision
from torchvision import transforms

from nets.nn import resnet50, resnet50FPN
from utils.loss import yoloLoss
from utils.dataset import Dataset
import argparse
import re


def main(args):
    # CUDA를 사용하여 장치 설정
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')

    root = args.data_dir
    num_epochs = args.epoch
    batch_size = args.batch_size
    learning_rate = args.lr

    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    # net = resnet50()
    net = resnet50FPN()

    if (args.pre_weights != None):
        pattern = 'yolov1_([0-9]+)'
        strs = args.pre_weights.split('.')[-2]
        f_name = strs.split('/')[-1]
        epoch_str = re.search(pattern, f_name).group(1)
        epoch_start = int(epoch_str) + 1
        state_dict = torch.load(
            f'./weights/{args.pre_weights}', map_location=device)['state_dict']
        net.load_state_dict(state_dict, strict=False)
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

    # summary(net,input_size=(3,448,448))
    # different learning rate

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

    # optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    with open('./Dataset/train.txt') as f:
        train_names = f.readlines()
    train_dataset = Dataset(root, train_names, train=True,
                            transform=[])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=0)

    with open('./Dataset/test.txt') as f:
        test_names = f.readlines()
    test_dataset = Dataset(root, test_names, train=False,
                           transform=[])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size // 2, shuffle=False,
                                              num_workers=0)

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
            images = images.to(device)  # (8, 448, 448, 3)
            images = images.permute(0, 3, 1, 2)
            target = target.to(device)  # (8, 14, 14, 30)
            # target 텐서에서 boxes와 labels를 추출하는 코드
            grid_size = 14
            target_shape = target.shape  # (batch_size, 14, 14, 30)
            batch_size = target_shape[0]
            cell_size = 1. / grid_size

            target_dict = []
            for b in range(batch_size):
                dd = {'boxes': [], 'labels': []}
                for i in range(grid_size):
                    for j in range(grid_size):
                        cell = target[b, i, j]
                        # confidence가 있는 cell만 처리
                        if cell[4] > 0.5 or cell[9] > 0.5:
                            # 두 개의 bounding box 처리
                            for b_idx in [0, 5]:
                                box_confidence = cell[4 + b_idx]
                                if box_confidence > 0.5:
                                    # 클래스 확률이 가장 높은 클래스 찾기
                                    class_probabilities = cell[10:]
                                    class_label = torch.argmax(
                                        class_probabilities)
                                    dd['labels'].append(class_label.item())

                                    # bounding box 정보 추출
                                    cx = (j + cell[b_idx]) * cell_size
                                    cy = (i + cell[b_idx + 1]) * cell_size
                                    width = cell[b_idx + 2] * cell_size
                                    height = cell[b_idx + 3] * cell_size
                                    x1 = (cx - width / 2) * target_shape[2]
                                    y1 = (cy - height / 2) * target_shape[3]
                                    x2 = (cx + width / 2) * target_shape[2]
                                    y2 = (cy + height / 2) * target_shape[3]
                                    dd['boxes'].append([x1, y1, x2, y2])

                dd['boxes'] = torch.tensor(dd['boxes']).to(device)
                dd['labels'] = torch.tensor(dd['labels']).to(device)
                target_dict.append(dd)

            for i in range(0, len(target_dict)):
                print(target_dict[i]['boxes'])
                print(target_dict[i]['labels'])
            target_dict = np.array(target_dict)
            pred = net(images, target_dict)
            print(pred)
            optimizer.zero_grad()
            # loss = criterion(pred, target.float())

            # loss.backward()
            optimizer.step()

            # total_loss += loss.item()
            # total_loss += pred.item('loss_objectness')
            total_loss = 0
            mem = '%.3gG' % (torch.cuda.memory_reserved(0) /
                             1E9 if torch.cuda.is_available() else 0)
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
                images = images.to(device)  # (8, 448, 448, 3)
                images = images.permute(0, 3, 1, 2)
                target = target.to(device)  # (8, 14, 14, 30)
                # target 텐서에서 boxes와 labels를 추출하는 코드
                grid_size = 14
                target_shape = target.shape  # (batch_size, 14, 14, 30)
                batch_size = target_shape[0]
                cell_size = 1. / grid_size

                target_dict = []
                for b in range(batch_size):
                    dd = {'boxes': [], 'labels': []}
                    for i in range(grid_size):
                        for j in range(grid_size):
                            cell = target[b, i, j]
                            # confidence가 있는 cell만 처리
                            if cell[4] > 0.5 or cell[9] > 0.5:
                                # 두 개의 bounding box 처리
                                for b_idx in [0, 5]:
                                    box_confidence = cell[4 + b_idx]
                                    if box_confidence > 0.5:
                                        # 클래스 확률이 가장 높은 클래스 찾기
                                        class_probabilities = cell[10:]
                                        class_label = torch.argmax(
                                            class_probabilities)
                                        dd['labels'].append(class_label.item())

                                        # bounding box 정보 추출
                                        cx = (j + cell[b_idx]) * cell_size
                                        cy = (i + cell[b_idx + 1]) * cell_size
                                        width = cell[b_idx + 2] * cell_size
                                        height = cell[b_idx + 3] * cell_size
                                        x1 = (cx - width / 2) * target_shape[2]
                                        y1 = (cy - height / 2) * \
                                            target_shape[3]
                                        x2 = (cx + width / 2) * target_shape[2]
                                        y2 = (cy + height / 2) * \
                                            target_shape[3]
                                        dd['boxes'].append([x1, y1, x2, y2])

                    dd['boxes'] = torch.tensor(dd['boxes']).to(device)
                    dd['labels'] = torch.tensor(dd['labels']).to(device)
                    target_dict.append(dd)

                for i in range(0, len(target_dict)):
                    print(target_dict[i]['boxes'])
                    print(target_dict[i]['labels'])
                target_dict = np.array(target_dict)
                prediction = net(images, target_dict)
                # loss = criterion(prediction, target)
                # validation_loss += loss.data

        # validation_loss /= len(test_loader)
        # validation_loss = prediction.item('loss_objectness')
        validation_loss = 0
        print(f'Validation_Loss:{validation_loss:07.3}')

        # if epoch % 5:
        #    save = {'state_dict': net.state_dict()}
        #    torch.save(save, f'./weights/yolov1_{epoch+1:04d}.pth')
        save = {'state_dict': net.state_dict()}
        torch.save(save, f'./weights/yolov1_{epoch:04d}.pth')

    save = {'state_dict': net.state_dict()}
    torch.save(save, './weights/yolov1_final.pth')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--data_dir", type=str, default='./Dataset')
    parser.add_argument("--pre_weights", type=str, help="pretrained weight")
    parser.add_argument("--save_dir", type=str, default="./weights")
    parser.add_argument("--img_size", type=int, default=448)
    args = parser.parse_args()

    args.pre_weights = 'yolov1_0010.pth'
    main(args)
