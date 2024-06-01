import os
import os.path

import random
import numpy as np

import torch
import torch.utils.data as data

import cv2


class Dataset(data.Dataset):
    image_size = 448

    def __init__(self, root, file_names, train, transform):
        print('DATA INITIALIZATION')
        self.root_images = os.path.join(root, 'Images')
        self.root_labels = os.path.join(root, 'Labels')
        self.train = train
        self.transform = transform
        self.f_names = []
        self.boxes = []
        self.labels = []
        self.mean = (123, 117, 104)  # RGB

        for line in file_names:
            line = line.rstrip()
            with open(f"{self.root_labels}/{line}.txt") as f:
                objects = f.readlines()
                self.f_names.append(line + '.jpg')
                box = []
                label = []
                for object in objects:
                    c, x1, y1, x2, y2 = map(float, object.rstrip().split())
                    box.append([x1, y1, x2, y2])
                    label.append(int(c) + 1)
                self.boxes.append(torch.Tensor(box))
                self.labels.append(torch.LongTensor(label))
        self.num_samples = len(self.boxes)

    def __getitem__(self, idx):
        f_name = self.f_names[idx]
        img = cv2.imread(os.path.join(self.root_images, f_name))
        boxes = self.boxes[idx].clone()
        labels = self.labels[idx].clone()

        if self.train:
            # img = self.random_bright(img)
            img, boxes = self.random_flip(img, boxes)
            img, boxes = self.randomScale(img, boxes)
            img = self.randomBlur(img)
            img = self.RandomBrightness(img)
            # img = self.RandomHue(img)
            img = self.RandomSaturation(img)
            img, boxes, labels = self.randomShift(img, boxes, labels)
            img, boxes, labels = self.randomCrop(img, boxes, labels)
            
            """ 추가된 데이터 증강 기법들 """
            img, boxes = self.randomRotate(img, boxes)
            img = self.addGaussianNoise(img)

            """ 바운딩 박스를 이미지 크기에 맞게 조정 """
            h, w, _ = img.shape
            boxes = self.clamp_boxes(boxes, w, h)

            """ 모자이크 기법 추가 """
            img, boxes, labels = mosaic(img, boxes, labels)
            
        # # debug
        # box_show = boxes.numpy().reshape(-1)
        # print(box_show)
        # img_show = self.BGR2RGB(img)
        # pt1=(int(box_show[0]),int(box_show[1])); pt2=(int(box_show[2]),int(box_show[3]))
        # cv2.rectangle(img_show,pt1=pt1,pt2=pt2,color=(0,255,0),thickness=1)
        # plt.figure()
        #
        # # cv2.rectangle(img,pt1=(10,10),pt2=(100,100),color=(0,255,0),thickness=1)
        # plt.imshow(img_show)
        # plt.show()
        # # debug

        h, w, _ = img.shape
        boxes /= torch.Tensor([w, h, w, h]).expand_as(boxes)
        img = self.BGR2RGB(img)
        img = self.subMean(img, self.mean)
        img = cv2.resize(img, (self.image_size, self.image_size))

        target = self.encoder(boxes, labels)  # 14x14x30
        for t in self.transform:
            img = t(img)

        return img, target

    def __len__(self):
        return self.num_samples

    def encoder(self, boxes, labels):
        grid_num = 14
        target = torch.zeros((grid_num, grid_num, 30))
        cell_size = 1. / grid_num
        wh = boxes[:, 2:] - boxes[:, :2]
        cxcy = (boxes[:, 2:] + boxes[:, :2]) / 2
        for i in range(cxcy.size()[0]):
            cxcy_sample = cxcy[i]
            #grid cell의 Y축과 X축의 index 계산
            ij = (cxcy_sample / cell_size).ceil() - 1
            #grid cell의 2개 bbox의  confidence score을 1로 set
            target[int(ij[1]), int(ij[0]), 4] = 1
            target[int(ij[1]), int(ij[0]), 9] = 1
            #grid cell의 class probability을 1로 set
            target[int(ij[1]), int(ij[0]), int(labels[i]) + 9] = 1
            #bbox의 중심점 (cx,cy)를 (i,j) grid cell의 원점으로 부터
            # offset값으로 (delta_x, delta_y) 계산하고 target 행렬 tensor의 
            # (i,j) grid cell 위치에 정규화한 bbox정보를 저장
            xy = ij * cell_size
            delta_xy = (cxcy_sample - xy) / cell_size
            target[int(ij[1]), int(ij[0]), 2:4] = wh[i]
            target[int(ij[1]), int(ij[0]), :2] = delta_xy
            target[int(ij[1]), int(ij[0]), 7:9] = wh[i]
            target[int(ij[1]), int(ij[0]), 5:7] = delta_xy
        return target

    def BGR2RGB(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def BGR2HSV(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    def HSV2BGR(self, img):
        return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    
    """ 바운딩 박스 좌표의 유효성 확인을 위한 클램프 박스 추가 """
    @staticmethod
    def clamp_boxes(boxes, width, height):
        boxes[:, 0].clamp_(0, width)
        boxes[:, 1].clamp_(0, height)
        boxes[:, 2].clamp_(0, width)
        boxes[:, 3].clamp_(0, height)
        return boxes
    
    """ 랜덤 회전, 가우시안 노이즈 추가 """
    def randomRotate(self, bgr, boxes, angle_range=(-10, 10)):
        if random.random() < 0.5:
            angle = random.uniform(*angle_range)
            height, width = bgr.shape[:2]
            center = (width // 2, height // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            cos = np.abs(matrix[0, 0])
            sin = np.abs(matrix[0, 1])
            new_width = int((height * sin) + (width * cos))
            new_height = int((height * cos) + (width * sin))
            matrix[0, 2] += (new_width / 2) - center[0]
            matrix[1, 2] += (new_height / 2) - center[1]
            rotated_img = cv2.warpAffine(bgr, matrix, (new_width, new_height))
            
            # 바운딩 박스 좌표 회전
            boxes_np = boxes.numpy()
            for i in range(len(boxes_np)):
                xmin, ymin, xmax, ymax = boxes_np[i]
                points = np.array([[xmin, ymin], [xmax, ymin], [xmin, ymax], [xmax, ymax]])
                ones = np.ones(shape=(len(points), 1))
                points_ones = np.hstack([points, ones])
                rotated_points = matrix @ points_ones.T
                xmin, ymin = rotated_points[:2].min(axis=1)
                xmax, ymax = rotated_points[:2].max(axis=1)
                boxes_np[i] = [xmin, ymin, xmax, ymax]
            boxes = torch.Tensor(boxes_np)
            return rotated_img, boxes
        return bgr, boxes

    def addGaussianNoise(self, bgr, mean=0, std=0.01):
        if random.random() < 0.5:
            noise = np.random.normal(mean, std, bgr.shape).astype(bgr.dtype)
            bgr = cv2.add(bgr, noise)
        return bgr
    
    """ 모자이크 기법 """
    def mosaic(img, boxes, labels, ratio=0.2):
        """
        이미지에 모자이크를 적용하여 데이터 증강을 수행하는 함수

        Args:
            img (numpy.ndarray): 원본 이미지
            boxes (numpy.ndarray): 바운딩 박스들의 좌표 [x_min, y_min, x_max, y_max]
            labels (numpy.ndarray): 바운딩 박스들에 대한 라벨
            ratio (float): 모자이크 비율, 기본값은 0.2

        Returns:
            numpy.ndarray: 모자이크가 적용된 이미지
            numpy.ndarray: 모자이크가 적용된 바운딩 박스들의 좌표
            numpy.ndarray: 모자이크가 적용된 바운딩 박스들의 라벨
        """
        h, w, _ = img.shape
        new_img = img.copy()

        # 모자이크를 적용할 영역의 크기 계산
        mosaiced_h = int(h * ratio)
        mosaiced_w = int(w * ratio)

        # 랜덤하게 모자이크 적용할 영역 선택
        y_offset = np.random.randint(0, h - mosaiced_h + 1)
        x_offset = np.random.randint(0, w - mosaiced_w + 1)

        # 모자이크 적용
        mosaic_area = img[y_offset:y_offset + mosaiced_h, x_offset:x_offset + mosaiced_w]
        mosaic_area = cv2.resize(mosaic_area, (w, h), interpolation=cv2.INTER_NEAREST)
        new_img[y_offset:y_offset + mosaiced_h, x_offset:x_offset + mosaiced_w] = mosaic_area

        # 모자이크 영역에 속하는 바운딩 박스들의 좌표 및 라벨을 업데이트
        for i in range(len(boxes)):
            box = boxes[i]
            if box[0] >= x_offset and box[2] <= x_offset + mosaiced_w and box[1] >= y_offset and box[3] <= y_offset + mosaiced_h:
                # 모자이크 영역에 속하는 바운딩 박스의 좌표를 모자이크 영역 내부로 이동
                box[0] = max(box[0] - x_offset, 0)
                box[1] = max(box[1] - y_offset, 0)
                box[2] = min(box[2] - x_offset, mosaiced_w)
                box[3] = min(box[3] - y_offset, mosaiced_h)
                # 모자이크 영역에 속하는 바운딩 박스의 라벨을 모자이크된 이미지에 맞게 업데이트
                boxes[i] = box

        return new_img, boxes, labels


    def RandomBrightness(self, bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            v = v * adjust
            v = np.clip(v, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def RandomSaturation(self, bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            s = s * adjust
            s = np.clip(s, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def RandomHue(self, bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            h = h * adjust
            h = np.clip(h, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def randomBlur(self, bgr):
        if random.random() < 0.5:
            bgr = cv2.blur(bgr, (5, 5))
        return bgr

    def randomShift(self, bgr, boxes, labels):
        center = (boxes[:, 2:] + boxes[:, :2]) / 2
        if random.random() < 0.5:
            height, width, c = bgr.shape
            after_shfit_image = np.zeros((height, width, c), dtype=bgr.dtype)
            after_shfit_image[:, :, :] = (104, 117, 123)  # bgr
            shift_x = random.uniform(-width * 0.2, width * 0.2)
            shift_y = random.uniform(-height * 0.2, height * 0.2)

            if shift_x >= 0 and shift_y >= 0:
                after_shfit_image[int(shift_y):, int(shift_x):, :] = bgr[:height - int(shift_y), :width - int(shift_x),
                                                                     :]
            elif shift_x >= 0 and shift_y < 0:
                after_shfit_image[:height + int(shift_y), int(shift_x):, :] = bgr[-int(shift_y):, :width - int(shift_x),
                                                                              :]
            elif shift_x < 0 and shift_y >= 0:
                after_shfit_image[int(shift_y):, :width + int(shift_x), :] = bgr[:height - int(shift_y), -int(shift_x):,
                                                                             :]
            elif shift_x < 0 and shift_y < 0:
                after_shfit_image[:height + int(shift_y), :width + int(shift_x), :] = bgr[-int(shift_y):,
                                                                                      -int(shift_x):, :]

            shift_xy = torch.FloatTensor([[int(shift_x), int(shift_y)]]).expand_as(center)
            center = center + shift_xy
            mask1 = (center[:, 0] > 0) & (center[:, 0] < width)
            mask2 = (center[:, 1] > 0) & (center[:, 1] < height)
            mask = (mask1 & mask2).view(-1, 1)
            boxes_in = boxes[mask.expand_as(boxes)].view(-1, 4)
            if len(boxes_in) == 0:
                return bgr, boxes, labels
            box_shift = torch.FloatTensor([[int(shift_x), int(shift_y), int(shift_x), int(shift_y)]]).expand_as(
                boxes_in)
            boxes_in = boxes_in + box_shift
            labels_in = labels[mask.view(-1)]
            return after_shfit_image, boxes_in, labels_in
        return bgr, boxes, labels

    def randomScale(self, bgr, boxes):
        if random.random() < 0.5:
            scale = random.uniform(0.8, 1.2)
            height, width, c = bgr.shape
            bgr = cv2.resize(bgr, (int(width * scale), height))
            scale_tensor = torch.FloatTensor([[scale, 1, scale, 1]]).expand_as(boxes)
            boxes = boxes * scale_tensor
            return bgr, boxes
        return bgr, boxes

    def randomCrop(self, bgr, boxes, labels):
        if random.random() < 0.5:
            center = (boxes[:, 2:] + boxes[:, :2]) / 2
            height, width, c = bgr.shape
            h = random.uniform(0.6 * height, height)
            w = random.uniform(0.6 * width, width)
            x = random.uniform(0, width - w)
            y = random.uniform(0, height - h)
            x, y, h, w = int(x), int(y), int(h), int(w)

            center = center - torch.FloatTensor([[x, y]]).expand_as(center)
            mask1 = (center[:, 0] > 0) & (center[:, 0] < w)
            mask2 = (center[:, 1] > 0) & (center[:, 1] < h)
            mask = (mask1 & mask2).view(-1, 1)

            boxes_in = boxes[mask.expand_as(boxes)].view(-1, 4)
            if len(boxes_in) == 0:
                return bgr, boxes, labels
            box_shift = torch.FloatTensor([[x, y, x, y]]).expand_as(boxes_in)

            boxes_in = boxes_in - box_shift
            boxes_in[:, 0] = boxes_in[:, 0].clamp_(min=0, max=w)
            boxes_in[:, 2] = boxes_in[:, 2].clamp_(min=0, max=w)
            boxes_in[:, 1] = boxes_in[:, 1].clamp_(min=0, max=h)
            boxes_in[:, 3] = boxes_in[:, 3].clamp_(min=0, max=h)

            labels_in = labels[mask.view(-1)]
            img_croped = bgr[y:y + h, x:x + w, :]
            return img_croped, boxes_in, labels_in
        return bgr, boxes, labels

    def subMean(self, bgr, mean):
        mean = np.array(mean, dtype=np.float32)
        bgr = bgr - mean
        return bgr

    def random_flip(self, im, boxes):
        if random.random() < 0.5:
            im_lr = np.fliplr(im).copy()
            h, w, _ = im.shape
            xmin = w - boxes[:, 2]
            xmax = w - boxes[:, 0]
            boxes[:, 0] = xmin
            boxes[:, 2] = xmax
            return im_lr, boxes
        return im, boxes

    def random_bright(self, im, delta=16):
        alpha = random.random()
        if alpha > 0.3:
            im = im * alpha + random.randrange(-delta, delta)
            im = im.clip(min=0, max=255).astype(np.uint8)
        return im


def main():
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms
    file_root = './Dataset'
    with open('./Dataset/train.txt') as f:
        train_names = f.readlines()
    train_dataset = Dataset(root=file_root, file_names=train_names, train=True,
                            transform=[transforms.ToTensor()])
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=os.cpu_count() - 2)
    train_iter = iter(train_loader)
    for i in range(10):
        img, target = next(train_iter)
        print(img, target)


if __name__ == '__main__':
    main()