import numpy as np
import torch

# Cutout 정의
class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        return img

# GridMask 정의
class GridMask(object):
    def __init__(self, d1=96, d2=224, rotate=1, ratio=0.6, mode=1, prob=0.6):
        self.d1 = d1
        self.d2 = d2
        self.rotate = rotate
        self.ratio = ratio
        self.mode = mode
        self.prob = prob

    def __call__(self, img):
        if np.random.rand() > self.prob:
            return img

        n_masks = np.random.randint(self.d1, self.d2)
        mask = np.ones((img.size(1), img.size(2)), np.float32)

        for _ in range(n_masks):
            x = np.random.randint(img.size(2))
            y = np.random.randint(img.size(1))
            width = np.random.randint(16, 64)
            height = np.random.randint(16, 64)
            x1 = np.clip(x - width // 2, 0, img.size(2))
            x2 = np.clip(x + width // 2, 0, img.size(2))
            y1 = np.clip(y - height // 2, 0, img.size(1))
            y2 = np.clip(y + height // 2, 0, img.size(1))
            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        return img

# Mosaic 정의
class Mosaic(object):
    def __init__(self, p=0.5, min_offset=0.2, max_offset=0.9):
        self.p = p  # 증강을 적용할 확률
        self.min_offset = min_offset  # 조각을 자를 최소 위치 비율
        self.max_offset = max_offset  # 조각을 자를 최대 위치 비율

    def __call__(self, img):
        if np.random.rand() > self.p:
            return img

        # 이미지의 크기 가져오기
        height, width = img.size(1), img.size(2)

        # Mosaic 적용할 이미지의 사이즈 결정 (4개의 조각으로 나눔)
        half_height = height // 2
        half_width = width // 2

        # 조각의 시작과 끝 위치를 무작위로 결정
        offsets = [
            (0, half_height, 0, half_width),
            (0, half_height, half_width, width),
            (half_height, height, 0, half_width),
            (half_height, height, half_width, width)
        ]

        mosaic_img = img.clone()

        # 각 조각을 자르고 섞어서 새로운 이미지 생성
        for offset in offsets:
            y1, y2, x1, x2 = offset
            mosaic_img[:, y1:y2, x1:x2] = img[:, y1:y2, x1:x2][torch.randperm(half_height), :, :]

        return img

# Mixup 정의
class Mixup(object):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, img1, img2):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        img = lam * img1 + (1 - lam) * img2
        return img, lam

