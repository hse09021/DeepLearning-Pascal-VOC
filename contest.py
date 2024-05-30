import os
import shutil
from utils.util import *
from collections import defaultdict
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import transforms

from nets.nn import resnet50
from utils.dataset import Dataset

import argparse
import re
import time


VOC_CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']

CLASSES_DIC = {'aeroplane': 0,'bicycle':1,'bird':2,'boat':3,
                    'bottle':4, 'bus':5, 'car':6, 'cat':7,
                    'chair':8,'cow':9,'diningtable':10, 'dog':11,
                    'horse':12,'motorbike':13, 'person':14, 'pottedplant':15,
                    'sheep':16, 'sofa':17, 'train':18, 'tvmonitor':19}

COLORS = {'aeroplane': (0, 0, 0),
          'bicycle': (128, 0, 0),
          'bird': (0, 128, 0),
          'boat': (128, 128, 0),
          'bottle': (0, 0, 128),
          'bus': (128, 0, 128),
          'car': (0, 128, 128),
          'cat': (128, 128, 128),
          'chair': (64, 0, 0),
          'cow': (192, 0, 0),
          'diningtable': (64, 128, 0),
          'dog': (192, 128, 0),
          'horse': (64, 0, 128),
          'motorbike': (192, 0, 128),
          'person': (64, 128, 128),
          'pottedplant': (192, 128, 128),
          'sheep': (0, 64, 0),
          'sofa': (128, 64, 0),
          'train': (0, 192, 0),
          'tvmonitor': (128, 192, 0)}

def contest(args):

    im_show = args.im_show
    weight_file = args.weight

    predictions = []
    image_list = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    t_performance_start = time.time()
    print('DATA PREPARING...')
    
    image_list = os.listdir('./Dataset/CImages')
    image_list.sort()


    print('DONE.\n')
    print('START CONTESTING...')
    

    model = resnet50().to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.load_state_dict(torch.load('./weights/' + weight_file)['state_dict'])
    model.eval()
    
    with torch.no_grad():
        for image_name in tqdm(image_list):

            result = predict(model, image_name, root_path='./Dataset/CImages/')
            for xs in result:
                predictions.append(xs)
                                   
            if im_show:
                image = cv2.imread('./Dataset/CImages/' + image_name)

                for x1y1, x2y2, class_name, _, prob in result:
                    color = COLORS[class_name]
                    cv2.rectangle(image, x1y1, x2y2, color, 2)

                    label = class_name + str(round(prob, 2))
                    text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)

                    p1 = (x1y1[0], x1y1[1] - text_size[1])
                    cv2.rectangle(image, (p1[0] - 2 // 2, p1[1] - 2 - baseline),
                                (p1[0] + text_size[0], p1[1] + text_size[1]), color, -1)

                    cv2.putText(image, label, (p1[0], p1[1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
                                8)

                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                plt.imshow(image)
                plt.show()

    if not im_show:
        print('\nSTART FILE WRIGHTING ...')
        conFile_path = "./assets/CResult.txt"
        confile = open(conFile_path,'w')
        
        for (x1, y1), (x2, y2), class_name, image_name, conf in predictions:# 추출한 물체수만큼 루프를 돈다
            image_name_list = image_name.split('.')  
            f_name = image_name_list[:-1]
            str_format = f_name[0] + ' ' + \
                    f'{CLASSES_DIC[class_name]:02d} {x1:06.2f} {y1:06.2f} {x2:06.2f} {y2:06.2f} {conf:04.3f}\n' 
            confile.write(str_format)
        
        confile.close()
        print('\nDONE.')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", type=str, default="yolov1_final.pth", help="weight file")
    parser.add_argument("--im_show", type=bool, default=False, help="image show or not")
    args = parser.parse_args()
    
    contest(args)