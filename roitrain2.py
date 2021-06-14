from __future__ import division

from roipool2 import *
from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
# from test import evaluate

from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse
import warnings 

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
import warnings
warnings.filterwarnings("ignore", category=UserWarning)





if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device: ', device)

    data_config = parse_data_config('config/cafe_distance.data')
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    model = Darknet('config/yolov3-tiny.cfg', 416).to(device)

    model.load_state_dict(torch.load('checkpoints_cafe_distance/tiny1_2500.pth', map_location=device))
    model.eval()

    dataset = ListDataset(train_path, augment=True, multiscale=True)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    model_distance = ROIPool((3, 3)).to(device)
    model_parameters = filter(lambda p: p.requires_grad, model_distance.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Params: ', params)
    
    optimizer = torch.optim.Adam(model_distance.parameters())

    
    a = []
    for epoch in range(2000):

        warnings.filterwarnings('ignore', category=UserWarning)
        for batch_i, (img_path, imgs, targets, targets_distance) in enumerate(dataloader):
            

            imgs = Variable(imgs.to(device))
            with torch.no_grad():

                featuremap, detections = model(imgs)
            # print(featuremap.shape)
            featuremap = Variable(featuremap.to(device))
            
            detections = non_max_suppression(detections, 0.8, 0.4)
            targets_distance = torch.tensor(targets_distance[0])
            targets_distance = Variable(targets_distance, requires_grad=True)
            


            if detections is not None:
                detections[0] = Variable(detections[0], requires_grad=True)
                

                loss, outputs = model_distance(featuremap, detections[0], targets=targets_distance)
                # loss = torch.tensor([loss]).to(device)
                # loss.requires_grad = True
                # print(model_distance.fc1.bias)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # print(model_distance.fc1.bias)

            # print(batch_i)
        print(epoch)

            # print(featuremap)
        if epoch % 10 == 0:
            optimizer.param_groups[0]['lr'] /= 2

        if epoch % 10 == 0:
            torch.save(model_distance.state_dict(), f'checkpoints_distance11/tiny1_{epoch}.pth')
