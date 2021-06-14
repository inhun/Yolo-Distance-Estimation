from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import module

from utils.utils import *


class ROIPool(nn.Module):
    def __init__(self, output_size):
        super(ROIPool, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(output_size)
        self.size = output_size
        self.fc1 = nn.Linear(2304, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 1)
        self.softplus = nn.Softplus()
        self.smoothl1 = nn.SmoothL1Loss()
        self.mse = nn.MSELoss()


    def target_detection_iou(self, box1, box2):
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

        # get the corrdinates of the intersection rectangle
        b1_x1 = b1_x1.type(torch.float64)
        b1_y1 = b1_y1.type(torch.float64)
        b1_x2 = b1_x2.type(torch.float64)
        b1_y2 = b1_y2.type(torch.float64)

        inter_rect_x1 = torch.max(b1_x1, b2_x1)
        inter_rect_y1 = torch.max(b1_y1, b2_y1)
        inter_rect_x2 = torch.min(b1_x2, b2_x2)
        inter_rect_y2 = torch.min(b1_y2, b2_y2)
        # Intersection area
        inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
            inter_rect_y2 - inter_rect_y1 + 1, min=0
        )
        # Union Area
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

        iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

        return iou

    def similar_bbox(self, detections, targets):
        rescaled_boxes = rescale_boxes(detections, 416, (480, 640))
        similar_box = list(range(len(rescaled_boxes)))
        for i in range(len(rescaled_boxes)):
            for j in range(len(targets)):
                target_xyxy = [(targets[j][0]-(targets[j][2]/2))*640, (targets[j][1]-(targets[j][3]/2))*480, (targets[j][0]+(targets[j][2]/2))*640, (targets[j][1]+(targets[j][3]/2))*480]
                target_xyxy = torch.tensor(target_xyxy)
                iou = self.target_detection_iou(rescaled_boxes[i][:4], target_xyxy)
                if iou > 0.01:
                    similar_box[i] = targets[j][-1]
                    break
                else:
                    similar_box[i] = -1
        return similar_box


    def cal_scale(self, x, detections, targets):
        targets_distance = targets[:, :4]
        square_targets = []
        
        for target_distance in targets_distance:
            x1 = (target_distance[0]-(target_distance[2]/2))*416
            y1 = ((target_distance[1]-(target_distance[3]/2))*480+80)*13/15
            x2 = (target_distance[0]+(target_distance[2]/2))*416
            y2 = ((target_distance[1]+(target_distance[3]/2))*480+80)*13/15
        
            square_targets.append([x1, y1, x2, y2])
        square_targets = torch.tensor(square_targets)
        
        scale = get_scale(square_targets)
        output_distance = []

        roi_results = []
        for i in scale:
            x1_scale = i[0]
            y1_scale = i[1]
            x2_scale = i[2]
            y2_scale = i[3]
            
            output = x[:, :, x1_scale:x2_scale+1, y1_scale:y2_scale+1]
        
            output = self.maxpool(output)
            
            output = output.view(1, -1).cuda()
            # print(output)
            roi_results.append(output)
        return roi_results

    def cal_scale_evaL(self, x, detections):
        detections = detections[:, :4]
        scale = get_scale(detections)
        output_distance = []
        roi_results = []
        for i in scale:
            x1_scale = i[0]
            y1_scale = i[1]
            x2_scale = i[2]
            y2_scale = i[3]

            output = x[:, :, y1_scale:y2_scale+1, x1_scale:x2_scale+1]
            output = self.maxpool(output)
            output = output.view(1, -1).cuda()
            roi_results.append(output)
        return roi_results

    def forward(self, x, detections, targets=None):
        if targets is not None:
            distances = targets[:, 4]
            distances = distances * 10
            # distances = distances * 10
            # print(f'disatnces = {distances}')
            # targets_distance = targets[:, :4]
            # square_targets = []
            
            # for target_distance in targets_distance:
            #     x1 = (target_distance[0]-(target_distance[2]/2))*416
            #     y1 = ((target_distance[1]-(target_distance[3]/2))*480+80)*13/15
            #     x2 = (target_distance[0]+(target_distance[2]/2))*416
            #     y2 = ((target_distance[1]+(target_distance[3]/2))*480+80)*13/15
            
            #     square_targets.append([x1, y1, x2, y2])
            # square_targets = torch.tensor(square_targets)
            
            # scale = get_scale(square_targets)
            # output_distance = []

            # roi_results = []
            # for i in scale:
            #     x1_scale = i[0]
            #     y1_scale = i[1]
            #     x2_scale = i[2]
            #     y2_scale = i[3]
                
            #     output = x[:, :, x1_scale:x2_scale+1, y1_scale:y2_scale+1]
            
            #     output = self.maxpool(output)
                
            #     output = output.view(1, -1).cuda()
            #     # print(output)
            #     roi_results.append(output)
            roi_results = self.cal_scale(x, detections, targets)

            output = torch.cat(roi_results, 0)
            # print(output.shape)
            # print(output.shape)
            output = self.fc1(output)
            output = self.fc2(output)
            output = self.fc3(output)
            output = self.softplus(output)
            # print(f'output = {output}')
            #loss = 0
            # output_distance = torch.tensor(output, requires_grad=True)


            '''
            output = x
            # output = x[:, :, y1_scale:y2_scale+1, x1_scale:x2_scale+1]
            output = self.maxpool(output)
            output = output.view(1, -1).cuda()
            # print(output.shape)
            output = self.fc1(output)
            output = self.fc2(output)
            output = self.fc3(output)
            output = self.softplus(output)
            '''

            # output_distance = torch.cuda.FloatTensor(output_distance, requires_grad=True)#.to('cpu')
            
            #print(f'output_distance = {output_distance}')
            #print(output_distance.shape)
            #print(f'distances = {distances}')
            #print(distances.shape)
            distances = distances.cuda()
            # print(f'output = {output}')
            # print(f'output = {output}')
            # print(f'distances = {distances}')
            loss = self.smoothl1(output, distances.float())
            # print(f'loss = {loss}')
            
            # print(f'output_distance = {output_distance}')
            # print(f'distances = {distances}')
            # print(f'loss = {loss}')
            return loss, output

        else:

            '''
            detections = detections[:, :4]
            scale = get_scale(detections)
            output_distance = []
            for i in scale:
                x1_scale = i[0]
                y1_scale = i[1]
                x2_scale = i[2]
                y2_scale = i[3]

                output = x[:, :, y1_scale:y2_scale+1, x1_scale:x2_scale+1]
                output = self.maxpool(output)
                output = output.view(1, -1).cuda()
            '''
            roi_results = self.cal_scale_evaL(x, detections)
            output = torch.cat(roi_results, 0)
                #   print(f'output = {output.shape}')
            output = self.fc1(output)
            output = self.fc2(output)
            output = self.fc3(output)
            output = self.softplus(output)
            print(f'output = {output}')
            

            return output


        '''
        scale = get_scale(detections)

        
        output_distance = []
        for i in scale:
            x1_scale = i[0]
            y1_scale = i[1]
            x2_scale = i[2]
            y2_scale = i[3]

            output = x[:, :, y1_scale:y2_scale+1, x1_scale:x2_scale+1]
            # output = x[:, :, x1_scale:x2_scale+1, y1_scale:y2_scale+1]
            output = self.maxpool(output)
            output = output.view(1, -1).cuda()
            output = self.fc1(output)
            output = self.fc2(output)

            output_distance.append(output)
        
        if targets is None:
            return output_distance, 0
            
        else:
            loss = 0
            box_similar_distance = self.similar_bbox(detections, targets)
            for i in range(len(box_similar_distance)):
                if box_similar_distance[i] == -1:
                    output_distance[i] = -1
            
            
            output_distance = torch.FloatTensor(output_distance).to('cpu')
            box_similar_distance = torch.FloatTensor(box_similar_distance).to('cpu')

            
            # print(f'output_distance = {output_distance}')
            # print(f'target_distance = {box_similar_distance}')
            loss = self.smoothl1(output_distance, box_similar_distance)
        '''




        


        