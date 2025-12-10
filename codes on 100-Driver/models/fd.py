import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import random
import numpy as np
import torchvision as tv
from torch.utils.data import TensorDataset,DataLoader

import torchvision
import torchvision.datasets as Dataset
import torchvision.models as models
import torch.optim as optim
import torchvision.transforms as transforms
class learningDIF(nn.Module):
    def __init__(self,driver_num):
        super(learningDIF,self).__init__()
        
        model = models.mobilenet_v2(pretrained=True)#MSFIN_basic()
        
        
        self.features = model.features
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        dim = 160



        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(in_features=1280, out_features=22, bias=True)
        )
        
        self.classifier_d = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(in_features=1280, out_features=driver_num, bias=True)
        )
        

        self.classifier_b = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(in_features=1280, out_features=22, bias=True)
        )
        
        self.gate1 = nn.Parameter(torch.ones(1280)+1)
        self.gate2 = nn.Parameter(torch.ones(7,7)+1)
        
    def getGAP(self,f):
        return self.avgpool(f).squeeze(-1).squeeze(-1)
    def getBehaviorPreds(self,f):
        #f = self.features1_b(f)
        f = self.getGAP(f)
        pred = self.classifier_b(f)
        return pred
    def getIDPreds(self,f):
        #f = self.features1_d(f)
        f = self.getGAP(f)
        pred = self.classifier_d(f)
        return pred
    def forward_visual(self,x,is_warming = True):
        f = self.features(x)
        
        avgf = self.getGAP(f)
        preds = self.classifier(avgf)

        #f = f.detach()
        
        gate1 = f.mean(1).unsqueeze(1)  * self.gate1.unsqueeze(-1).unsqueeze(-1)
        gate2 = (f.sum(-1).sum(-1)/49).unsqueeze(-1).unsqueeze(-1) * self.gate2.unsqueeze(0)
        gate1 = nn.Sigmoid()(gate1)
        gate2 = nn.Sigmoid()(gate2)
        gate = gate1 * gate2
        
        f_driver = f * gate

        f_behavior = f * (1-gate)
        
        preds1 = self.getIDPreds(f_driver) #能区分司机
        preds2 = self.getIDPreds(f_behavior) #不能区分司机

        preds3 = self.getBehaviorPreds(f_behavior) #能区分行为
        preds4 = self.getBehaviorPreds(f_driver) #不能区分行为


        #return preds,preds1,preds2,preds3,preds4
        return f,'gate1','gate2',f_driver,f_behavior,preds1,preds2,preds3,preds4,preds
    def forward(self,x):
        f = self.features(x)
        
        avgf = self.getGAP(f)
        preds = self.classifier(avgf)

        #f = f.detach()
        
        gate1 = f.mean(1).unsqueeze(1)  * self.gate1.unsqueeze(-1).unsqueeze(-1)
        gate2 = (f.sum(-1).sum(-1)/49).unsqueeze(-1).unsqueeze(-1) * self.gate2.unsqueeze(0)
        gate1 = nn.Sigmoid()(gate1)
        gate2 = nn.Sigmoid()(gate2)
        gate = gate1 * gate2
        
        f_driver = f * gate

        f_behavior = f * (1-gate)
        
        preds1 = self.getIDPreds(f_driver) #能区分司机
        preds2 = self.getIDPreds(f_behavior) #不能区分司机

        preds3 = self.getBehaviorPreds(f_behavior) #能区分行为
        preds4 = self.getBehaviorPreds(f_driver) #不能区分行为


        return preds,preds1,preds2,preds3,preds4
