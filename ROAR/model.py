import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np


class SCEMILA(nn.Module):

    def __init__(self, class_count, multi_attention, device):

        '''Initialize model. Takes in parameters:
        - class_count: int, amount of classes --> relevant for output vector
        - multicolumn: boolean. Defines if multiple attention vectors should be used.
        - device: either 'cuda:0' or the corresponding cpu counterpart.
        '''

        super(SCEMILA, self).__init__()

        self.L = 500                    # condense every image into self.L features (further encoding before actual MIL starts)
        self.D = 128                    # hidden layer size for attention network

        self.class_count = class_count
        self.multicolumn = multicolumn
        self.device = device
 


        # feature extractor before multiple instance learning starts
        self.FT_DIM_IN = 512
        self.ftr_proc = nn.Sequential(
            nn.Conv2d(self.FT_DIM_IN, int(self.FT_DIM_IN*1.5), kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(int(self.FT_DIM_IN*1.5), int(self.FT_DIM_IN*2), kernel_size=2),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(output_size=(1,1)),
            nn.Flatten(),
            nn.Linear(int(self.FT_DIM_IN*2), self.L),
            nn.ReLU(),
        )
        
        
        
        # Networks for single attention approach
        ##### attention network (single attention approach)
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, 1)
        )
        
        
        ##### classifier (single attention approach)
        self.classifier = nn.Sequential(
            nn.Linear(self.L, 64),
            nn.ReLU(),
            nn.Linear(64, self.class_count)
        )
        
        
        
        
        

    def forward(self, x):
        '''Forward pass of bag x through network. '''
        #normal mil
        #!!!!! adjust
        #ft = x.squeeze(0)
        
        #single IBA
        ft = x
        ft = self.ftr_proc(ft)
        '''
        prediction = self.classifier(ft)
            
        return prediction
        '''
        # switch between multi- and single attention classification
        if(self.multicolumn):
            prediction = []
            bag_feature_stack = []
            attention_stack = []
            # calculate attention
            att_raw = self.attention_multi_column(ft)
            att_raw = torch.transpose(att_raw, 1, 0)

            # for every possible class, repeat
            for a in range(self.class_count):
                # softmax + Matrix multiplication
                att_softmax = F.softmax(att_raw[a, ...][None, ...], dim=1)
                bag_features = torch.mm(att_softmax, ft)
                bag_feature_stack.append(bag_features)
                # final classification with one output value (value indicating this specific class to be predicted)
                pred = self.classifier_multi_column[a](bag_features)
                prediction.append(pred)
                
            prediction = torch.stack(prediction).view(1, self.class_count)
            bag_feature_stack = torch.stack(bag_feature_stack).squeeze()
            # final softmax to obtain probabilities over all classes
            # prediction = F.softmax(prediction, dim=1)

            return prediction, att_raw, F.softmax(att_raw, dim=1)
        else:
            # calculate attention
            att_raw = self.attention(ft)
            att_raw = torch.transpose(att_raw, 1, 0)
            # Softmax + Matrix multiplication
            att_softmax = F.softmax(att_raw, dim=1)
            bag_features = torch.mm(att_softmax, ft)
            # final classification
            prediction = self.classifier(bag_features)
            
            return prediction, att_raw, F.softmax(att_raw, dim=1)
        
        
        
        


