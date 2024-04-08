import sys
import matplotlib.pyplot as plt
import cv2

from ..input_iba.models import build_attributor
from ..input_iba.datasets import build_dataset, build_pipeline

import mmcv
import os
import shutil
import ssl
import json

import numpy as np
from PIL import Image

#import seaborn as sns
import matplotlib.pyplot as plt
import sys  
import os, time
import argparse as ap
import torchvision.models as models
import torch.nn.functional as F

# strange workaround for bug: libgcc_s.so.1 must be installed for pthread_cancel to work
import ctypes
libgcc_s = ctypes.CDLL('libgcc_s.so.1')

import torch
import torch.nn as nn
import torch.multiprocessing
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
torch.multiprocessing.set_sharing_strategy('file_system')
from ..FullMILModel import FullMILModel

from ..masked_dataloader import MaskedDataLoader
from ..model import SCEMILA  
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import pickle


cfg_aml = mmcv.Config.fromfile('mil_struct.py')
cfg_aml.attribution_cfg.input_iba.beta = 10.0 #20.0
cfg_aml.attributor.layer = "feature_extractor.layer1.0.conv1" #4


est_loader_aml = torch.utils.data.DataLoader(MaskedAMLDataLoader(), num_workers=1)
dataiter = iter(est_loader_aml)
device = 'cuda:0'


item =dataiter.next()
input_tensor = item['input']
target = item['target'].item()
patient = item['input_name'][0]

input_tensor = input_tensor.squeeze().to(device)

TOP_CELLS = 50 #number of top cells
PATIENT_DATA_FOLDER = #Add path to your patien data
'''
Substitute patinet_top_cells_mds.dat with your file in the following format:
{'patient name': [indexes of the most important cells of the bag based on attention from max to min], ...}
'''

with open("patinet_top_cells_mds.dat", "rb") as f:
    patient_data = pickle.load(f)
    
    
for i in range(len(dataiter)): 
    
    
    item =dataiter.next()
    input_tensor = item['input']
    target = item['target'].item()
    patient = item['input_name'][0]
    
    pat_npy = patient + ".npy"
    #if pat_npy in done:
    #    continue
    
    print("Working on patient ", patient)
    
    
    input_tensor = input_tensor.squeeze().to(device)
    
    
    input_tensor_att = input_tensor[patient_data[patient][0][:TOP_CELLS], :,:,:]
    
    attributor = build_attributor(cfg_aml.attributor, default_args=dict(device=device))

    attributor.estimate(est_loader_aml, cfg_aml.estimation_cfg)
    
    predicted_label = torch.argmax(attributor.classifier(input_tensor_att.to(device))).item()
                                   
    input_tensor = input_tensor_att.float().to(device)
    attributor.make_attribution(input_tensor,
                            predicted_label,
                            attribution_cfg=cfg_aml.attribution_cfg) 
                                   
    
    attributor.show_feat_mask("iba_masks",patient,out_file=None, show=True)
    attributor.show_input_mask("iba_masks",patient,out_file=None, show=True)


