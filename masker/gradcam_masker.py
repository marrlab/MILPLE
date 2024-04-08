import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import gzip

from ..FullMILModel import FullMILModel


TOP_CELLS_COUNT = 50 #Do gradcam on top 50 cells in a bag
PATIENT_DATA_FOLDER = #Add path to your patien data
'''
Substitute patinet_top_cells_mds.dat with your file in the following format:
{'patient name': [indexes of the most important cells of the bag based on attention from max to min], ...}
'''

# load model
model = FullMILModel().cuda()


for param in model.parameters():
    param.requires_grad = True

with open("patinet_top_cells_mds.dat", "rb") as f:
    patient_data = pickle.load(f)


target_layers = [model.feature_extractor.layer4[-2]]

result = {}
for patient_name in list(patient_data.keys()):
    with open(PATIENT_DATA_FOLDER + patient_name + ".dat", 'rb') as f:
        imgs = pickle.load(f)[0]

    imgs = imgs[patient_data[patient_name][0][:TOP_CELLS_COUNT]]
    timgs = imgs.cuda()



    # DO GRADCAM HERE
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    targets = [ClassifierOutputTarget(np.argmax(patient_data[patient_name][1]))]
    grayscale_cam = cam(input_tensor=timgs, targets=targets)
    gc = grayscale_cam
    result[patient_name] = gc
    #### END of GRADCAM


    print(patient_name, "done")



with gzip.open("masks/gradcam_masks.dat", "wb") as f:
    pickle.dump(result, f)

