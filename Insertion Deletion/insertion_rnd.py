import sys
import torch
import numpy as np
import pickle
import gzip
import random

from tqdm import tqdm

from FullMILModelOut import FullMILModelOut

PATIENT_DATA_FOLDER = #Add path to your patien data
'''
Substitute patinet_top_cells_mds.dat with your file in the following format:
{'patient name': [indexes of the most important cells of the bag based on attention from max to min], ...}
'''

WHAT = "rnd"

def mask(img, percent):
    ret = torch.zeros_like(img)
    for i in range(len(imgs)):
        perm = torch.randperm(144*144)
        indices = perm[:int(144*144*percent)]
        # indices = random.sample(range(144*144), int(144*144*percent))
        # indices = np.unravel_index(indices, (144,144))
        indices = torch.floor_divide(indices, 144), indices % 144
        ret[i].permute(1, 2, 0)[indices] = img[i].permute(1, 2, 0)[indices]

    return ret


cuda = torch.cuda.is_available()

# load model
model = FullMILModel()
if cuda:
    model = model.cuda()
model.eval()


for param in model.parameters():
    param.requires_grad = False


with open("../masker/patinet_top_cells_mds.dat", "rb") as f:
    patient_top_cells = pickle.load(f)



result = {}

for patientname in tqdm(list(patient_top_cells.keys())):
    result[patientname] = []
    
    with open(PATIENT_DATA_FOLDER + patientname + ".dat", 'rb') as f:
  
        imgs = pickle.load(f)[0]

    imgs = imgs[patient_top_cells[patientname][0][:50]]
    gt = np.argmax(patient_top_cells[patientname][1][0])

    if cuda:
        imgs = imgs.cuda()
    imgs_bk = imgs.clone()

    for percent in range(100):  
        percent = percent + 1
        percent = percent / 100
        imgs = imgs_bk.clone()
        imgs = mask(imgs, percent)

        out = model(imgs)[0]
        del imgs

        if torch.argmax(out[0]).cpu() == gt:
            result[patientname].append(1)
        else:
            result[patientname].append(0)
    del imgs_bk
    torch.cuda.empty_cache()




with open("insertion_" + WHAT + ".dat", "wb") as f:
    pickle.dump(result, f)
