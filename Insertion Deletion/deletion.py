import sys
import torch
import numpy as np
import pickle
import gzip

from tqdm import tqdm

from FullMILModelOut import FullMILModelOut

WHAT = sys.argv[1]
PATIENT_DATA_FOLDER = #Add path to your patien data
'''
Substitute patinet_top_cells_mds.dat with your file in the following format:
{'patient name': [indexes of the most important cells of the bag based on attention from max to min], ...}
'''

def mask(cm, img, percent):
    f = np.quantile(cm, 1-percent, axis=(1, 2))
    for i in range(len(cm)):
        img[i].permute(1, 2, 0)[cm[i] > f[i]] = torch.tensor([0,0,0]).float().cuda()
    return img


cuda = torch.cuda.is_available()

# load model
model = FullMILModel()
if cuda:
    model = model.cuda()
model.eval()


for param in model.parameters():
    param.requires_grad = False

with gzip.open("../masker/" + WHAT + "_masks.dat", "rb") as f:
    cam_masks = pickle.load(f)

with open("../masker/patinet_top_cells_mds.dat", "rb") as f:
    patient_top_cells = pickle.load(f)



result = {}

for patientname in tqdm(list(cam_masks.keys())):
    result[patientname] = []
    
    with open(PATIENT_DATA_FOLDER + patientname + ".dat", 'rb') as f:
    
        imgs = pickle.load(f)[0]

    imgs = imgs[patient_top_cells[patientname][0][:len(cam_masks[patientname])]]
    gt = np.argmax(patient_top_cells[patientname][1][0])

    if cuda:
        imgs = imgs.cuda()
    imgs_bk = imgs.clone()

    for percent in range(100):  
        percent = percent / 100
        imgs = imgs_bk.clone()
        imgs = mask(cam_masks[patientname], imgs, percent)

        out = model(imgs)[0]
        del imgs

        if torch.argmax(out[0]).cpu() == gt:
            result[patientname].append(1)
        else:
            result[patientname].append(0)
    del imgs_bk
    torch.cuda.empty_cache()




with open("deletion_" + WHAT + ".dat", "wb") as f:
    pickle.dump(result, f)
