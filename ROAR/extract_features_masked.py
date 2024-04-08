import sys
import torch
import pickle
import numpy as np
import gzip
from tqdm import tqdm
from ..FullMILModelFeats import FullMILModelFeats

PERCENT = float(sys.argv[2])
WHAT = sys.argv[1]
PATIENT_DATA_FOLDER = #Add path to your patien data
'''
Substitute patinet_top_cells_mds.dat with your file in the following format:
{'patient name': [indexes of the most important cells of the bag based on attention from max to min], ...}
'''

def mask_rnd(img):
    ret = torch.zeros_like(img)
    for i in range(len(imgs)):
        perm = torch.randperm(144*144)
        indices = perm[:int(144*144*PERCENT)]
        # indices = random.sample(range(144*144), int(144*144*percent))
        # indices = np.unravel_index(indices, (144,144))
        indices = torch.floor_divide(indices, 144), indices % 144
        ret[i].permute(1, 2, 0)[indices] = img[i].permute(1, 2, 0)[indices]

    return ret

def mask(cm, img):
    f = np.quantile(cm, PERCENT, axis=(1, 2))
    avg_pix = torch.mean(img, dim=(2, 3))
    for i in range(len(cm)):
        img[i].permute(1, 2, 0)[cm[i] > f[i]] = torch.tensor([0,0,0]).float().cuda()##avg_pix[i]
    return img


cuda = torch.cuda.is_available()

# load model
model = FullMILModel()
if cuda:
    model = model.cuda()
model.eval()

for param in model.parameters():
    param.requires_grad = False

sd = 50
if WHAT != "rnd":
    with gzip.open("../masker/masks/" + WHAT + "_masks.dat", "rb") as f:
        cam_masks = pickle.load(f)
    sd = 50

with open("../masker/patinet_top_cells_mds.dat", "rb") as f:
    patient_top_cells = pickle.load(f)

loader_kwargs = {'num_workers': 30, 'pin_memory': True} if cuda else {}
path_masks = "../masker/iba_masks/input"
masks = os.listdir(path_masks)
masks = [f.split('.')[0] for f in masks]


features = {}

for patientname in tqdm(list(patient_top_cells.keys())):
    if patientname not in masks:
        continue
    with open(PATIENT_DATA_FOLDER + patientname + ".dat", 'rb') as f:
    
        imgs = pickle.load(f)[0]

    imgs = imgs[patient_top_cells[patientname][0][:sd]]

    if cuda:
        imgs = imgs.cuda()

    if WHAT == "rnd":
        imgs = mask_rnd(imgs)
    else:
        imgs = mask(cam_masks[patientname], imgs)
    


    _, feats = model(imgs)
    features[patientname] = {"features": feats.cpu().detach(),
                             "label": patient_top_cells[patientname][3]}

with open("features/features" + WHAT + "_" + str(PERCENT) + ".pkl", "wb") as f:
    pickle.dump(features, f)
