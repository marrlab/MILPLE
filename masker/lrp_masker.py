import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from zennit.attribution import Gradient, SmoothGrad
from zennit.core import Stabilizer
from zennit.composites import EpsilonGammaBox, EpsilonPlusFlat, EpsilonAlpha2Beta1, EpsilonAlpha2Beta1Flat
from zennit.composites import SpecialFirstLayerMapComposite, NameMapComposite
from zennit.image import imgify, imsave
from zennit.rules import Epsilon, ZPlus, ZBox, Norm, Pass, Flat, Gamma, AlphaBeta
from zennit.types import Convolution, Activation, AvgPool, Linear as AnyLinear
from zennit.types import BatchNorm, MaxPool
from zennit.torchvision import VGGCanonizer, ResNetCanonizer
import gzip

from ..FullMILModel import FullMILModel

TOP_CELLS_COUNT = 50  #Do LRP on top 50 cells in a bag
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

canonizer = ResNetCanonizer()

low, high = torch.tensor([[[[[0.]]] * 3], [[[[1.]]] * 3]])

composite = EpsilonGammaBox(low=low, high=high, canonizers=[canonizer])

result = {}
for patient_name in list(patient_data.keys()):
    with open(PATIENT_DATA_FOLDER + patient_name + ".dat", 'rb') as f:
        imgs = pickle.load(f)[0]

    imgs = imgs[patient_data[patient_name][0][:TOP_CELLS_COUNT]]
    timgs = imgs.cuda()

    # DO LRP HERE
    target = torch.eye(3)[[np.argmax(patient_data[patient_name][1])]]
    with Gradient(model=model, composite=composite) as attributor:
        output, attribution = attributor(timgs, target.cuda())

    relevance = attribution.cpu().sum(1)
    result[patient_name] = relevance
    #### END of LRP

    print(patient_name, "done")

with gzip.open("masks/lrp_masks.dat", "wb") as f:
    pickle.dump(result, f)
