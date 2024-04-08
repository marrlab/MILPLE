import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18
import sys
sys.path.append("training")

class FullMILModelFeats(nn.Module):
    def __init__(self):
        super(FullMILModelFeats, self).__init__()

        self.feature_extractor = torch.load("../models/model.pt", map_location=torch.device('cpu')) # substitute with your backbone model

        self.mil = torch.load("../models/model_mil_sing_att_3classes_08062023.pt", map_location=torch.device('cpu')) # substitute with your mil model

    def forward(self, x):
        x = self.feature_extractor.conv1(x)
        x = self.feature_extractor.bn1(x)
        x = self.feature_extractor.relu(x)
        x = self.feature_extractor.maxpool(x)

        x = self.feature_extractor.layer1(x)
        x = self.feature_extractor.layer2(x)
        x = self.feature_extractor.layer3(x)
        feats = self.feature_extractor.layer4(x)

        out = self.mil(feats)


        return out, feats
