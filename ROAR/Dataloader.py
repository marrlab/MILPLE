import os
from PIL import Image
import numpy as np
import cv2
from torch.utils.data import Dataset


def load_image(path):
    img = cv2.imread(path)
    # try:
    #     img = Image.open(path)
    #     img = np.asarray(img)[:,:,:3]
    # except:
    #     return False, None
    if img is None or img.shape[0] != 363:
        return False, None
    if img.shape[0] != 144 or img.shape[1] != 144:
        img = cv2.resize(img, (144,144))

    img = (img - np.min(img))/np.ptp(img)
    # img = img / 255
    img = np.moveaxis(img, -1, 0)
    return True, img


class PatientLoader(Dataset):

    def __init__(self, root):
        self.root = root
        self.patient_list = os.listdir(root)

    def __len__(self):
        return len(self.patient_list)

    def __getitem__(self, index):
        imgs = []
        for img in os.listdir(os.path.join(self.root, self.patient_list[index])):
            if img[0] != ".":
                st, im = load_image(os.path.join(self.root, self.patient_list[index], img))
            
                if st:
                    imgs.append(im)
        
        return self.patient_list[index], np.array(imgs)
