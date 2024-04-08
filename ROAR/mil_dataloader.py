
import os
import torch
import numpy as np
import pickle
from sklearn.model_selection import train_test_split


classes = {
    CLASS0: 0,
    CLASS1: 1,
    CLASS2: 2,
    ...
} #Paste your classes here in the right format

class MilDataLoader:
    
    def __init__(self, WHAT, PERCENT, train):

        with open("features/features_"+ WHAT + "_" + str(PERCENT) +".pkl", "rb") as f:
            data = pickle.load(f)

        self.patient_list = self.get_patient_list(train)

        self.data = dict((k, data[k]) for k in data.keys() if k in self.patient_list)
        print("loading done")

    def __len__(self):
        
        length = len(self.patient_list)
        return length 
    
    def __getitem__(self, index):

        patient = self.patient_list[index]
        features = self.data[patient]["features"]
        label = self.data[patient]["label"]
        return features, label


    def get_patient_list(self, train):
        if os.path.exists("split.dat"):
            with open("split.dat", "rb") as f:
                train_pats, test_pats = pickle.load(f)
        else:
            patients = list(self.data.keys())
            labels = [classes[self.data[p]['label']] for p in patients]
            train_pats, test_pats = train_test_split(
                patients,
                test_size=0.3,
                shuffle=True,
                stratify=labels)

            with open("split.dat", "wb") as f:
                pickle.dump([train_pats, test_pats], f)

        return train_pats if train else test_pats
