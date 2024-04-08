import pickle
import numpy as np
import gzip
import os


TOP_CELLS_COUNT = 50
'''
Substitute patinet_top_cells_mds.dat with your file in the following format:
{'patient name': [indexes of the most important cells of the bag based on attention from max to min], ...}
'''

with open("patinet_top_cells.dat", "rb") as f:
    patient_data = pickle.load(f)

input_iba_root = "iba_masks/input"
feat_iba_root = "iba_masks/feat"

result_input = {}
result_feat = {}
for patient_name in list(patient_data.keys()):
 

    input_mask = np.load(os.path.join(input_iba_root, patient_name + ".npy"))
    feat_mask = np.load(os.path.join(feat_iba_root, patient_name + ".npy"))

    result_input[patient_name] = input_mask
    result_feat[patient_name] = feat_mask

    print(patient_name, "done")

with gzip.open("masks/input_iba_masks.dat", "wb") as f:
    pickle.dump(result_input, f)

with gzip.open("masks/feat_iba_masks.dat", "wb") as f:
    pickle.dump(result_feat, f)
