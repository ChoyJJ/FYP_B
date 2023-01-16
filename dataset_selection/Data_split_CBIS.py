import pandas as pd
import numpy as np
import os
import pathlib

PATH = "/mnt/d/Datasets/CBIS-DDSM/manifest-ZkhPvrLo5216730872708713142/CBIS-DDSM"
train_csv_path = '/mnt/d/Datasets/CBIS-DDSM/calc_case_description_train_set.csv'
df = pd.read_csv (train_csv_path)
# print (df.pathology)
pathology = df.pathology.to_numpy()
patient = df.patient_id.to_numpy()
location = df.left_or_right_breast.to_numpy()
view = df.image_view.to_numpy()
benign_wc = []
patient_id = []
result = []
for count,x in enumerate(pathology) :
    # print(count)
    if x == 'BENIGN_WITHOUT_CALLBACK':
        benign_wc.append(x)
        patient_id.append(patient[count])
        if count == 0:
            # patient_id.append(patient[count])
            dcm_path=PATH + "/Calc-Training_" + str(patient[count])+ "_" + str(location[count]) + "_" + str(view[count])
            result.append(list(pathlib.Path(dcm_path).rglob("*.dcm")))
        elif (count > 0):
            if (patient[count] != patient_id[:-2]):
                # patient_id.append(patient[count])
                dcm_path=PATH + "/Calc-Training_" + str(patient[count])+ "_" + str(location[count]) + "_" + str(view[count])
                result.append(list(pathlib.Path(dcm_path).rglob("*.dcm")))
        



print(result)

# do something
print(len(result))
