import os
import numpy as np
import pandas as pd
import json


base_dir = '/home/jeff/project/poc-project/asl_signs/training/asl_signs/'  # path to competition data
ROWS_PER_FRAME = 543  # number of landmarks per frame


def load_relevant_data_subset(pq_path):
    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)


def read_dict(file_path):
    path = os.path.expanduser(file_path)
    with open(path, "r") as f:
        dic = json.load(f)
    return dic


train = pd.read_csv(base_dir + '/train.csv')
label_index = read_dict(f"{base_dir}/sign_to_prediction_index_map.json")
index_label = dict([(label_index[key], key) for key in label_index])
train["label"] = train["sign"].map(lambda sign: label_index[sign])
print(train.shape)

xyz, Y = [], []

for i in range(len(train[:10000])):
    sample = train.loc[i]
    yy = load_relevant_data_subset(base_dir + sample['path'])
    lab = sample['label']

    xyz.append(yy)
    Y.append(lab)
    if i % 1000 == 0:
        print(i)

os.makedirs('gen_xyz', exist_ok=True)
np.save('gen_xyz/data.npy', np.array(xyz, dtype=object))
np.save('gen_xyz/Y.npy', np.array(Y))

