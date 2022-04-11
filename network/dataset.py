import torch
from torch.utils.data import Dataset
import os
import numpy as np
from tqdm import tqdm

class CNNDataset(Dataset):
    def __init__(self, x, y):
        x = torch.FloatTensor(np.transpose(x, [0, 2, 1]))
        y = torch.LongTensor(y)
        self.x_data = x
        self.y_data = y
    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

class LSTMDataset(Dataset):
    def __init__(self, x, y, seq_len, n_slidings=4, step_sec=1):
        x = torch.tensor(np.transpose(x, [0, 2, 1]))
        y = torch.tensor(y)
        x_tmp = []
        fs = x[0].shape[-1] // 30
        sliding_step = int(fs * step_sec)
        print('Processing data augmentation')
        for i in tqdm(range(1, len(x))): # b, c, i
            sliding_seriese = []
            for j in range(0, n_slidings * sliding_step, sliding_step):
                x_aug1 = x[i-1, 0, j:].unsqueeze(0)
                x_aug2 = x[i, 0, :j].unsqueeze(0)
                x_aug = torch.cat([x_aug1, x_aug2], dim=-1)
                sliding_seriese.append(x_aug)
            sliding_seriese = torch.cat(sliding_seriese, dim=-2)
            x_tmp.append(sliding_seriese.unsqueeze(0))
        x = torch.cat(x_tmp)
        x_list = []
        y_list = []
        for i in range(0, len(x), seq_len):
            s, c, l = x[i:i+seq_len].shape
            try:
                x_list.append(x[i:i+seq_len].view(1, seq_len, c, l))
                y_list.append(y[i:i+seq_len].view(1, -1))
            except:
                pass
        x = torch.cat(x_list)
        y = torch.cat(y_list)

        self.x_data = x
        self.y_data = y

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

def read_data(file_name):
    x = np.load(file_name)["x"]
    y = np.load(file_name)["y"]
    return x, y

def load_dataset(data_dir, input_type, cv_idx):
    train_x, train_y = [], []
    val_x, val_y = [], []
    test_x, test_y = [], []

    file_path = data_dir + input_type
    file_list = os.listdir(file_path)
    n_files = len(file_list)
    train_idx = [i for i in range(n_files)]
    if input_type == 'SHHS':
        n_test = len(file_list) // 20
        if cv_idx == 20:
            test_idx = train_idx[(cv_idx - 1) * n_test:]
        else:
            test_idx = train_idx[(cv_idx - 1) * n_test: cv_idx * n_test]
        for idx in test_idx:
            train_idx.remove(idx)
        val_idx = train_idx[:int(len(train_idx) * 0.1)]
        for idx in val_idx:
            train_idx.remove(idx)
        fs = 125
        n_classes = 5

    else:
        if cv_idx == 14:
            test_idx = [26]
            val_idx = [27, 28]
        elif cv_idx < 14:
            test_idx = [2 * (cv_idx - 1), 2 * (cv_idx - 1) + 1]
            val_idx = [2 * (cv_idx - 1) + 2, 2 * (cv_idx - 1) + 3]
        elif cv_idx > 14:
            test_idx = [2 * (cv_idx - 1) - 1, 2 * (cv_idx - 1)]
            val_idx = [2 * (cv_idx - 1) - 3, 2 * (cv_idx - 1) - 2]
        if cv_idx == 13:
            val_idx == [26]
        for idx in test_idx:
            train_idx.remove(idx)
        for idx in val_idx:
            train_idx.remove(idx)
        fs = 100
        n_classes = 5

    print('Train file indicies', train_idx)
    print('Validation file indicies', val_idx)
    print('Test file indicies', test_idx)
    train_files, val_files, test_files = [], [], []

    for idx in test_idx:
        test_files.append(file_list[idx])
        x, y = read_data(file_path + '/' + file_list[idx])
        test_x.append(x)
        test_y.append(y)

    for idx in val_idx:
        val_files.append(file_list[idx])
        x, y = read_data(file_path + '/' + file_list[idx])
        val_x.append(x)
        val_y.append(y)

    for idx in train_idx:
        train_files.append(file_list[idx])
        x, y = read_data(file_path + '/' + file_list[idx])
        train_x.append(x)
        train_y.append(y)

    if (val_files in train_files) or (val_idx in train_idx) or \
            (test_files in train_files) or (test_idx in train_idx) or \
            (test_files in val_files) or (test_idx in val_idx):
        print('train', train_idx)
        print('test', test_idx)
        print('val', val_idx)
        print('Train data:', train_files)
        print('Test data:', test_files)
        print('Validation data:', val_files)
        raise KeyError('Test or Validation file is contained in training set')

    train_x = np.concatenate(train_x)
    train_y = np.concatenate(train_y)
    test_x = np.concatenate(test_x)
    test_y = np.concatenate(test_y)
    val_x = np.concatenate(val_x)
    val_y = np.concatenate(val_y)

    return (train_x, train_y), (val_x, val_y), (test_x, test_y), fs, n_classes

