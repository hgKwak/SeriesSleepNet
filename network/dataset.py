import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, psg, hyp, istrain, is1, seq_len):
        self.x_data = []
        self.y_data = []

        for i in range(len(psg)):
            if is1:
                if istrain:
                    self.x_data.extend(psg[i])
                    self.y_data.extend(hyp[i])

                else:
                    self.x_data.append(psg[i])
                    self.y_data.append(hyp[i])

            else:
                psg_tmp = []
                hyp_tmp = []
                if istrain:
                    for j in range(0, (len(psg[i]) - seq_len + 1)):
                        psg_tmp.append(psg[i][j:j + seq_len])
                        hyp_tmp.append(hyp[i][j:j + seq_len])

                    self.x_data.extend(psg_tmp)
                    self.y_data.extend(hyp_tmp)

                else:
                    for j in range(0, (len(psg[i]) - seq_len + 1)):
                        psg_tmp.append(psg[i][j:j + seq_len])
                        hyp_tmp.append(hyp[i][j:j + seq_len])

                    self.x_data.append(psg_tmp)
                    self.y_data.append(hyp_tmp)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_data[idx])
        y = self.y_data[idx]
        return x, y