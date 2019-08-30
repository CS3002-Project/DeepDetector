import numpy as np
from torch.utils.data import Dataset
import torch
from sklearn.metrics import accuracy_score


class TimeSeries(Dataset):

    def __init__(self, path):
        self.path = path
        self.features, self.labels = [], []
        with open(path, "r") as f:
            for line in f.readlines():
                tokens = line.split("\t")
                self.features.append(np.fromstring(tokens[0], sep=" "))
                self.labels.append(int(tokens[1].strip()))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def evaluate_model(model, iterator):
    all_preds = []
    all_y = []
    for idx, batch in enumerate(iterator):
        if torch.cuda.is_available():
            x_features, x_labels = batch
            x_features, x_labels = x_features.cuda(), x_labels.cuda()
        else:
            x_features, x_labels = batch
        y_pred = model(x_features)
        predicted = torch.max(y_pred.cpu().data, 1)[1] + 1
        all_preds.extend(predicted.numpy())
        all_y.extend(x_labels.cpu().data.numpy())
    score = accuracy_score(all_y, np.array(all_preds).flatten())
    return score

