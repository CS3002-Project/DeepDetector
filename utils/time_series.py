import matplotlib.pyplot as plot
import numpy as np
import os
from torch.utils.data import Dataset


class TimeSeries(Dataset):

    def __init__(self, path):
        self.path = path
        self.size = len(os.listdir(path))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        times, features, labels = [], [], []
        with open(os.path.join(self.path, "{}.txt".format(idx))) as f:
            for line in f.readlines():
                tokens = line.split("\t")
                times.append(float(tokens[0]))
                features.append(np.fromstring(tokens[1], sep=" "))
                labels.append(int(tokens[2].strip()))

        return np.array(times), np.array(features), np.array(labels)

    @staticmethod
    def plot(times, features, labels):
        features = list(features.T)
        args = []
        for dim_values in features:
            args += [times, dim_values]
        args += [times, labels]
        plot.plot(*args)
        plot.title("Time series")
        plot.xlabel('Time')
        plot.ylabel('Readings')
        plot.legend([str(i) for i in range(len(features))] + ["labels"])
        plot.grid(True, which='both')
        plot.axhline(y=0, color='k')
        plot.show()
