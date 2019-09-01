import numpy as np
import os
import random


def generate_sine_waves(output_dir, size, dim, num):
    label_array = range(5)
    interval = 0.05

    for idx in range(size):
        label = random.choice(label_array)
        sine_waves = create_sine_waves(num, label, dim)

        with open(os.path.join(output_dir, "{}.txt".format(idx)), 'w') as f:
            for i, wave in enumerate(sine_waves):
                features = " ".join([str(x) for x in list(wave)])
                time_stamp = i * interval
                f.writelines("\t".join([str(time_stamp), features, str(label)]) + "\n")


def create_sine_waves(num, label, dim):
    sine_waves = []
    for d in range(dim):
        x = np.linspace(-np.pi * np.random.rand(), np.pi * np.random.rand(), num)
        y = np.sin(x-d*label) + np.random.normal(0, 0.5)
        sine_waves.append(y)
    return np.transpose(np.array(sine_waves))
