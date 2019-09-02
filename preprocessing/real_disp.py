import numpy as np
import os


def preprocess_real_disp(input_path, output_dir,
                         num_samples,
                         start_feature_idx=None,
                         end_feature_idx=None,
                         min_warm_up_sec=1., max_window_sec=3.):
    time_stamps = []
    time_stamp = 0.
    start_sec = None
    start_feature_idx = 0 if start_feature_idx is None else start_feature_idx
    end_feature_idx = 117 if end_feature_idx is None else end_feature_idx
    data = []
    with open(input_path, "r") as f:
        for i, line in enumerate(f.readlines()):
            if i % 1000 == 0:
                print("Loaded {} Real Disp lines".format(i))
            tokens = line.split("\t")
            if i == 0:
                start_sec = float(tokens[0]) + float(tokens[1]) / 10.0 ** 6
                time_stamp = 0.
            else:
                time_stamp = float(tokens[0]) + float(tokens[1]) / 10.0 ** 6 - start_sec
            label = tokens[-1].strip()
            readings = tokens[2: 119][start_feature_idx: end_feature_idx]
            data.append([str(time_stamp), " ".join(readings), label])
            time_stamps.append(time_stamp)

    data_size = len(data)
    if data_size > 0:
        sample_indices = []
        interval = time_stamp / data_size
        max_sample_size = int(max_window_sec / interval) + 1
        warm_up_size = int(min_warm_up_sec / interval) + 1
        while len(sample_indices) < num_samples:
            if len(sample_indices) % 10 == 0:
                print("Sampled {} Real Disp lines".format(len(sample_indices)))
            start_idx = np.random.randint(low=0, high=data_size-max_sample_size)
            sample_size = np.random.randint(low=1, high=max_sample_size)
            if (start_idx, start_idx+sample_size) not in sample_indices and \
                    data[start_idx+sample_size-warm_up_size][2] == data[start_idx+sample_size][2]:
                sample_indices.append((start_idx, start_idx+sample_size))

        for sample_idx, (start_idx, end_idx) in enumerate(sample_indices):
            with open(os.path.join(output_dir, "{}.txt".format(sample_idx)), "w") as f:
                sampled_data = data[start_idx: end_idx]
                for points in sampled_data:
                    f.writelines("\t".join(points) + "\n")
