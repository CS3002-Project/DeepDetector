import numpy as np
import os
import pickle
from scipy.stats import skew
import matplotlib.pyplot as plt
from utils import TimeSeries


def visualize_example(input_path, feature_nums, windows):
    times, features, labels = load_all_features(input_path)
    selected_features = []
    for feature in features:
        selected_feature = []
        for n in feature_nums:
            selected_feature.append(feature[n])
        selected_features.append(selected_feature)
    TimeSeries.plot(times[:windows], np.array(selected_features[:windows]), labels[:windows])


def load_all_features(input_path):
    times, features, labels = [], [], []
    with open(input_path, "r") as f:
        for line in f.readlines():
            tokens = line.split("\t")
            times.append(tokens[0])
            features.append(np.fromstring(tokens[1], sep=" "))
            labels.append(tokens[2].strip())
    return times, features, labels


def extract_mean(channel_features):
    mean = np.mean(channel_features, axis=0)
    return mean


def extract_variance(channel_features):
    var = np.var(channel_features, axis=0)
    return var


def extract_poly_fit(channel_features):
    poly_coeff = np.polyfit(range(len(channel_features)), channel_features, 1)
    return poly_coeff.flatten()


def extract_skewness(channel_features):
    skewness = skew(channel_features, axis=0)
    return skewness


def extract_average_amplitude_change(channel_features):
    amplitude_changes = []
    for i in range(0, len(channel_features)-1):
        amplitude_changes.append(np.abs(channel_features[i+1]-channel_features[i]))
    return np.mean(amplitude_changes, axis=0)


def _extract_features(times, features, labels, channel_size, padding):
    extracted_times, extracted_features, extracted_labels = [], [], []
    index = 0
    while (index + channel_size) < len(features):
        channel_features = features[index: (index+channel_size)]
        mean = extract_mean(channel_features)
        variance = extract_variance(channel_features)
        poly_fit = extract_poly_fit(channel_features)
        skewness = extract_skewness(channel_features)
        average_amplitude_change = extract_average_amplitude_change(channel_features)
        feature_vector = np.concatenate([mean, variance, poly_fit, skewness, average_amplitude_change])
        extracted_times.append(times[index + channel_size])
        extracted_features.append(feature_vector)
        extracted_labels.append(labels[index + channel_size])
        index += padding
    return extracted_times, extracted_features, extracted_labels


def extract_real_disp_features(input_dir, output_dir, channel_size=5, padding=2):
    for file_name in os.listdir(input_dir):
        input_path = os.path.join(input_dir, file_name)
        times, features, labels = load_all_features(input_path)
        extracted_times, extracted_features, extracted_labels = _extract_features(times, features,
                                                                                  labels, channel_size, padding)
        output_path = os.path.join(output_dir, file_name)
        with open(output_path, "w") as f:
            for i in range(len(extracted_features)):
                f.writelines("\t".join([extracted_times[i], " ".join([str(v) for v in extracted_features[i]]),
                                        extracted_labels[i]]) + "\n")


def extract_label_stats(label_dict):
    normalized_label_dict = {}
    for label, samples in label_dict.items():
        print(label)
        all_samples = samples[0] + samples[1]
        amplitude = np.abs(all_samples)
        max_amplitude = np.max(amplitude, axis=0)
        normalized_label_dict[label] = {}
        normalized_negative_val = np.divide(samples[0], max_amplitude)
        mean_negative_val = extract_mean(normalized_negative_val)
        variance_negative_val = extract_variance(normalized_negative_val)
        normalized_label_dict[label][0] = [mean_negative_val, variance_negative_val]
        normalized_positive_val = np.divide(samples[1], max_amplitude)
        mean_positive_val = extract_mean(normalized_positive_val)
        variance_positive_val = extract_variance(normalized_positive_val)
        normalized_label_dict[label][1] = [mean_positive_val, variance_positive_val]
    return normalized_label_dict


def feature_analysis(input_dir, output_dir):
    label_dict = {}
    for file_name in os.listdir(input_dir):
        print(file_name)
        input_path = os.path.join(input_dir, file_name)
        times, features, labels = load_all_features(input_path)
        if len(labels) == 0:
            continue
        label = labels[-1]  # get the label of the last time stamp
        if label not in label_dict:
            label_dict[label] = {0: [], 1: []}
        for _label, values in label_dict.items():
            if _label == label:
                label_dict[_label][1] += features
            else:
                label_dict[_label][0] += features
    label_stats = extract_label_stats(label_dict)
    with open(os.path.join(output_dir, "label_stats.pickle"), "wb") as f:
        pickle.dump(label_stats, f)


def plot_feature_importance(output_dir, label_stats):
    for label, values in label_stats.items():
        print(label)
        mean_positive_val, variance_positive_val = values[1][0], values[1][1]
        mean_negative_val, variance_negative_val = values[0][0], values[0][1]
        mean_margin, variance_margin = np.abs(np.subtract(mean_positive_val, mean_negative_val)), \
            np.abs(np.add(variance_positive_val, variance_negative_val))

        mean_margin_ranking = list(np.array(mean_margin).argsort()[::-1].flatten())
        variance_margin_ranking = list(np.array(variance_margin).argsort().flatten())

        final_ranking = np.zeros(len(mean_margin))
        for feature_idx in range(len(mean_margin)):
            mean_margin_rank = mean_margin_ranking.index(feature_idx)
            variance_margin_rank = variance_margin_ranking.index(feature_idx)
            final_ranking[feature_idx] = mean_margin_rank + variance_margin_rank
        final_ranking = final_ranking.argsort()
        print("Top 10 most important features for label {} are {}".format(label, final_ranking[:10]))

        plt.rcdefaults()
        feature_num = np.arange(len(mean_positive_val))

        plt.bar(feature_num, mean_margin, align='center', alpha=0.5)
        plt.ylabel('Mean margin')
        plt.xlabel('Feature number')
        plt.title('Mean margin of label {}'.format(label))
        plt.savefig(os.path.join(output_dir, "mean_margin_{}.png".format(label)))

        plt.clf()
        plt.cla()
        plt.close()

        plt.rcdefaults()
        feature_num = np.arange(len(mean_positive_val))

        plt.bar(feature_num, variance_margin, align='center', alpha=0.5)
        plt.ylabel('Variance margin')
        plt.xlabel('Feature number')
        plt.title('Variance margin of label {}'.format(label))
        plt.savefig(os.path.join(output_dir, "variance_margin_{}.png".format(label)))

        plt.clf()
        plt.cla()
        plt.close()


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
