import numpy as np
import random
import torch


def generate_sine_wave(output_path, size):
	np.random.seed(2)

	label_array = [1, 2, 3, 4]

	dataset = []
	for i in range(size):
		label = random.choice(label_array)
		sine_wave = create_sine_wave(size, label)
		features = " ".join([str(x) for x in list(sine_wave)])
		dataset.append("\t".join([features, str(label)]))
	with open(output_path, "w") as f:
		for row in dataset:
			f.writelines(row + "\n")


def create_sine_wave(size, label):
	param = label
	x = np.linspace(-np.pi * np.random.rand(), np.pi * np.random.rand(), size)
	data = x / param
	return data
