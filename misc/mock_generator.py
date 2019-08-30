import numpy as np
import random
import torch


def generate_sine_wave(output_path, size):
	np.random.seed(2)

	label_array = [0, 1, 2, 3]
	N = 20
	
	dataset = []
	for i in range(size):
		label = random.choice(label_array)
		sine_wave = create_sine_wave(N, label)
		features = " ".join([str(sine_wave) for x in list(sine_wave)])
		dataset.append("\t".join([features, str(label)]))
	with open(output_path, "w") as f:
		for row in dataset:
			f.writelines(row + "\n")

def create_sine_wave(N, label):
	T = label
	x = np.linspace(-np.pi * np.random.rand(), np.pi * np.random.rand(), N)
	data = x / (T + 1)
	return data