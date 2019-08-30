from cnn import train_cnn
from misc import generate_sine_wave

if __name__ == "__main__":
	generate_sine_wave("data/data.train", 50000)
	generate_sine_wave("data/data.test", 1000)
	train_cnn()
