from train import train
from misc import generate_sine_wave
from cnn import CNN, Config as CNNConfig
from lstm import LSTM, Config as LSTMConfig

if __name__ == "__main__":
	generate_sine_wave("data/data.train", 1000)
	generate_sine_wave("data/data.test", 100)
	print("-" * 5 + "CNN" + "-" * 5)
	train(model_cls=CNN, config=CNNConfig)
	print("-" * 5 + "LSTM" + "-" * 5)
	train(model_cls=LSTM, config=LSTMConfig)
