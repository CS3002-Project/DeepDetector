from cnn import CNN, Config as CNNConfig
from lstm import LSTM, Config as LSTMConfig
from misc import generate_sine_waves
from preprocessing import preprocess_real_disp
from train import train
from utils import TimeSeries


def sine_wave_example():
	generate_sine_waves("data/sine_waves/train", size=1000, dim=5, num=30)
	generate_sine_waves("data/sine_waves/test", size=10, dim=5, num=30)
	train_time_series_data = TimeSeries("data/sine_waves/train")
	test_time_series_data = TimeSeries("data/sine_waves/test")
	# TimeSeries.plot(*train_time_series_data[2])
	print("-" * 5 + "CNN" + "-" * 5)
	train(train_dataset=train_time_series_data, test_dataset=test_time_series_data, model_cls=CNN, config=CNNConfig)
	print("-" * 5 + "LSTM" + "-" * 5)
	train(train_dataset=train_time_series_data, test_dataset=test_time_series_data, model_cls=LSTM, config=LSTMConfig)


def real_disp():
	preprocess_real_disp("data/real_disp_sample.txt", "data/real_disp_sample_out.txt")


if __name__ == "__main__":
	real_disp()
