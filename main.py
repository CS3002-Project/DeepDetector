from cnn import CNN, Config as CNNConfig
from lstm import LSTM, Config as LSTMConfig
from misc import generate_sine_waves
from utils import TimeSeries, train
from preprocessing import preprocess_real_disp


def sine_wave_example():
    generate_sine_waves("data/sine_waves/train", size=1000, dim=5, num=30)
    generate_sine_waves("data/sine_waves/test", size=10, dim=5, num=30)
    train_time_series_data = TimeSeries("data/sine_waves/train", max_ts_size=30)
    test_time_series_data = TimeSeries("data/sine_waves/test", max_ts_size=30)
    # TimeSeries.plot(*train_time_series_data[2])
    print("-" * 5 + "CNN" + "-" * 5)
    train(train_dataset=train_time_series_data, test_dataset=test_time_series_data, model_cls=CNN, config=CNNConfig)
    print("-" * 5 + "LSTM" + "-" * 5)
    train(train_dataset=train_time_series_data, test_dataset=test_time_series_data, model_cls=LSTM, config=LSTMConfig)


def real_disp():
    # preprocess_real_disp("data/real_disp/subject3_ideal.log", "data/real_disp/processed", 200,
    #                      min_warm_up_sec=3., max_window_sec=5.)
    train_real_disp_data = TimeSeries("data/real_disp/processed", max_ts_size=700)
    train(train_dataset=train_real_disp_data, eval_out_dir="training/real_disp/cnn", model_cls=CNN, config=CNNConfig)
    train(train_dataset=train_real_disp_data, eval_out_dir="training/real_disp/lstm", model_cls=LSTM, config=LSTMConfig)


if __name__ == "__main__":
    # sine_wave_example()
    real_disp()
