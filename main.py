from cnn import CNN, Config as CNNConfig
from lstm import LSTM, Config as LSTMConfig
from misc import generate_sine_waves
from utils import TimeSeries, train
from preprocessing import preprocess_real_disp, expand_real_disp_features, feature_analysis, \
    plot_feature_importance, visualize_example, extract_real_disp_features
import pickle


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
    # preprocess_real_disp("data/real_disp/subject3_ideal.log", "data/real_disp/extracted", 200,
    #                      min_warm_up_sec=3., max_window_sec=5.)
    train_real_disp_data = TimeSeries("data/real_disp/processed", max_ts_size=700)
    # train(train_dataset=train_real_disp_data, eval_out_dir="results/real_disp/cnn", model_cls=CNN, config=CNNConfig)
    train(train_dataset=train_real_disp_data, eval_out_dir="results/real_disp/lstm", model_cls=LSTM, config=LSTMConfig)


def analyse_real_disp():
    expand_real_disp_features(
        input_dir="data/real_disp/processed",
        output_dir="data/real_disp/expanded",
        channel_size=5,
        padding=2
    )
    feature_analysis(input_dir="data/real_disp/expanded", output_dir="data/real_disp/analysis")
    # with open("data/real_disp/analysis/label_stats.pickle", "rb") as f:
    #     label_stats = pickle.load(f)
    # plot_feature_importance(output_dir="data/real_disp/analysis", label_stats=label_stats)
    visualize_example("data/real_disp/extracted/2.txt", feature_nums=[0, 1, 2], windows=5)
    important_features = [192, 135, 456, 585, 586, 203, 588, 590, 399, 146, 602, 219, 667, 668, 670,
                          669, 220, 673, 26,
                          288, 612, 165, 614, 679, 680, 683, 236, 429, 48, 179, 181, 120, 377, 638, 639]
    #
    # extract_real_disp_features(input_dir="data/real_disp/expanded", output_dir="data/real_disp/extracted",
    #                            feature_nums=important_features)


if __name__ == "__main__":
    # sine_wave_example()
    # analyse_real_disp()
    real_disp()

