class Config(object):
    embed_size = 1
    hidden_layers = 2
    hidden_size = 64
    bidirectional = True
    output_size = 4
    max_epochs = 30
    lr = 0.25
    batch_size = 64
    max_sen_len = 10  # Sequence length for RNN
    dropout_keep = 0.8
    split_ratio = 0.8
