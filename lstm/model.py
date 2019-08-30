import torch
from torch import nn


class LSTM(nn.Module):
    def __init__(self, config):
        super(LSTM, self).__init__()
        self.config = config
        self.optimizer = None
        self.loss_op = None

        self.lstm = nn.LSTM(input_size=self.config.embed_size,
                            hidden_size=self.config.hidden_size,
                            num_layers=self.config.hidden_layers,
                            dropout=self.config.dropout_keep,
                            bidirectional=self.config.bidirectional)

        self.dropout = nn.Dropout(self.config.dropout_keep)

        # Fully-Connected Layer
        self.fc = nn.Linear(
            self.config.hidden_size * self.config.hidden_layers * (1 + self.config.bidirectional),
            self.config.output_size
        )

        # Softmax non-linearity
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.unsqueeze(2).permute(1, 0, 2).float()
        lstm_out, (h_n, c_n) = self.lstm(x)
        final_feature_map = self.dropout(h_n)  # shape=(num_layers * num_directions, 64, hidden_size)

        final_feature_map = torch.cat([final_feature_map[i, :, :] for i in range(final_feature_map.shape[0])], dim=1)
        final_out = self.fc(final_feature_map)
        return self.softmax(final_out)

    def add_optimizer(self, optimizer):
        self.optimizer = optimizer

    def add_loss_op(self, loss_op):
        self.loss_op = loss_op

    def reduce_lr(self):
        print("Reducing LR")
        for g in self.optimizer.param_groups:
            g['lr'] = g['lr'] / 2
