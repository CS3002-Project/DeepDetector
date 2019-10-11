from torch import nn
import torch


class CNN(nn.Module):

    @classmethod
    def name(cls):
        return "cnn"

    def __init__(self, config):
        super(CNN, self).__init__()
        self.config = config
        self.optimizer = None
        self.loss_op = None

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=self.config.embed_size, out_channels=self.config.num_channels,
                      kernel_size=self.config.kernel_size[0]),
            nn.ReLU(),
            nn.MaxPool1d(self.config.max_sen_len - self.config.kernel_size[0] + 1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=self.config.embed_size, out_channels=self.config.num_channels,
                      kernel_size=self.config.kernel_size[1]),
            nn.ReLU(),
            nn.MaxPool1d(self.config.max_sen_len - self.config.kernel_size[1] + 1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=self.config.embed_size, out_channels=self.config.num_channels,
                      kernel_size=self.config.kernel_size[2]),
            nn.ReLU(),
            nn.MaxPool1d(self.config.max_sen_len - self.config.kernel_size[2] + 1)
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=self.config.embed_size, out_channels=self.config.num_channels,
                      kernel_size=self.config.kernel_size[3]),
            nn.ReLU(),
            nn.MaxPool1d(self.config.max_sen_len - self.config.kernel_size[3] + 1)
        )

        self.dropout = nn.Dropout(self.config.dropout_keep)

        # Fully-Connected Layer
        self.fc = nn.Linear(self.config.num_channels * len(self.config.kernel_size), self.config.output_size)

        # Softmax non-linearity
        self.softmax = nn.Softmax(dim=1)

    def forward(self, times, x_features):
        embedded_features = x_features.permute(0, 2, 1).float()
        read_out = self.forward_single(embedded_features)
        output = self.softmax(read_out)
        return output

    def forward_single(self, embedded_sent):
        conv_out1 = self.conv1(embedded_sent).squeeze(2)  # shape=(64, num_channels, 1) (squeeze 1)
        conv_out2 = self.conv2(embedded_sent).squeeze(2)
        conv_out3 = self.conv3(embedded_sent).squeeze(2)
        conv_out4 = self.conv4(embedded_sent).squeeze(2)

        all_out = torch.cat((conv_out1, conv_out2, conv_out3, conv_out4), 1)
        final_feature_map = self.dropout(all_out)
        return self.fc(final_feature_map)

    def add_optimizer(self, optimizer):
        self.optimizer = optimizer

    def add_loss_op(self, loss_op):
        self.loss_op = loss_op

    def reduce_lr(self):
        print("Reducing LR")
        for g in self.optimizer.param_groups:
            g['lr'] = g['lr'] / 2
