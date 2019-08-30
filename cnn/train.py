from cnn.config import Config
from cnn.model import *
from cnn.utils import *
import torch
import torch.optim as optim
from torch.utils.data import DataLoader


def train_cnn(train_file="data/data.train", test_file="data/data.test"):
    config = Config()

    train_dataset = TimeSeries(train_file)
    test_dataset = TimeSeries(test_file)
    train_dataset_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_dataset_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)

    model = CNN(config)
    if torch.cuda.is_available():
        model.cuda()
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=config.lr)
    NLLLoss = nn.NLLLoss()
    model.add_optimizer(optimizer)
    model.add_loss_op(NLLLoss)
    ##############################################################

    train_losses = []
    val_accuracies = []

    for i in range(config.max_epochs):
        print("Epoch: {}".format(i))
        train_loss, val_accuracy = model.run_epoch(iter(train_dataset_loader), iter(test_dataset_loader), i)
        train_losses.append(train_loss)
        val_accuracies.append(val_accuracy)

    train_acc = evaluate_model(model, iter(train_dataset_loader))
    val_acc = evaluate_model(model, iter(test_dataset_loader))
    test_acc = evaluate_model(model, iter(test_dataset_loader))

    print('Final Training Accuracy: {:.4f}'.format(train_acc))
    print('Final Validation Accuracy: {:.4f}'.format(val_acc))
    print('Final Test Accuracy: {:.4f}'.format(test_acc))