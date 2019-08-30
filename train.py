from cnn.config import Config as CNNConfig
from cnn.model import *
from utils.utils import *
import torch
import torch.optim as optim
from torch.utils.data import DataLoader


def run_epoch(model, train_iterator, val_iterator, epoch):
    train_losses = []
    val_accuracies = []
    losses = []

    # Reduce learning rate as number of epochs increase
    if (epoch == int(model.config.max_epochs / 3)) or (epoch == int(2 * model.config.max_epochs / 3)):
        model.reduce_lr()

    for i, batch in enumerate(train_iterator):
        model.optimizer.zero_grad()
        if torch.cuda.is_available():
            x_features, x_labels = batch
            x_features, x_labels = x_features.cuda(), x_labels.cuda()
            y = (x_labels - 1).type(torch.cuda.LongTensor)
        else:
            x_features, x_labels = batch
            y = (x_labels - 1).type(torch.LongTensor)
        y_pred = model(x_features)
        loss = model.loss_op(y_pred, y)
        loss.backward()
        losses.append(loss.data.cpu().numpy())
        model.optimizer.step()

        if i % 100 == 0:
            print("Iter: {}".format(i + 1))
            avg_train_loss = np.mean(losses)
            train_losses.append(avg_train_loss)
            print("\tAverage training loss: {:.5f}".format(avg_train_loss))
            losses = []

            # Evalute Accuracy on validation set
            val_accuracy = evaluate_model(model, val_iterator)
            print("\tVal Accuracy: {:.4f}".format(val_accuracy))
            model.train()

    return train_losses, val_accuracies


def train(train_file="data/data.train", test_file="data/data.test", val_file=None, model_cls=CNN, config=CNNConfig()):

    train_dataset = TimeSeries(train_file)
    test_dataset = TimeSeries(test_file)

    if val_file:
        val_dataset = TimeSeries(val_file)
    else:
        size = len(train_dataset)
        train_size = int(size * config.split_ratio)
        val_size = size - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    print("Loaded {} training examples".format(len(train_dataset)))
    print("Loaded {} test examples".format(len(test_dataset)))
    print("Loaded {} validation examples".format(len(val_dataset)))

    train_dataset_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataset_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)
    test_dataset_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)

    model = model_cls(config)
    if torch.cuda.is_available():
        model.cuda()
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=config.lr)
    NLLLoss = nn.NLLLoss()
    model.add_optimizer(optimizer)
    model.add_loss_op(NLLLoss)

    train_losses = []
    val_accuracies = []

    for i in range(config.max_epochs):
        print("Epoch: {}".format(i))
        train_loss, val_accuracy = run_epoch(model, train_dataset_loader, val_dataset_loader, i)
        train_losses.append(train_loss)
        val_accuracies.append(val_accuracy)

    train_acc = evaluate_model(model, train_dataset_loader)
    val_acc = evaluate_model(model, val_dataset_loader)
    test_acc = evaluate_model(model, test_dataset_loader)

    print('Final Training Accuracy: {:.4f}'.format(train_acc))
    print('Final Validation Accuracy: {:.4f}'.format(val_acc))
    print('Final Test Accuracy: {:.4f}'.format(test_acc))
