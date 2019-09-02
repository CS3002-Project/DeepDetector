from cnn.config import Config as CNNConfig
from cnn.model import *
import numpy as np
from prettytable import PrettyTable
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import write_csv


def run_epoch(model, train_iterator, val_iterator, epoch):
    train_losses = []
    val_accuracies = []
    losses = []

    # Reduce learning rate as number of epochs increase
    if (epoch == int(model.config.max_epochs / 3)) or (epoch == int(2 * model.config.max_epochs / 3)):
        model.reduce_lr()

    for i, batch in enumerate(train_iterator):
        model.optimizer.zero_grad()
        x_times, x_features, x_labels = batch
        if torch.cuda.is_available():
            x_times, x_features, x_labels = x_times.cuda(), x_features.cuda(), x_labels.cuda()
            y = (x_labels[:, -1]).type(torch.cuda.LongTensor)
        else:
            y = (x_labels[:, -1]).type(torch.LongTensor) 
        y_pred = model(x_times, x_features)
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
            val_accuracy = evaluate(model, val_iterator)
            model.train()

    return train_losses, val_accuracies


def train(train_dataset, test_dataset=None, val_dataset=None, model_cls=CNN, config=CNNConfig()):

    if not val_dataset:
        size = len(train_dataset)
        train_size = int(size * config.split_ratio)
        val_size = size - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    print("Loaded {} training examples".format(len(train_dataset)))
    if test_dataset is not None:
        print("Loaded {} test examples".format(len(test_dataset)))
    print("Loaded {} validation examples".format(len(val_dataset)))

    train_dataset_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataset_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)

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

    train_evaluation_output = "train_evaluation.csv"
    validate_evaluation_output = "validate_evaluation.csv"
    evaluate(model, train_dataset_loader, train_evaluation_output)
    evaluate(model, val_dataset_loader, validate_evaluation_output)

    print('Training evaluation saved to {}'.format(train_evaluation_output))
    print('Validation evaluation saved to {}'.format(validate_evaluation_output))
    if test_dataset is not None:
        test_dataset_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)
        test_evaluation_output = "test_evaluation.csv"
        evaluate(model, test_dataset_loader, test_evaluation_output)
        print('Test evaluation saved to {}'.format(test_evaluation_output))


def evaluate(model, iterator, output_result=None):
    y_preds = []
    y_truths = []
    for idx, batch in enumerate(iterator):
        x_times, x_features, x_labels = batch
        if torch.cuda.is_available():
            x_times, x_features, x_labels = x_times.cuda(), x_features.cuda(), x_labels.cuda()
        y_pred = model(x_times, x_features)
        predicted = torch.max(y_pred.cpu().data, 1)[1]
        y_preds.extend(predicted.numpy())
        y_truths.extend(x_labels[:, -1].cpu().data.numpy())
    evaluate_multi_class(y_preds, y_truths, output_result)


def evaluate_multi_class(y_preds, y_truths, output_result=None):
    pred_labels = set(y_preds)
    true_labels = set(y_truths)
    all_labels = pred_labels.union(true_labels)
    print(pred_labels, true_labels)
    label2idx, idx2label = {}, {}

    for i, label in enumerate(all_labels):
        label2idx[label] = i
        idx2label[i] = label

    preds = [label2idx[p] for p in y_preds]
    truths = [label2idx[t] for t in y_truths]

    accuracy = accuracy_score(truths, preds)
    individual_precision = precision_score(truths, preds, average=None)
    individual_recall = recall_score(truths, preds, average=None)
    individual_f1 = f1_score(truths, preds, average=None)
    micro_precision = precision_score(truths, preds, average="micro")
    micro_recall = recall_score(truths, preds, average="micro")
    micro_f1 = f1_score(truths, preds, average="micro")
    macro_precision = precision_score(truths, preds, average="macro")
    macro_recall = recall_score(truths, preds, average="macro")
    macro_f1 = f1_score(truths, preds, average="macro")
    result = {
        "accuracy": accuracy,
        "individual_precision": individual_precision,
        "individual_recall": individual_recall,
        "individual_f1": individual_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "label2idx": label2idx,
        "idx2label": idx2label
    }

    header = ["label", "accuracy", "precision", "recall", "f1"]
    csv_content = []
    table = PrettyTable(header)

    for label, idx in label2idx.items():
        row = [label, "", str(individual_precision[idx]), str(individual_recall[idx]),
               str(individual_f1[idx])]
        table.add_row(row)
        csv_content.append(row)
    macro_row = ["macro", accuracy, macro_precision, macro_recall, macro_f1]
    micro_row = ["micro", "", micro_precision, micro_recall, micro_f1]
    table.add_row(macro_row)
    table.add_row(micro_row)
    csv_content.append(macro_row)
    csv_content.append(micro_row)
    if output_result is not None:
        write_csv(csv_content, header, output_result)
    print(table)
