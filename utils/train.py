from cnn.config import Config as CNNConfig
from cnn.model import *
from collections import Counter
import numpy as np
import os
from prettytable import PrettyTable
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import write_csv
from torch.utils.tensorboard import SummaryWriter


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
            evaluate(model, val_iterator)
            model.train()

    return train_losses, val_accuracies


def unbalanced_ce_weights(num_labels, coeff):
    weight_0 = 1. / ((coeff * (num_labels-1))+1)
    if torch.cuda.is_available():
        return torch.cuda.FloatTensor([weight_0] + [weight_0 * coeff] * (num_labels-1))
    return torch.FloatTensor([weight_0] + [weight_0 * coeff] * (num_labels-1))


def write_results(eval_results, writer, name_space, epoch, loss=None):
    out_results = {
        "macro_f1": eval_results["macro_f1"],
        "macro_precision": eval_results["macro_precision"],
        "macro_recall": eval_results["macro_recall"],
        "accuracy": eval_results["accuracy"]
    }
    if loss is not None:
        out_results["loss"] = sum(loss)
    writer.add_scalars(name_space, out_results, epoch)


def train(train_dataset, eval_out_dir, test_dataset=None, val_dataset=None, model_cls=CNN, config=CNNConfig(),
          log_dir="exp_log"):
    exp_name = model_cls.name()
    evaluate_per_epoch = 2
    writer = SummaryWriter(os.path.join(log_dir, exp_name))

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
    NLLLoss = nn.CrossEntropyLoss(weight=unbalanced_ce_weights(config.output_size, 5))
    model.add_optimizer(optimizer)
    model.add_loss_op(NLLLoss)

    train_losses = []
    val_accuracies = []

    train_evaluation_output = os.path.join(eval_out_dir, "train_evaluation.csv")
    validate_evaluation_output = os.path.join(eval_out_dir, "validate_evaluation.csv")
    train_confusion_output = os.path.join(eval_out_dir, "train_confusion.csv")
    validate_confusion_output = os.path.join(eval_out_dir, "validate_confusion.csv")

    for epoch in range(config.max_epochs):
        print("Epoch: {}".format(epoch))
        train_loss, val_accuracy = run_epoch(model, train_dataset_loader, val_dataset_loader, epoch)
        train_losses.append(train_loss)
        val_accuracies.append(val_accuracy)
        train_results = evaluate(model, train_dataset_loader, train_evaluation_output, train_confusion_output)
        write_results(train_results, writer, "train", epoch, train_loss)

        if epoch % evaluate_per_epoch == 0:
            eval_results = evaluate(model, val_dataset_loader, validate_evaluation_output, validate_confusion_output)
            write_results(eval_results, writer, "eval", epoch)

    print('Training evaluation saved to {}'.format(train_evaluation_output))
    print('Validation evaluation saved to {}'.format(validate_evaluation_output))
    if test_dataset is not None:
        test_dataset_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)
        test_evaluation_output = os.path.join(eval_out_dir, "test_evaluation.csv")
        evaluate(model, test_dataset_loader, test_evaluation_output)
        print('Test evaluation saved to {}'.format(test_evaluation_output))


def evaluate(model, iterator, output_metrics_result=None, output_confusion_result=None):
    y_preds = []
    y_truths = []
    example_feature, example_pred = None, None
    for idx, batch in enumerate(iterator):
        x_times, x_features, x_labels = batch
        if torch.cuda.is_available():
            x_times, x_features, x_labels = x_times.cuda(), x_features.cuda(), x_labels.cuda()
        y_pred = model(x_times, x_features)
        predicted = torch.max(y_pred.cpu().data, 1)[1]
        if example_feature is None and example_pred is None:
            example_feature, example_pred = x_features, predicted
        y_preds.extend(predicted.numpy())
        y_truths.extend(x_labels[:, -1].cpu().data.numpy())
    print("Example-----")
    print("-----Feature: {}".format(example_feature))
    print("-----Prediction: {}".format(example_pred))
    return evaluate_multi_class(y_preds, y_truths, output_metrics_result, output_confusion_result)


def evaluate_multi_class(y_preds, y_truths, output_metrics_result=None, output_confusion_result=None):
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
    eval_confusion_matrix = confusion_matrix(y_truths, y_preds, labels=list(all_labels))
    results = {
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
        "idx2label": idx2label,
        "confusion_matrix": eval_confusion_matrix
    }

    label_count = Counter(y_truths)
    header = ["label", "accuracy", "precision", "recall", "f1", "count"]
    csv_metrics_content = []
    metrics_table = PrettyTable(header)

    for label, idx in label2idx.items():
        row = [label, "", str(individual_precision[idx]), str(individual_recall[idx]),
               str(individual_f1[idx]), label_count[label]]
        metrics_table.add_row(row)
        csv_metrics_content.append(row)
    macro_row = ["macro", accuracy, macro_precision, macro_recall, macro_f1, ""]
    micro_row = ["micro", "", micro_precision, micro_recall, micro_f1, ""]
    metrics_table.add_row(macro_row)
    metrics_table.add_row(micro_row)
    csv_metrics_content.append(macro_row)
    csv_metrics_content.append(micro_row)
    if output_metrics_result is not None:
        write_csv(csv_metrics_content, header, output_metrics_result)

    csv_confusion_content = []
    sorted_all_labels = sorted(list(all_labels))
    confusion_header = ["truth \\ prediction"] + sorted_all_labels
    confusion_table = PrettyTable(confusion_header)
    for i, class_row in enumerate(eval_confusion_matrix):
        row = [sorted_all_labels[i]]
        row.extend(class_row)
        confusion_table.add_row(row)
        csv_confusion_content.append(row)

    if output_confusion_result is not None:
        write_csv(csv_confusion_content, header, output_confusion_result)

    print("Metric table")
    print(metrics_table)
    print("Confusion matrix")
    print(confusion_table)

    return results
