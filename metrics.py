import torch


def binary_accuracy(y_logits, y_true):
    return ((y_logits > 0.0) == y_true).sum().item() / y_true.size(0)


def binary_precision(y_logits, y_true):
    true_positives = ((y_logits > 0.0) & (y_true > 0)).sum().item()
    predicted_positives = (y_logits > 0.0).sum().item()
    return true_positives / predicted_positives if predicted_positives != 0 else 0


def binary_recall(y_logits, y_true):
    true_positives = ((y_logits > 0.0) & (y_true > 0)).sum().item()
    actual_positives = y_true.sum().item()
    return true_positives / actual_positives if actual_positives != 0 else 0


def binary_f1_score(y_logits, y_true):
    prec = binary_precision(y_logits, y_true)
    rec = binary_recall(y_logits, y_true)
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) != 0 else 0


def multi_class_accuracy(y_logits, y_true):
    return (torch.argmax(y_logits, dim=1) == y_true).sum().item() / y_true.size(0)


def multi_class_precision(y_logits, y_true):
    y_pred = torch.argmax(y_logits, dim=1)
    true_positives = torch.sum((y_pred == y_true) & (y_true != 0)).item()
    predicted_positives = torch.sum(y_pred != 0).item()
    return true_positives / predicted_positives if predicted_positives != 0 else 0


def multi_class_recall(y_logits, y_true):
    y_pred = torch.argmax(y_logits, dim=1)
    true_positives = torch.sum((y_pred == y_true) & (y_true != 0)).item()
    actual_positives = torch.sum(y_true != 0).item()
    return true_positives / actual_positives if actual_positives != 0 else 0


def multi_class_f1_score(y_logits, y_true):
    prec = multi_class_precision(y_logits, y_true)
    rec = multi_class_recall(y_logits, y_true)
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) != 0 else 0


def mse(y_pred, y_true):
    return torch.mean((y_true - y_pred) ** 2).item()


def r2_score(y_pred, y_true):
    y_mean = torch.mean(y_true)
    total_sum_of_squares = torch.sum((y_true - y_mean) ** 2)
    residual_sum_of_squares = torch.sum((y_true - y_pred) ** 2)
    r2 = 1 - (residual_sum_of_squares / total_sum_of_squares)
    return r2.item()
