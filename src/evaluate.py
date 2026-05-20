import numpy as np
from data.labels import LABELS


def accuracy(model, X):
    correct = 0

    for i, x in enumerate(X):
        pred_idx = model.predict_class(x)
        if pred_idx == i:
            correct += 1

    return correct / len(X)


def confusion_matrix_simple(model, X, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)

    for i, x in enumerate(X):
        pred_idx = model.predict_class(x)
        cm[i][pred_idx] += 1

    return cm


def print_report(model, X):
    print("\nLaporan Klasifikasi")
    print("-" * 50)

    for i, x in enumerate(X):
        pred_idx = model.predict_class(x)
        scores = model.confidence_scores(x)
        confidence = scores[pred_idx] * 100

        status = "BENAR" if pred_idx == i else "SALAH"

        print(
            f"{status} | {LABELS[i]} -> {LABELS[pred_idx]} "
            f"({confidence:.2f}%)"
        )