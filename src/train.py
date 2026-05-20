import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.perceptron import Perceptron
from data.characters import ALL_CHARACTERS
from data.labels import LABELS, N_CLASSES, get_target


def prepare_data():
    X = np.array(ALL_CHARACTERS)
    T = np.array([get_target(i) for i in range(N_CLASSES)])
    return X, T


def train_model(alpha=1.0, theta=0.2, max_epoch=1000, verbose=True):
    X, T = prepare_data()

    model = Perceptron(
        n_input=X.shape[1],
        n_output=N_CLASSES,
        alpha=alpha,
        theta=theta
    )

    if verbose:
        print("Training Perceptron Huruf Arab")
        print("-" * 40)
        print(f"Jumlah data   : {X.shape[0]}")
        print(f"Jumlah input  : {X.shape[1]}")
        print(f"Jumlah kelas  : {N_CLASSES}")
        print(f"Alpha         : {alpha}")
        print(f"Theta         : {theta}")
        print("-" * 40)

    history = model.train(X, T, max_epoch=max_epoch, verbose=verbose)

    if verbose:
        print("\nHasil Prediksi:")
        print("-" * 40)

    correct = 0

    for i, x in enumerate(X):
        pred_idx = model.predict_class(x)
        actual_label = LABELS[i]
        pred_label = LABELS[pred_idx]

        if pred_idx == i:
            correct += 1
            status = "BENAR"
        else:
            status = "SALAH"

        if verbose:
            print(f"{status} | Aktual: {actual_label} | Prediksi: {pred_label}")

    accuracy = correct / len(X) * 100

    if verbose:
        print("-" * 40)
        print(f"Akurasi: {correct}/{len(X)} = {accuracy:.2f}%")

    return model, history


if __name__ == "__main__":
    train_model()