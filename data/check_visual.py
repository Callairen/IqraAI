import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from characters import ALL_CHARACTERS
from labels import LABELS, N_CLASSES


def visualize_all():
    """Tampilkan semua 28 huruf Arab dalam grid visual."""
    cols = 7
    rows = (N_CLASSES + cols - 1) // cols 

    fig, axes = plt.subplots(rows, cols, figsize=(16, 10))
    fig.suptitle('Preview Dataset — 28 Huruf Arab (Grid 7×5 Bipolar)',
                 fontsize=14, fontweight='bold', y=1.01)
    axes = axes.flatten()

    for i in range(len(axes)):
        if i < N_CLASSES:
            grid = ALL_CHARACTERS[i].reshape(7, 5)
            axes[i].imshow(grid, cmap='binary', vmin=-1, vmax=1,
                           interpolation='nearest')
            axes[i].set_title(LABELS[i], fontsize=9, pad=4)
        axes[i].axis('off')

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), 'preview_huruf.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Preview tersimpan di: {output_path}")
    plt.show()


def check_uniqueness():
    """Periksa apakah ada huruf yang grid-nya identik (berbahaya untuk Perceptron)."""
    print("\n Mengecek keunikan setiap huruf...")
    duplicates_found = False

    for i in range(N_CLASSES):
        for j in range(i + 1, N_CLASSES):
            if np.array_equal(ALL_CHARACTERS[i], ALL_CHARACTERS[j]):
                print(f"DUPLIKAT: {LABELS[i]} dan {LABELS[j]} memiliki grid yang sama!")
                duplicates_found = True

    if not duplicates_found:
        print(" Semua 28 huruf memiliki representasi yang unik.")


def check_dimensions():
    """Pastikan semua vektor berdimensi 35."""
    print("\nMengecek dimensi vektor...")
    all_ok = True
    for i, char in enumerate(ALL_CHARACTERS):
        if char.shape != (35,):
            print(f"  {LABELS[i]}: dimensi salah! ({char.shape})")
            all_ok = False
    if all_ok:
        print(f"  Semua {N_CLASSES} huruf berdimensi (35,) — OK")


def print_summary():
    """Tampilkan ringkasan dataset."""
    print("\nRINGKASAN DATASET")
    print(f"  Total huruf   : {N_CLASSES}")
    print(f"  Dimensi input : 35 (grid 7×5)")
    print(f"  Format nilai  : bipolar (1 dan -1)")
    active_pixels = [int(np.sum(c == 1)) for c in ALL_CHARACTERS]
    print(f"  Pixel aktif   : min={min(active_pixels)}, "
          f"max={max(active_pixels)}, "
          f"rata²={sum(active_pixels)/len(active_pixels):.1f}")


if __name__ == '__main__':
    print("=" * 50)
    print("  CEK DATASET — 28 Huruf Arab")
    print("=" * 50)
    check_dimensions()
    check_uniqueness()
    print_summary()
    visualize_all()