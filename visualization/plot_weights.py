import matplotlib.pyplot as plt
import numpy as np

def plot_weight_heatmap(model, labels, save_path=None):
    n_classes = model.n_output
    cols = 7
    rows = (n_classes + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * 2, rows * 2.5))
    axes = axes.flatten()
    for j in range(n_classes):
        w = model.W[:, j].reshape(7, 5)
        axes[j].imshow(w, cmap='RdBu',
                       vmin=-np.abs(w).max(),
                       vmax=np.abs(w).max())
        axes[j].set_title(labels[j], fontsize=9)
        axes[j].axis('off')
    for i in range(n_classes, len(axes)):
        axes[i].axis('off')
    fig.suptitle('Bobot Akhir per Neuron Output',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig