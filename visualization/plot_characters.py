import matplotlib.pyplot as plt
import numpy as np

def plot_single_character(char_vector, label, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 4))
    grid = char_vector.reshape(7, 5)
    ax.imshow(grid, cmap='RdBu', vmin=-1, vmax=1)
    ax.set_title(label, fontsize=11)
    ax.set_xticks(range(5))
    ax.set_yticks(range(7))
    ax.grid(True, color='gray', linewidth=0.5)
    for i in range(7):
        for j in range(5):
            val = int(grid[i, j])
            color = 'white' if val == 1 else 'black'
            ax.text(j, i, str(val),
                    ha='center', va='center',
                    fontsize=7, color=color)
    return ax

def plot_all_characters(characters, labels, save_path=None):
    n = len(characters)
    cols = 7
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * 2, rows * 2.5))
    axes = axes.flatten()
    for i in range(n):
        plot_single_character(characters[i], labels[i], axes[i])
    for i in range(n, len(axes)):
        axes[i].axis('off')
    fig.suptitle('Dataset Huruf Arab — Grid 7x5',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig

def plot_prediction_result(input_vector, pred_idx, scores, labels, save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    plot_single_character(input_vector, 'Input', ax1)
    colors = ['#16A34A' if i == pred_idx else '#93C5FD'
              for i in range(len(labels))]
    ax2.barh(labels, scores * 100, color=colors)
    ax2.set_xlabel('Confidence (%)')
    ax2.set_title(f'Prediksi: {labels[pred_idx]}')
    ax2.set_xlim(0, 100)
    for i, score in enumerate(scores):
        ax2.text(score * 100 + 0.5, i,
                 f'{score*100:.1f}%',
                 va='center', fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig