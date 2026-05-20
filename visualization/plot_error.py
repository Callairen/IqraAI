import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

def plot_error_per_epoch(history, save_path=None):
    errors = history['error_per_epoch']
    epochs = list(range(1, len(errors) + 1))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, errors,
            color='#2563EB',
            linewidth=2,
            marker='o',
            markersize=4,
            label='Jumlah error')
    ax.fill_between(epochs, errors, alpha=0.1, color='#2563EB')

    if errors[-1] == 0:
        conv_epoch = len(errors)
        ax.axvline(x=conv_epoch,
                   color='#16A34A',
                   linestyle='--',
                   linewidth=1.5,
                   label=f'Konvergen di epoch {conv_epoch}')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Jumlah Error', fontsize=12)
    ax.set_title('Konvergensi Training Perceptron\nPengenalan Huruf Arab', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, len(epochs))
    ax.set_ylim(bottom=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig