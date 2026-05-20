import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.characters import ALL_CHARACTERS
from data.labels import LABELS
from src.train import train_model
from visualization.plot_error import plot_error_per_epoch
from visualization.plot_characters import plot_all_characters, plot_prediction_result
from visualization.plot_weights import plot_weight_heatmap
import matplotlib.pyplot as plt

print("Memulai training...")
model, history = train_model(verbose=False)
print("Training selesai!")

print("Membuat visualisasi...")
plot_all_characters(ALL_CHARACTERS, LABELS,
                    'visualization/output_characters.png')
plot_error_per_epoch(history,
                     'visualization/output_error.png')
plot_weight_heatmap(model, LABELS,
                    'visualization/output_weights.png')

scores = model.confidence_scores(ALL_CHARACTERS[0])
plot_prediction_result(ALL_CHARACTERS[0], 0, scores, LABELS,
                       'visualization/output_prediction.png')

print("Semua visualisasi tersimpan di folder visualization/!")
plt.show()