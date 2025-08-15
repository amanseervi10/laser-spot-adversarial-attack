import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re

# --- Data Processing ---
def process_file(filename):
    success_times = []
    try:
        with open(filename, 'r') as f:
            content = f.read()
        entries = re.split(r'\n(?=Processed)', content.strip())
        for entry in entries:
            time_match = re.search(r'Time taken: ([\d.]+)s', entry)
            score_match = re.search(r'Best score: (\d+)', entry)
            if time_match and score_match:
                time = float(time_match.group(1))
                score = int(score_match.group(1))
                if score == 0:
                    success_times.append(time)
    except FileNotFoundError:
        print(f"Warning: File {filename} not found")
    return np.array(success_times)

# --- Model and Dot Parameters ---
files_yolo_nano = [
    ('./analysis/dot_optimization/yolo_nano/without_gaussian_blur/log_file_2.txt', 2),
    ('./analysis/dot_optimization/yolo_nano/without_gaussian_blur/log_file_3.txt', 3),
    ('./analysis/dot_optimization/yolo_nano/without_gaussian_blur/log_file_4.txt', 4),
    ('./analysis/dot_optimization/yolo_nano/without_gaussian_blur/log_file_5.txt', 5)
]

files_yolo_small = [
    ('./analysis/dot_optimization/yolo_small/log_file_2.txt', 2),
    ('./analysis/dot_optimization/yolo_small/log_file_3.txt', 3),
    ('./analysis/dot_optimization/yolo_small/log_file_4.txt', 4),
    ('./analysis/dot_optimization/yolo_small/log_file_5.txt', 5)
]

# --- Combine Data ---
def gather_data(files, model_size):
    data = []
    for file, dots in files:
        success_times = process_file(file)
        for time in success_times:
            data.append([model_size, dots, time])
    return data

# Collect data for both models
yolo_nano_data = gather_data(files_yolo_nano, 'YOLO-Nano')
yolo_small_data = gather_data(files_yolo_small, 'YOLO-Small')

# --- Prepare DataFrame ---
df = pd.DataFrame(yolo_nano_data + yolo_small_data, columns=["Model", "Dots", "Time"])

# --- Prepare Quantiles ---
percentiles = [50, 75, 90, 95]
quantile_data = []
for model in df["Model"].unique():
    for dots in df["Dots"].unique():
        times = df[(df["Model"] == model) & (df["Dots"] == dots)]["Time"]
        quantiles = np.percentile(times, percentiles)
        quantile_data.append([model, dots] + list(quantiles))

# --- Create DataFrame for Heatmap ---
quantile_df = pd.DataFrame(quantile_data, columns=["Model", "Dots", "50%", "75%", "90%", "95%"])

# --- Pivot the data for heatmap plotting ---
quantile_pivot_yolo_nano = quantile_df[quantile_df["Model"] == "YOLO-Nano"].set_index("Dots").iloc[:, 1:].T
quantile_pivot_yolo_small = quantile_df[quantile_df["Model"] == "YOLO-Small"].set_index("Dots").iloc[:, 1:].T

# --- Plot 1: YOLO-Nano ---
plt.figure(figsize=(10, 8))
ax=sns.heatmap(
    quantile_pivot_yolo_nano,
    annot=True,
    annot_kws={"size": 18},  # Bigger numbers inside cells
    cmap="YlGnBu",
    cbar_kws={'label': 'Time (s)'}
)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=16)  # Tick labels
cbar.set_label('Time (s)', size=18)  # Axis label
plt.ylabel('Percentiles', fontsize=18)
plt.xlabel('Number of Spots', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)            
plt.tight_layout()
plt.savefig("new_graph1.png", dpi=300, bbox_inches='tight')
plt.show()

# --- Plot 2: YOLO-Small ---
plt.figure(figsize=(10, 8))
ax = sns.heatmap(
    quantile_pivot_yolo_small,
    annot=True,
    annot_kws={"size": 18},
    cmap="YlGnBu",
    cbar_kws={'label': 'Time (s)'}
)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=16)  # Tick labels
cbar.set_label('Time (s)', size=18)  # Axis label
plt.ylabel('Percentiles', fontsize=18)
plt.xlabel('Number of Spots', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig("new_graph2.png", dpi=300, bbox_inches='tight')
plt.show()


