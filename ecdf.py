import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from scipy.stats import percentileofscore

# For yolo_nano
# files = [('./analysis/dot_optimization/yolo_nano/without_gaussian_blur/log_file_2.txt', 2), 
#         ('./analysis/dot_optimization/yolo_nano/without_gaussian_blur/log_file_3.txt', 3),
#         ('./analysis/dot_optimization/yolo_nano/without_gaussian_blur/log_file_4.txt', 4),
#         ('./analysis/dot_optimization/yolo_nano/without_gaussian_blur/log_file_5.txt', 5)]

# For yolo_small
# files = [('./analysis/dot_optimization/yolo_small/log_file_2.txt', 2), 
#         ('./analysis/dot_optimization/yolo_small/log_file_3.txt', 3),
#         ('./analysis/dot_optimization/yolo_small/log_file_4.txt', 4),
#         ('./analysis/dot_optimization/yolo_small/log_file_5.txt', 5)]

# For yolo_medium
files = [('./analysis/dot_optimization/yolo_medium/log_file_2.txt', 2), 
        ('./analysis/dot_optimization/yolo_medium/log_file_3.txt', 3),
        ('./analysis/dot_optimization/yolo_medium/log_file_4.txt', 4),
        ('./analysis/dot_optimization/yolo_medium/log_file_5.txt', 5)]

# Custom ECDF function
def ecdf(data):
    """Compute ECDF for a one-dimensional array of data."""
    x = np.sort(data)
    y = np.arange(1, len(x)+1) / len(x)
    return x, y

# Create a figure with 4 subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Cumulative Distribution of Successful Attack Times by Number of Dots', 
             fontsize=16, fontweight='bold', y=1.02)

# Colors for percentiles
colors = ['#fdae61', '#d7191c', '#abdda4', '#ffffbf', '#756bb1']

# Process each file separately
for idx, (filename, num_dots) in enumerate(files):
    try:
        # Data collection for current file
        success_times = []
        
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
                    
        # Calculate ECDF
        x, y = ecdf(np.array(success_times))
        
        # Find time thresholds for different percentiles
        percentiles = [50, 75, 90, 95, 99]
        thresholds = {p: np.percentile(success_times, p) for p in percentiles}
        
        # Select subplot
        ax = axes[idx//2, idx%2]
        
        # Plot ECDF
        ax.step(x, y, where='post', linewidth=2.5, color='#2c7bb6', 
                label=f'{num_dots} Dots ECDF')
        
        # Add threshold annotations
        for (p, t), c in zip(thresholds.items(), colors):
            y_val = percentileofscore(success_times, t) / 100
            ax.vlines(t, 0, y_val, linestyles='dashed', colors=c, alpha=0.7)
            ax.hlines(y_val, 0, t, linestyles='dashed', colors=c, alpha=0.7)
            ax.scatter(t, y_val, color=c, s=80, zorder=10, 
                      label=f'{p}% ({t:.2f}s)')
        
        # Subplot styling
        ax.set_title(f'{num_dots} Dots Configuration', fontsize=14)
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Cumulative Probability', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(title='Percentiles Thresholds:', title_fontsize=10, 
                 loc='lower right', frameon=True)
        ax.fill_between(x, y, step='post', color='#2c7bb6', alpha=0.1)
        
        # Add statistics annotation
        ax.text(0.95, 0.05, 
                f"Successful Attacks: {len(success_times)}\nMedian Time: {np.median(success_times):.2f}s",
                transform=ax.transAxes, ha='right', va='bottom',
                bbox=dict(facecolor='white', alpha=0.8))
        
        # Print statistics to terminal
        print(f"\nStatistics for {num_dots} dots:")
        print(f"Total successful attacks: {len(success_times)}")
        print(f"Median time: {np.median(success_times):.2f}s")
        print("Time Thresholds for Percentiles:")
        for p, t in thresholds.items():
            print(f"{p}% of successful attacks complete within {t:.2f}s")
            
    except FileNotFoundError:
        print(f"Warning: File {filename} not found")
        continue

# Adjust layout and display
plt.tight_layout()
plt.show()