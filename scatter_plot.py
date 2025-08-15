import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import re

# For yolo_nano
files = [('./analysis/dot_optimization/yolo_nano/without_gaussian_blur/log_file_2.txt', 2), 
        ('./analysis/dot_optimization/yolo_nano/without_gaussian_blur/log_file_3.txt', 3),
        ('./analysis/dot_optimization/yolo_nano/without_gaussian_blur/log_file_4.txt', 4),
        ('./analysis/dot_optimization/yolo_nano/without_gaussian_blur/log_file_5.txt', 5)]

# For yolo_small
# files = [('./analysis/dot_optimization/yolo_small/log_file_2.txt', 2), 
#         ('./analysis/dot_optimization/yolo_small/log_file_3.txt', 3),
#         ('./analysis/dot_optimization/yolo_small/log_file_4.txt', 4),
#         ('./analysis/dot_optimization/yolo_small/log_file_5.txt', 5)]

# For yolo_medium
# files = [('./analysis/dot_optimization/yolo_medium/log_file_2.txt', 2), 
#         ('./analysis/dot_optimization/yolo_medium/log_file_3.txt', 3),
#         ('./analysis/dot_optimization/yolo_medium/log_file_4.txt', 4),
#         ('./analysis/dot_optimization/yolo_medium/log_file_5.txt', 5)]

# Data collection
data = []
for filename, num_dots in files:
    try:
        with open(filename, 'r') as f:
            content = f.read()
            
        # Split into individual entries
        entries = re.split(r'\n(?=Processed)', content.strip())
        
        for entry in entries:
            time_match = re.search(r'Time taken: ([\d.]+)s', entry)
            score_match = re.search(r'Best score: (\d+)', entry)
            
            if time_match and score_match:
                time = float(time_match.group(1))
                score = int(score_match.group(1))
                
                if score == 0:
                    data.append((num_dots, time))
                    
    except FileNotFoundError:
        print(f"Warning: File {filename} not found")
        continue


'''
    Scatter plot
'''
# Prepare plot data with jitter
x_values = [d[0] + np.random.uniform(-0.1, 0.1) for d in data]
y_values = [d[1] for d in data]
colors = [d[0] for d in data]

# Create plot
plt.figure(figsize=(10, 6))
scatter = plt.scatter(x_values, y_values, c=colors, cmap='viridis', 
                    alpha=0.7, edgecolor='black', linewidth=0.5)

# Formatting
plt.xticks([2, 3, 4, 5], labels=['2 dots', '3 dots', '4 dots', '5 dots'])
plt.xlabel('Number of Adversarial Dots')
plt.ylabel('Time Taken (seconds)')
plt.title('Successful Attack Times (Score=0) by Number of Dots')
plt.grid(True, linestyle='--', alpha=0.7)
cbar = plt.colorbar(scatter)
cbar.set_label('Number of Dots', rotation=270, labelpad=15)

plt.tight_layout()
plt.show()


'''
    Box plot
'''

# # Create DataFrame from previous data collection code
# df = pd.DataFrame(data, columns=['Dots', 'Time'])

# plt.figure(figsize=(10, 6))
# sns.boxplot(x='Dots', y='Time', data=df, width=0.4, palette='Pastel1')
# sns.swarmplot(x='Dots', y='Time', data=df, color='black', size=5, alpha=0.7)
# plt.title('Time Distribution by Number of Dots (Box + Swarm)')
# plt.xlabel('Number of Adversarial Dots')
# plt.ylabel('Time Taken (seconds)')
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.show()



'''
    Violin plot
'''

# # Create DataFrame from previous data collection code
# df = pd.DataFrame(data, columns=['Dots', 'Time'])

# plt.figure(figsize=(10, 6))
# sns.violinplot(x='Dots', y='Time', data=df, inner='quartile', palette='Set3')
# plt.title('Time Density Distribution by Number of Dots')
# plt.xlabel('Number of Adversarial Dots')
# plt.ylabel('Time Taken (seconds)')
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.show()


'''
    ECDF plot
'''

# # Create DataFrame from previous data collection code
# df = pd.DataFrame(data, columns=['Dots', 'Time'])

# plt.figure(figsize=(10, 6))
# for dots in sorted(df['Dots'].unique()):
#     subset = df[df['Dots'] == dots]['Time']
#     x = np.sort(subset)
#     y = np.arange(1, len(x)+1)/len(x)
#     plt.plot(x, y, marker='.', linestyle='--', label=f'{dots} dots')

# plt.title('ECDF of Attack Times')
# plt.xlabel('Time Taken (seconds)')
# plt.ylabel('Cumulative Probability')
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.show()

'''
    Faceted histogram
''' 

# # Create DataFrame from previous data collection code
# df = pd.DataFrame(data, columns=['Dots', 'Time'])

# g = sns.FacetGrid(df, col='Dots', col_wrap=2, height=4, sharey=False)
# g.map(sns.histplot, 'Time', kde=True, bins=15, color='teal')
# g.set_titles("Dots: {col_name}")
# g.fig.suptitle('Time Distribution by Number of Dots', y=1.05)
# plt.show()