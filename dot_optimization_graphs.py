import os
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure the output directory exists
output_dir = "graphs"
os.makedirs(output_dir, exist_ok=True)

# Parsing function
def parse_stats(file_path):
    total_time = 0
    count = 0
    zero_score_times = []
    non_zero_scores = []
    non_zero_times = []
    
    with open(file_path, 'r') as file:
        content = file.readlines()

    for i in range(len(content)):
        if content[i].startswith("Processed bbox"):
            if i + 2 >= len(content):  # Ensure we have enough lines to read
                continue
            
            score_line = content[i + 1].strip()
            time_line = content[i + 2].strip()

            if "Best score:" in score_line and "Time taken:" in time_line:
                score = float(score_line.split(":")[-1])
                time = float(time_line.split(":")[-1].split()[0])

                total_time += time
                count += 1

                if score == 0:
                    zero_score_times.append(time)
                else:
                    non_zero_scores.append(score)
                    non_zero_times.append(time)

    return zero_score_times, non_zero_scores, non_zero_times

# Load data
file_path = "stats.txt"
zero_score_times, non_zero_scores, non_zero_times = parse_stats(file_path)

# --- 1. Histogram of Processing Times ---
plt.figure(figsize=(8,5))
sns.histplot(zero_score_times + non_zero_times, bins=30, kde=True, color='blue')
plt.xlabel("Processing Time (seconds)")
plt.ylabel("Frequency")
plt.title("Histogram of Processing Times")
plt.savefig(os.path.join(output_dir, "hist_processing_time.png"))
plt.close()

# --- 2. Histogram of Success vs. Failure Counts ---
plt.figure(figsize=(6,5))
sns.barplot(x=["Success (Score = 0)", "Failure (Score > 0)"], y=[len(zero_score_times), len(non_zero_scores)], palette="coolwarm")
plt.ylabel("Count")
plt.title("Comparison of Successful vs Unsuccessful Cases")
plt.savefig(os.path.join(output_dir, "bar_success_vs_failure.png"))
plt.close()

# --- 3. Box Plot of Time Taken (Success vs Failure) ---
plt.figure(figsize=(6,5))
sns.boxplot(data=[zero_score_times, non_zero_times], palette="coolwarm")
plt.xticks([0, 1], ["Success (Score = 0)", "Failure (Score > 0)"])
plt.ylabel("Time Taken (seconds)")
plt.title("Box Plot of Processing Times")
plt.savefig(os.path.join(output_dir, "boxplot_time_success_failure.png"))
plt.close()

# --- 4. Scatter Plot of Confidence Score vs Time Taken ---
plt.figure(figsize=(8,5))
plt.scatter(non_zero_times, non_zero_scores, color='red', alpha=0.5)
plt.xlabel("Time Taken (seconds)")
plt.ylabel("Confidence Score")
plt.title("Confidence Score vs Time Taken")
plt.grid()
plt.savefig(os.path.join(output_dir, "scatter_confidence_vs_time.png"))
plt.close()

# --- 5. Histogram of Success Case Processing Times ---
plt.figure(figsize=(8,5))
sns.histplot(zero_score_times, bins=30, kde=True, color='green')
plt.xlabel("Processing Time (seconds)")
plt.ylabel("Frequency")
plt.title("Histogram of Successful Attack Times")
plt.savefig(os.path.join(output_dir, "hist_success_time.png"))
plt.close()

print(f"Graphs saved in '{output_dir}' directory.")
