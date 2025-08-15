import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Data
dots = np.array([2, 3, 4, 5])
wavelengths = [380, 480, 532, 580, 680]
accuracy = {
    380: [4.89, 5.17, 8.35, 9.01],
    480: [7.99, 10.81, 14.82, 17.79],
    532: [7.81, 11.06, 14.51, 16.92],
    580: [9.13, 12.86, 16.75, 20.07],
    680: [4.89, 6.37, 8.93, 11.67]
}

# Wavelength color mapping (approximate)
wavelength_colors = {
    380: '#6B5B95',  # Violet
    480: '#007ACC',  # Blue
    532: '#00CC66',  # Green
    580: '#FFAA33',  # Orange-Yellow
    680: '#CC3333'   # Red
}

# Different markers
markers = ['o', 's', 'D', '^', 'v']

# Plot setup
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")

# Plot each wavelength
for i, wl in enumerate(wavelengths):
    plt.plot(dots, accuracy[wl], 
             label=f"{wl} nm", 
             color=wavelength_colors[wl], 
             marker=markers[i], 
             markersize=8, 
             linewidth=2)

# Labels and aesthetics
plt.xlabel("Number of Dots", fontsize=14)
plt.ylabel("Accuracy (%)", fontsize=14)
plt.title("Accuracy vs. Number of Dots for Different Wavelengths", fontsize=16, fontweight='bold')
plt.xticks(dots)
plt.legend(title="Wavelength", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.7)

# Show plot
plt.show()
