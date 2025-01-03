import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Data for the heatmap
models = ['Nano', 'Small', 'Medium', 'Large', 'Extreme']
processing_stages = ['Preprocessing', 'Inference', 'Postprocessing']
times = np.array([
    [2.5, 17.2, 2.1],   # Nano
    [2.3, 21.5, 2.2],   # Small
    [2.2, 27.2, 2.2],  # Medium
    [2.2, 34.8, 2.1],  # Large
    [2.2, 73.7, 2.1]  # Extreme
])

# Transpose the data to match the desired layout
times_transposed = times.T

# Create the heatmap
fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(
    times_transposed,
    annot=True,                     # Annotate cells with values
    fmt=".2f",                      # Format numbers to 2 decimal places
    cmap="YlGnBu",                  # Color map
    cbar_kws={'label': 'Time (ms)'},  # Color bar label
    xticklabels=models,             # Model names on x-axis
    yticklabels=processing_stages,  # Processing stages on y-axis
    linewidths=0.5                  # Add gridlines
)

# Add title and labels
ax.yaxis.set_tick_params(labelrotation=0)
ax.set_title('Timing Analysis of YOLOv11 Models per Image', fontsize=14, fontweight='bold')
ax.set_xlabel('Model', fontsize=12)

# Adjust layout for better appearance
plt.tight_layout()

# Save the heatmap as an image
output_file = 'timing_analysis_heatmap.png'
plt.savefig(output_file, dpi=300)

# Confirm the heatmap was saved
print(f"Heatmap saved as {output_file}")

# Show the plot
plt.show()
