import matplotlib.pyplot as plt
import numpy as np
import os

# Model names (n, s, m, l, x)
models = ['nano', 'small', 'medium', 'large', 'extreme']

# Bar width
bar_width = 0.15

# Positions for the bars on the x-axis (metrics)
metrics = ['mAP', 'Precision', 'Recall', 'F1 Score']
index = np.arange(len(metrics))

# Create a figure and axis
fig, ax = plt.subplots(figsize=(12, 6))

# Plotting the bars for each model
bar1 = ax.bar(index - 2*bar_width, [0.90, 0.80, 0.80, 0.84], bar_width, label='nano', color='royalblue')
bar2 = ax.bar(index - bar_width, [0.84, 0.82, 0.80, 0.81], bar_width, label='small', color='mediumseagreen')
bar3 = ax.bar(index, [0.86, 0.85, 0.82, 0.83], bar_width, label='medium', color='orange')
bar4 = ax.bar(index + bar_width, [0.86, 0.84, 0.82, 0.83], bar_width, label='large', color='tomato')
bar5 = ax.bar(index + 2*bar_width, [0.94, 0.93, 0.86, 0.89], bar_width, label='extreme', color='purple')  # Extreme model's F1 values

# Adding labels, title, and legend
ax.set_xlabel('Evaluation Metrics')
ax.set_ylabel('Scores')
ax.set_title('Model Performance Comparison: YOLOv11 (n, s, m, l, x)')
ax.set_xticks(index)
ax.set_xticklabels(metrics)
ax.legend()

# Adding gridlines for better readability
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Tight layout to prevent clipping of labels
plt.tight_layout()

# Save the plot in the best quality (300 dpi is ideal for publication)
output_file = 'model_performance_comparison.png'
plt.savefig(output_file, dpi=300)

# Confirm that the plot was saved
if os.path.exists(output_file):
    print(f"Plot saved successfully as {output_file}")
else:
    print("Failed to save the plot.")

# Optionally display the plot
plt.show()
