import matplotlib.pyplot as plt
import numpy as np

# Data extracted
models = [
    'forward (512 fixed)', 'BIMODAL (512 fixed)', 'FB-RNN (512 fixed)', 'NADE (512 fixed)',
    'BIMODAL (512 random)', 'FB-RNN (512 random)', 'NADE (512 random)',
    'forward (1024 fixed)', 'BIMODAL (1024 fixed)', 'FB-RNN (1024 fixed)', 'NADE (1024 fixed)',
    'BIMODAL (1024 random)', 'FB-RNN (1024 random)', 'NADE (1024 random)',
    'BIMODAL (512 random 5x)', 'BIMODAL (512 random 10x)', 'FB-RNN (512 random 5x)', 'FB-RNN (512 random 10x)',
    'NADE (512 random 5x)', 'NADE (512 random 10x)', 'MORLD'
]

unique = [100, 99.8, 99.4, 100, 100, 100, 100, 100, 99.4, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
valid  = [93, 79, 51, 19, 89, 60, 7, 95, 84, 53, 21, 89, 63, 6, 91, 94, 62, 64, 5, 6, 95]
novel  = [89, 79, 51, 18, 89, 60, 7, 77, 81, 52, 21, 88, 62, 6, 90, 94, 62, 64, 5, 6, 100]

# Combine everything and sort based on 'novel'
combined = list(zip(models, unique, valid, novel))
combined.sort(key=lambda x: x[3])  # Sort by the novel value (index 3)

# Unpack the sorted data
models_sorted, unique_sorted, valid_sorted, novel_sorted = zip(*combined)

# Set position of bar on X axis
x = np.arange(len(models_sorted))
width = 0.25  # width of the bars

# Create the figure and axis
fig, ax = plt.subplots(figsize=(18, 8))

# Default bar colors
default_unique_color = 'tab:blue'
default_valid_color = 'tab:green'
default_novel_color = 'tab:red'

# Special bright colors for MORLD
morld_unique_color = 'deepskyblue'
morld_valid_color = 'lime'
morld_novel_color = 'orangered'

# Create color lists for each bar group
unique_colors = [morld_unique_color if model == 'MORLD' else default_unique_color for model in models_sorted]
valid_colors  = [morld_valid_color if model == 'MORLD' else default_valid_color for model in models_sorted]
novel_colors  = [morld_novel_color if model == 'MORLD' else default_novel_color for model in models_sorted]

# Make the bars with the appropriate colors
rects1 = ax.bar(x - width, unique_sorted, width, label='Unique (%)', color=unique_colors)
rects2 = ax.bar(x, valid_sorted, width, label='Valid (%)', color=valid_colors)
rects3 = ax.bar(x + width, novel_sorted, width, label='Novel (%)', color=novel_colors)

# Function to add value labels on top of bars
def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.0f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

# Add labels to all groups
add_labels(rects1)
add_labels(rects2)
add_labels(rects3)

# Add labels, title, and legend
ax.set_xlabel('Model')
ax.set_ylabel('Percentage')
ax.set_title('Comparatiive Analysis of Models')
ax.set_xticks(x)
ax.set_xticklabels(models_sorted, rotation=90)
ax.legend()

# Set y-axis limit
ax.set_ylim(0, 110)

# Add grid for better readability
ax.yaxis.grid(True)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300)
plt.show()
