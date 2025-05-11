import pandas as pd

import matplotlib.pyplot as plt

# Load the CSV file
file_path = '/home/satya/Desktop/BIOInfromaticsRl-LSTM/results/RL_Training/reward_metrics_log.csv'
data = pd.read_csv(file_path, header=None)

# Assign column names for better readability
data.columns = ['epoch', 'col1', 'col2', 'col3', 'qed', 'col5', 'col6']

# Convert the 'epoch' column to integers
data['epoch'] = pd.to_numeric(data['epoch'], errors='coerce')

# Drop rows with invalid 'epoch' values (e.g., NaN after conversion)
data = data.dropna(subset=['epoch'])

# Filter data for epochs up to 100
filtered_data = data[data['epoch'] <= 100]

# Plot QED vs Epoch with improved formatting
plt.figure(figsize=(10, 6))
plt.plot(filtered_data['epoch'], filtered_data['qed'], marker='o', linestyle='-', color='b', label='QED')

# Add labels, title, and legend with better formatting
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('QED', fontsize=12)
plt.title('QED vs Epoch', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)

# Show the plot
plt.tight_layout()
plt.show()