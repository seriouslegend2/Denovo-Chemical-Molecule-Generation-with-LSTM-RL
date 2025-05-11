import os
import numpy as np
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import Sequence
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# ✅ Set GPU Memory Limit
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5000)]
            )
        print("✅ GPU memory limit set to 5GB.")
    except RuntimeError as e:
        print(f"🚨 Error setting GPU memory limit: {e}")

# ✅ Load Dataset
dataset_file = "/home/satya/Desktop/BIOInfromaticsRl-LSTM/processed_data/split_data/split_1.pkl"
print(f"🔄 Loading dataset from {dataset_file}")
with open(dataset_file, "rb") as f:
    data = pickle.load(f)

X = pad_sequences([[data["char_to_idx"].get(token, data["char_to_idx"]["<UNK>"]) 
                    for token in tokens] for tokens in data["tokenized_smiles"]],
                   maxlen=data["max_smiles_length"], padding="post")
y = X[:, 1:]
X = X[:, :-1]

# ✅ Split into Train-Test
split_index = int(len(X) * 0.7)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# ✅ Define LSTM Model
def build_lstm_model(input_dim, output_dim, input_length):
    model = Sequential([
        Embedding(input_dim=input_dim, output_dim=128, input_length=input_length),
        LSTM(256, return_sequences=True),
        LSTM(256, return_sequences=True),
        Dense(output_dim, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

model = build_lstm_model(len(data["char_to_idx"]), len(data["char_to_idx"]), data["max_smiles_length"] - 1)

# ✅ Data Generator
class SparseDataGenerator(Sequence):
    def __init__(self, X, y, batch_size):
        self.X = X
        self.y = y
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        X_batch = self.X[index * self.batch_size:(index + 1) * self.batch_size]
        y_batch = self.y[index * self.batch_size:(index + 1) * self.batch_size]
        return X_batch, y_batch

# ✅ Training Settings
batch_size = 16
epochs = 10
data_fraction = 0.3

num_samples = len(X_train)
num_samples_per_epoch = int(num_samples * data_fraction)

# ✅ Prepare to Track Metrics
train_losses = []
train_accuracies = []

# ✅ Train Model
for epoch in range(epochs):
    print(f"🔄 Epoch {epoch + 1}/{epochs}")

    indices = np.random.choice(num_samples, num_samples_per_epoch, replace=False)
    X_train_subset = X_train[indices]
    y_train_subset = y_train[indices]

    train_generator = SparseDataGenerator(X_train_subset, y_train_subset, batch_size)

    history = model.fit(train_generator, epochs=1, verbose=1)
    train_losses.append(history.history['loss'][0])
    train_accuracies.append(history.history['accuracy'][0])

# ✅ Save Training Metrics Plot
results_dir = "/home/satya/Desktop/BIOInfromaticsRl-LSTM/results/LSTM_Results/"
os.makedirs(results_dir, exist_ok=True)

plt.figure(figsize=(12, 5))

# Plot Training Loss
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), train_losses, marker='o', label='Training Loss')
plt.title('Training Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

# Plot Training Accuracy
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), train_accuracies, marker='o', color='green', label='Training Accuracy')
plt.title('Training Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(results_dir, "training_metrics.png"))
plt.show()
print(f"✅ Saved training metrics plot at {results_dir}training_metrics.png")

# ✅ Evaluate Model
test_generator = SparseDataGenerator(X_test, y_test, batch_size)
test_loss, test_accuracy = model.evaluate(test_generator, batch_size=batch_size)
print(f"✅ Test Loss: {test_loss}")
print(f"✅ Test Accuracy: {test_accuracy}")

# Save test metrics for plotting
test_metrics = {"loss": test_loss, "accuracy": test_accuracy}
test_metrics_path = os.path.join(results_dir, "test_metrics.pkl")
with open(test_metrics_path, "wb") as f:
    pickle.dump(test_metrics, f)
print(f"✅ Test metrics saved at {test_metrics_path}")

# ✅ Save Testing Metrics Plot
plt.figure(figsize=(6, 4))

# Plot Test Metrics
plt.bar(["Test Loss", "Test Accuracy"], [test_loss, test_accuracy], color=["red", "blue"])
plt.title("Testing Metrics")
plt.ylabel("Value")
plt.tight_layout()

plt.savefig(os.path.join(results_dir, "testing_metrics.png"))
plt.show()
print(f"✅ Saved testing metrics plot at {results_dir}testing_metrics.png")

# ✅ Predict
y_test_pred_classes = []
for i in range(len(test_generator)):
    X_batch, _ = test_generator[i]
    y_batch_pred = model.predict(X_batch, batch_size=batch_size)
    y_batch_pred_classes = np.argmax(y_batch_pred, axis=-1)
    y_test_pred_classes.extend(y_batch_pred_classes)

y_test_pred_classes = np.array(y_test_pred_classes)

y_test_pred_flat = y_test_pred_classes.flatten()
y_test_true_flat = y_test.flatten()

# ✅ Classification Report
report = classification_report(y_test_true_flat, y_test_pred_flat, zero_division=0)
print("✅ Classification Report:")
print(report)

# Save classification report to a file
report_path = os.path.join(results_dir, "classification_report.txt")
with open(report_path, "w") as f:
    f.write(report)
print(f"✅ Classification report saved at {report_path}")

# ✅ Confusion Matrix as Heatmap
sample_size = 5000  # Sample
y_test_sample_true = y_test_true_flat[:sample_size]
y_test_sample_pred = y_test_pred_flat[:sample_size]

cm = confusion_matrix(y_test_sample_true, y_test_sample_pred, labels=np.arange(len(data["char_to_idx"])))

plt.figure(figsize=(12, 10))
plt.imshow(cm, interpolation='nearest', cmap='coolwarm')
plt.title('Confusion Matrix Heatmap (Sample)')
plt.colorbar()
tick_marks = np.arange(len(data["char_to_idx"]))
plt.xticks(tick_marks, list(data["char_to_idx"].keys()), rotation=90)
plt.yticks(tick_marks, list(data["char_to_idx"].keys()))

# Add labels to each cell
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black")

plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()

heatmap_path = os.path.join(results_dir, "confusion_matrix_heatmap.png")
plt.savefig(heatmap_path)
plt.show()
print(f"✅ Saved confusion matrix heatmap at {heatmap_path}")

# ✅ Save Trained Model
model_path = "/home/satya/Desktop/BIOInfromaticsRl-LSTM/saved_model/Orig/lstm_generator.h5"
os.makedirs(os.path.dirname(model_path), exist_ok=True)
model.save(model_path)
print(f"✅ Model saved at {model_path}")
