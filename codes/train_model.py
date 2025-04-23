import os
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import Sequence
from sklearn.metrics import classification_report  # Import classification_report

# ✅ Configure GPU Memory Limit
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Set memory growth and limit GPU memory usage to 5.5GB
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5632)]  # 5.5GB limit (5632 MB)
            )
        print("✅ GPU memory limit set to 5.5GB.")
    except RuntimeError as e:
        print(f"🚨 Error setting GPU memory limit: {e}")

# ✅ Load Dataset
dataset_file = "/home/satya/Desktop/BIOInfromaticsRl-LSTM/processed_data/split_data/split_1.pkl"  # Use one dataset
print(f"🔄 Loading dataset from {dataset_file}")
with open(dataset_file, "rb") as f:
    data = pickle.load(f)

X = pad_sequences([[data["char_to_idx"].get(token, data["char_to_idx"]["<UNK>"]) 
                    for token in tokens] for tokens in data["tokenized_smiles"]],
                   maxlen=data["max_smiles_length"], padding="post")
y = X[:, 1:]  # All tokens except first (next-token prediction)
X = X[:, :-1]  # All tokens except last

# ✅ Split Data into Training (70%) and Testing (30%)
split_index = int(len(X) * 0.7)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# ✅ Define LSTM Model with Sparse Categorical Cross-Entropy
def build_lstm_model(input_dim, output_dim, input_length):
    model = Sequential([
        Embedding(input_dim=input_dim, output_dim=128, input_length=input_length),
        LSTM(256, return_sequences=True),
        LSTM(256, return_sequences=True),
        Dense(output_dim, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# ✅ Initialize Model
model = build_lstm_model(len(data["char_to_idx"]), len(data["char_to_idx"]), data["max_smiles_length"] - 1)

# ✅ Data Generator Class (No One-Hot Encoding)
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
        return X_batch, y_batch  # Return integer-encoded labels directly

# ✅ Train Model by Selecting 30% of Data Randomly for Each Epoch
batch_size = 16
epochs = 10  # Number of epochs to train
data_fraction = 0.3  # Fraction of data to select randomly for each epoch

num_samples = len(X_train)
num_samples_per_epoch = int(num_samples * data_fraction)

for epoch in range(epochs):
    print(f"🔄 Epoch {epoch + 1}/{epochs}")

    # Randomly select 30% of the training data for this epoch
    indices = np.random.choice(num_samples, num_samples_per_epoch, replace=False)
    X_train_subset = X_train[indices]
    y_train_subset = y_train[indices]

    # Initialize Sparse Data Generator for the subset
    train_generator = SparseDataGenerator(X_train_subset, y_train_subset, batch_size)

    # Train on the subset
    model.fit(train_generator, epochs=1)  # Train for 1 epoch on the subset

# ✅ Evaluate Model on Test Data
test_generator = SparseDataGenerator(X_test, y_test, batch_size)
test_loss, test_accuracy = model.evaluate(test_generator, batch_size=batch_size)
print(f"✅ Test Loss: {test_loss}")
print(f"✅ Test Accuracy: {test_accuracy}")

# ✅ Predict Test Data in Smaller Batches
y_test_pred_classes = []
for i in range(len(test_generator)):
    X_batch, _ = test_generator[i]  # Get a batch of test data
    y_batch_pred = model.predict(X_batch, batch_size=batch_size)
    y_batch_pred_classes = np.argmax(y_batch_pred, axis=-1)
    y_test_pred_classes.extend(y_batch_pred_classes)

y_test_pred_classes = np.array(y_test_pred_classes)

# Flatten predictions and true labels for classification report
y_test_pred_flat = y_test_pred_classes.flatten()
y_test_true_flat = y_test.flatten()

# Generate classification report
report = classification_report(y_test_true_flat, y_test_pred_flat, zero_division=0)
print("✅ Classification Report:")
print(report)

# ✅ Save Trained Model
model_path = "/home/satya/Desktop/BIOInfromaticsRl-LSTM/saved_model/Orig/lstm_generator.h5"
os.makedirs(os.path.dirname(model_path), exist_ok=True)
model.save(model_path)
print(f"✅ Model saved at {model_path}")
