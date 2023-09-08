# Import necessary libraries
import tensorflow as tf  # For working with TensorFlow
import numpy as np  # For numerical operations
from transformers import TFDistilBertForSequenceClassification  # For using DistilBERT

# Load tokenized input data and labels for training and testing sets
x_train_input_ids = tf.convert_to_tensor(np.load("x_train_input_ids.npy"))
x_train_attention_mask = tf.convert_to_tensor(np.load("x_train_attention_mask.npy"))
x_test_input_ids = tf.convert_to_tensor(np.load("x_test_input_ids.npy"))
x_test_attention_mask = tf.convert_to_tensor(np.load("x_test_attention_mask.npy"))

y_train = tf.convert_to_tensor(np.load("y_train.npy"))
y_test = tf.convert_to_tensor(np.load("y_test.npy"))

# Initialize a DistilBERT model for sequence classification
bert_model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

# Compile the model with loss, optimizer, and evaluation metric
bert_model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
    metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")]
)

# Train the DistilBERT model on the training data for 3 epochs
bert_model.fit(
    {"input_ids": x_train_input_ids, "attention_mask": x_train_attention_mask},
    y_train,
    epochs=3
)
