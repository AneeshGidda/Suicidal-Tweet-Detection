# Import necessary libraries
import pandas as pd  # For handling data with DataFrames
import tensorflow as tf  # For working with TensorFlow
import numpy as np  # For numerical operations
from sklearn.model_selection import train_test_split  # For splitting the dataset
from transformers import DistilBertTokenizer  # For using the DistilBERT tokenizer

# Load the dataset from a CSV file into a DataFrame
data = pd.read_csv("Suicide_Ideation_Dataset(Twitter-based).csv")

# Define a mapping from string labels to numerical labels
numerical_encoding = {"Not Suicide post": 0, "Potential Suicide post": 1}

# Replace the string labels with numerical labels in the DataFrame
data["Suicide"] = data["Suicide"].replace(numerical_encoding)

# Extract the text data and labels from the DataFrame
x_data = data["Tweet"].values.astype(str)
y_data = data["Suicide"].values.astype(int)

# Split the data into training and testing sets (80% training, 20% testing)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

# Initialize the DistilBERT tokenizer from the Hugging Face Transformers library
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenize the training data and convert it to TensorFlow tensors
x_train_tokenized = tokenizer.batch_encode_plus(x_train,
                                                add_special_tokens=True,
                                                return_attention_mask=True,
                                                padding=True,
                                                return_tensors="tf")

# Extract the input IDs and attention masks for the training data
x_train_input_ids = x_train_tokenized["input_ids"]
x_train_attention_mask = x_train_tokenized["attention_mask"]

# Tokenize the testing data and convert it to TensorFlow tensors
x_test_tokenized = tokenizer.batch_encode_plus(x_test,
                                               add_special_tokens=True,
                                               return_attention_mask=True,
                                               padding=True,
                                               return_tensors="tf")

# Extract the input IDs and attention masks for the testing data
x_test_input_ids = x_test_tokenized["input_ids"]
x_test_attention_mask = x_test_tokenized["attention_mask"]

# Convert the training and testing labels to one-hot encoded format using TensorFlow
y_train_one_hot = tf.one_hot(y_train, depth=2)
y_test_one_hot = tf.one_hot(y_test, depth=2)

# Save the tokenized input data and labels as .npy files for future use
np.save("x_train_input_ids.npy", x_train_input_ids)
np.save("x_train_attention_mask.npy", x_train_attention_mask)
np.save("x_test_input_ids.npy", x_test_input_ids)
np.save("x_test_attention_mask.npy", x_test_attention_mask)
np.save("y_train.npy", y_train_one_hot)
np.save("y_test.npy", y_test_one_hot)
