import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf


Conv1D = tf.keras.layers.Conv1D
MaxPooling1D = tf.keras.layers.MaxPooling1D
BatchNormalization = tf.keras.layers.BatchNormalization
Dropout = tf.keras.layers.Dropout
Flatten = tf.keras.layers.Flatten
Dense = tf.keras.layers.Dense

# ---Loading and Preparing Data---
CSV_PATH = "features.csv"

try:
    df = pd.read_csv(CSV_PATH)
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print(f"Error: The file at '{CSV_PATH}' was not found.")
    exit()
    
X = df.drop('genre_label', axis=1)
y = df['genre_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nData preparation complete. Shapes before reshaping:")
print(f"X_train_scaled shape: {X_train_scaled.shape}")
print(f"X_test_scaled shape: {X_test_scaled.shape}")

#----Reshaping data for CNN i/p -------
print("\n--- Reshaping data for CNN model ---")

X_train_cnn = np.expand_dims(X_train_scaled, axis=2)
X_test_cnn = np.expand_dims(X_test_scaled, axis=2)

#----verification----
print("\nShapes after reshaping for CNN:")
print(f"X_train_cnn shape: {X_train_cnn.shape}") 
print(f"X_test_cnn shape: {X_test_cnn.shape}")

print(f"\ny_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# Initialize a Sequential model.
model = tf.keras.Sequential()
print("Sequential model canvas created successfully.")

#-----1st CONV-POOL-BN Layer-----
# Adding first 1D Convolutional layer
model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train_cnn.shape[1], 1)))
# Adding a MaxPooling1D layer for down-sampling feature maps
model.add(MaxPooling1D(pool_size=2))
# Adding Batch Normalization layer for stabilizing and speeding up training
model.add(BatchNormalization())

#-----2nd CONV-POOL-BN Layer-----
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(BatchNormalization())

#-----3rd CONV-POOL-BN Layer-----
model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(BatchNormalization())

#-----Flatten Layer-----
model.add(Flatten())

#----Dense Classification Head-----
# Add a Dense layer
model.add(Dense(units=64, activation='relu'))

# Add a Dropout layer for regularization for overfitting prevention
model.add(Dropout(0.3))

# --- Verification Step: Model Summary ---
model.summary()


