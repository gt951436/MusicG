import pandas as pd
import numpy as np
import joblib
import tensorflow as tf 
from sklearn.model_selection import train_test_split

print("--- Model Evaluation Script ---")

try:
    # --- 1. Load and Prepare the Test Data ----------------
    print("\n[1/4] Loading and preparing test data...")

    df = pd.read_csv("features.csv")
    X = df.drop('genre_label', axis=1)
    y = df['genre_label']

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print("Test data loaded and split successfully.")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    # --- 2. Load the scikit-learn Models and the Scaler -----------
    print("\n[2/4] Loading scikit-learn models and scaler...")

    scaler = joblib.load('scaler.joblib')

    log_reg_model = joblib.load('logistic_regression_model.joblib')
    svm_model = joblib.load('svm_model.joblib')
    rf_model = joblib.load('random_forest_model.joblib')

    print("Scikit-learn assets loaded successfully.")
    print(f"Scaler: {type(scaler)}")
    print(f"Logistic Regression Model: {type(log_reg_model)}")
    print(f"SVM Model: {type(svm_model)}")
    print(f"Random Forest Model: {type(rf_model)}")

    # --- 3. Load the Keras CNN Model ------------------------
    print("\n[3/4] Loading Keras CNN model...")

    cnn_model = tf.keras.models.load_model('music_genre_cnn.h5')

    print("Keras CNN model loaded successfully.")
    print(f"CNN Model: {type(cnn_model)}")
    
    cnn_model.summary()

    # --- 4. Prepare Test Data for Different Model Types ------------
    print("\n[4/4] Preparing test data for model predictions...")

    # for scikit-learn models, only scale the data
    X_test_scaled = scaler.transform(X_test)
    print(f"Shape of X_test_scaled (for scikit-learn): {X_test_scaled.shape}")

    # for CNN model,scale AND reshape the data to 3D
    X_test_cnn = np.expand_dims(X_test_scaled, axis=-1)
    print(f"Shape of X_test_cnn (for Keras): {X_test_cnn.shape}")
    
    print("\nAll models and data are loaded and ready for evaluation!")
    
    # --- 5. Generate Predictions for Each Model -----------------------
    print("\n[5/5] Generating predictions on the test set...")

    # Predictions for scikit-learn models
    y_pred_log_reg = log_reg_model.predict(X_test_scaled)
    y_pred_svm = svm_model.predict(X_test_scaled)
    y_pred_rf = rf_model.predict(X_test_scaled)
    print("Predictions generated for scikit-learn models.")

    # Predictions for Keras CNN model
    y_pred_cnn_probs = cnn_model.predict(X_test_cnn)
    y_pred_cnn = np.argmax(y_pred_cnn_probs, axis=1)
    print("Predictions generated for Keras CNN model.")

    # --- Verification Step ---
    print("\n--- Verifying Prediction Shapes ---")
    print(f"LR Predictions Shape: {y_pred_log_reg.shape}")
    print(f"SVM Predictions Shape: {y_pred_svm.shape}")
    print(f"RFC Predictions Shape: {y_pred_rf.shape}")
    print(f"CNN Predictions Shape: {y_pred_cnn.shape}")

    print("\nAll predictions have been generated successfully!")

except FileNotFoundError as e:
    print(f"\nERROR: A required file was not found: {e.filename}")
    print("Please ensure all model files ('scaler.joblib', '*.joblib', 'music_genre_cnn.h5') and 'features.csv' are in the correct directory.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")