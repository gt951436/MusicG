import pandas as pd
import numpy as np
import joblib
import tensorflow as tf 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import seaborn as sns

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
    #print(f"Scaler: {type(scaler)}")
    #print(f"Logistic Regression Model: {type(log_reg_model)}")
    #print(f"SVM Model: {type(svm_model)}")
    #print(f"Random Forest Model: {type(rf_model)}")

    # --- 3. Load the Keras CNN Model ------------------------
    print("\n[3/4] Loading Keras CNN model...")
    cnn_model = tf.keras.models.load_model('music_genre_cnn.h5')
    print("Keras CNN model loaded successfully.")
    #print(f"CNN Model: {type(cnn_model)}")
    
    #cnn_model.summary()

    # --- 4. Prepare Test Data for Different Model Types ------------
    print("\n[4/4] Preparing test data for model predictions...")
    # for scikit-learn models, only scale the data
    X_test_scaled = scaler.transform(X_test)
    # for CNN model,scale AND reshape the data to 3D
    X_test_cnn = np.expand_dims(X_test_scaled, axis=-1)
    
    print("\nAll models and data are ready for evaluation!")
    
    # =====================================================
    #    5 - GENERATE PREDICTIONS
    # =====================================================
    print("\n[5/5] Generating predictions on the test set...")
    
    # scikit-learn probability predictions
    proba_log_reg = log_reg_model.predict_proba(X_test_scaled)
    proba_svm = svm_model.predict_proba(X_test_scaled)
    proba_rf = rf_model.predict_proba(X_test_scaled)

    # CNN probabilities
    proba_cnn = cnn_model.predict(X_test_cnn)

     # class predictions
    y_pred_log_reg = np.argmax(proba_log_reg, axis=1)
    y_pred_svm = np.argmax(proba_svm, axis=1)
    y_pred_rf = np.argmax(proba_rf, axis=1)
    y_pred_cnn = np.argmax(proba_cnn, axis=1)

    # --- Verification Step ---
    #print("\n--- Verifying Prediction Shapes ---")
    #print(f"LR Predictions Shape: {y_pred_log_reg.shape}")
    #print(f"SVM Predictions Shape: {y_pred_svm.shape}")
    #print(f"RFC Predictions Shape: {y_pred_rf.shape}")
    #print(f"CNN Predictions Shape: {y_pred_cnn.shape}")

    print("\nAll predictions have been generated successfully!")
    
   # =====================================================
    #  6 — CLASSIFICATION REPORTS 
    # =====================================================
    genre_names = [
        'blues', 'classical', 'country', 'disco', 'hiphop', 
        'jazz', 'metal', 'pop', 'reggae', 'rock'
    ]

    print("\n======= CLASSIFICATION REPORTS =======")
    print("\nLogistic Regression:\n", classification_report(y_test, y_pred_log_reg, target_names=genre_names))
    print("\nSVM:\n", classification_report(y_test, y_pred_svm, target_names=genre_names))
    print("\nRandom Forest:\n", classification_report(y_test, y_pred_rf, target_names=genre_names))
    print("\nCNN:\n", classification_report(y_test, y_pred_cnn, target_names=genre_names))
    
    # =====================================================
    #   7 — CONFUSION MATRICES
    # =====================================================

    cm_log_reg = confusion_matrix(y_test, y_pred_log_reg)
    cm_svm = confusion_matrix(y_test, y_pred_svm)
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    cm_cnn = confusion_matrix(y_test, y_pred_cnn)

    print("\nConfusion matrices computed successfully.")
    
    # --- Visualize the Confusion Matrices as Heatmaps -------------------
    def plot_confusion_matrix(cm, labels, title, ax):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Confusion Matrices")

    plot_confusion_matrix(cm_log_reg, genre_names, "Logistic Regression", axes[0, 0])
    plot_confusion_matrix(cm_svm, genre_names, "SVM", axes[0, 1])
    plot_confusion_matrix(cm_rf, genre_names, "Random Forest", axes[1, 0])
    plot_confusion_matrix(cm_cnn, genre_names, "CNN", axes[1, 1])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    
    # =====================================================
    #  8 — MULTI-CLASS ROC CURVES
    # =====================================================
    
    print("\n======= ROC CURVES =======")
    
    # one-hot encode test labels
    lb = LabelBinarizer()
    y_test_bin = lb.fit_transform(y_test)
    
    def plot_multiclass_roc(y_true, y_proba, model_name, class_names):
        n_classes = len(class_names)
        fpr = {}
        tpr = {}
        roc_auc = {}

        # per-class ROC
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # micro-average ROC
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_proba.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        plt.figure(figsize=(10, 8))

        # micro-average
        plt.plot(
            fpr["micro"], tpr["micro"],
            label=f"Micro-average ROC (AUC = {roc_auc['micro']:.2f})",
            linewidth=2
        )

        # each class curve
        for i in range(n_classes):
            plt.plot(
                fpr[i], tpr[i],
                lw=2,
                label=f"{class_names[i]} (AUC = {roc_auc[i]:.2f})"
            )

        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve — {model_name}")
        plt.legend(loc="lower right")
        plt.show()
        
    # ---- PLOT ROC FOR EACH MODEL ----
    plot_multiclass_roc(y_test_bin, proba_log_reg, "Logistic Regression", genre_names)
    plot_multiclass_roc(y_test_bin, proba_svm, "SVM", genre_names)
    plot_multiclass_roc(y_test_bin, proba_rf, "Random Forest", genre_names)
    plot_multiclass_roc(y_test_bin, proba_cnn, "CNN", genre_names)

except FileNotFoundError as e:
    print(f"\nERROR: Missing file: {e.filename}")
except Exception as e:
    print(f"Unexpected Error: {e}")