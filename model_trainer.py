import pandas as pd 
import numpy as np 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

CSV_PATH = "features.csv"

try:
    df = pd.read_csv(CSV_PATH)
    print("Dataset successfully loaded!")
    
    X = df.drop("genre_label",axis=1)
    y = df['genre_label']
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    
    print("\nFirst 5 rows of Feature matrix (X):")
    print(X.head())
    
    print("\nFirst 5 rows of Target Vector (y):")
    print(y.head())
    
    # Encode genre labels
    if np.issubdtype(y.dtype,np.integer):
        print("Labels are already numerically encoded!")
    else:
        print("Labels are categorical, applying Label Encoding...")
    
    #print("\nVerification of target variable 'y':")
    #print(f"Data type of y: {y.dtype}")
    #print("First 5 labels:")
    #print(y.head())
    
    print("\n--- Splitting Data into Training and Testing Sets ---")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    #print("\nVerifying the shapes of the new sets:")
    #print(f"Shape of X_train: {X_train.shape}") 
    #print(f"Shape of X_test: {X_test.shape}")  
    #print(f"Shape of y_train: {y_train.shape}") 
    #print(f"Shape of y_test: {y_test.shape}")  
    
    print("\n--- Scaling Features ---")
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    #print("StandardScaler has been fitted to the training data.")
    #print(f"Learned means (μ) for the first 5 features: {scaler.mean_[:5]}")
    #print(f"Learned standard deviations (σ) for the first 5 features: {scaler.scale_[:5]}")
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("\nFeatures have been scaled.")
    
    #print("\nVerification of scaled data:")
    #print(f"Mean of first 5 features in X_train_scaled: {X_train_scaled[:, :5].mean(axis=0)}")
    #should be very close to 1.
    #print(f"Standard deviation of first 5 features in X_train_scaled: {X_train_scaled[:, :5].std(axis=0)}")
    #print(f"\nMean of first 5 features in X_test_scaled: {X_test_scaled[:, :5].mean(axis=0)}")
    
    print("\n--- Training Logistic Regression Model ---")
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train_scaled,y_train)
    print("Logistic Regression model trained successfully.")
    print(f"Model learned the following classes: {log_reg.classes_}")
    
    
except FileNotFoundError:
    print(f"Error: The file at '{CSV_PATH}' was not found.")
    print("Please ensure 'features.csv' is in the same directory.")
except Exception as e:
    print(f"An error occurred: {e}")
    