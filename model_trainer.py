import pandas as pd 
import numpy as np 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

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
    
    print("\nVerifying the shapes of the new sets:")
    print(f"Shape of X_train: {X_train.shape}") 
    print(f"Shape of X_test: {X_test.shape}")  
    print(f"Shape of y_train: {y_train.shape}") 
    print(f"Shape of y_test: {y_test.shape}")   
    
except FileNotFoundError:
    print(f"Error: The file at '{CSV_PATH}' was not found.")
    print("Please ensure 'features.csv' is in the same directory.")
except Exception as e:
    print(f"An error occurred: {e}")
    