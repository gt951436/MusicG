import pandas as pd 
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
    
except FileNotFoundError:
    print(f"Error: The file at '{CSV_PATH}' was not found.")
    print("Please ensure 'features.csv' is in the same directory.")
except Exception as e:
    print(f"An error occurred: {e}")
    