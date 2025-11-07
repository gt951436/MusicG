import pandas as pd

CSV_PATH = "features.csv"

try:
   df = pd.read_csv(CSV_PATH)
   print("Dataframes loaded successfully!")
   print("\n--- DataFrame Info ---")
   print(df.info())
   print("\n--- DataFrame Statistical Summary ---")
   print(df.describe())
except FileNotFoundError:
    print(f"Error:The file at '{CSV_PATH}' was not found.")
    print("Ensure running 'feature_extractor.py' file first to generate the dataset.")
except Exception as e:
    print(f"Error while loading Dataframe: {e}")
