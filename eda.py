import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

CSV_PATH = "features.csv"

try:
   df = pd.read_csv(CSV_PATH)
   print("Dataframes loaded successfully!")
   print("\n--- DataFrame Info ---")
   print(df.info())
   print("\n--- DataFrame Statistical Summary ---")
   print(df.describe())
   print("\n--- Missing Values Check ---")
   missing_values_count = df.isnull().sum()
    
   print("Number of missing values per column:")
   print(missing_values_count)
   
   if missing_values_count.sum() == 0:
       print("\nThe dataset is complete! no missing values.")
   else:
       print("\nThe dataset is missing values, needs handling.")
       
   genre_names = [
        'blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock'
    ]
   sns.set_style("whitegrid")
   plt.figure(figsize=(12,6))
   
   ax = sns.countplot(x='genre_label',data=df,palette = 'viridis')
   
   ax.set_title("Distribution of Music Genres in the Dataset",fontsize=16)
   ax.set_xlabel("Genre",fontsize=12)
   ax.set_ylabel("Number of Segments",fontsize=12)
   
   ax.set_xticklabels(genre_names,rotation=30)
   
   plt.tight_layout()
   plt.show()
   
   print("\n--- Generating Box Plot for Spectral Centroid ---")
   
   plt.figure(figsize=(14,7))
   box_ax = sns.boxplot(x='genre_label',y='25',data=df,palette='cubehelix')
   
   box_ax.set_title('Spectral Centroid Distribution Across Genres', fontsize=18)
   box_ax.set_xlabel('Genre', fontsize=14)
   box_ax.set_ylabel('Spectral Centroid', fontsize=14)
   
   box_ax.set_xticklabels(genre_names, rotation=30, ha="right") # 'ha' aligns the labels better

   plt.tight_layout()
   plt.show()
   
   print("\n--- Generating Violin Plot for First MFCC (Column 0) ---")
   
   plt.figure(figsize=(14, 7))
   violin_ax = sns.violinplot(x='genre_label', y='0', data=df, palette='Spectral')
   
   violin_ax.set_title('First MFCC (Timbre/Energy) Distribution Across Genres', fontsize=18)
   violin_ax.set_xlabel('Genre', fontsize=14)
   violin_ax.set_ylabel('MFCC 1 Value', fontsize=14)
   
   violin_ax.set_xticklabels(genre_names, rotation=30, ha="right")

   plt.tight_layout()
   plt.show()
       
except FileNotFoundError:
    print(f"Error:The file at '{CSV_PATH}' was not found.")
    print("Ensure running 'feature_extractor.py' file first to generate the dataset.")
except Exception as e:
    print(f"Error while loading Dataframe: {e}")

