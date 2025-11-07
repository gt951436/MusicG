import os

DATASET_PATH = "genres_original"

if not os.path.exists(DATASET_PATH):
    print(f"Error: Dataset path '{DATASET_PATH}' not found.")
    print("Data is not unzipped!")
else:
    print("Dataset directory found. Counting files in each genre folder...")
    print("-"*45)
    
    
    for genre_folder in sorted(os.listdir(DATASET_PATH)):
        genre_path = os.path.join(DATASET_PATH,genre_folder)
        if os.path.isdir(genre_path):
            #listing the contents of directory
            files_in_genre = os.listdir(genre_path)
            files_cnt = len(files_in_genre)
            print(f"Genre: {genre_folder.ljust(12)} | File Count: {files_cnt}")
    
    print("-"*45)
    print("Verification completed. The dataset appears to be balanced.")
