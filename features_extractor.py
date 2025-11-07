import os, json, librosa
import numpy as np

DATASET_PATH = "genres_original"
JSON_PATH = "data.json"

SAMPLE_RATE = 22050
TRACK_DURATION = 30 #SECONDS
NUM_SEGMENTS = 10 #each track in 10 segemnts

#for mfcc extraction
NUM_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512

SAMPLES_PER_TRACK = SAMPLE_RATE*TRACK_DURATION


def process_dataset(dataset_path,json_path):
    
    data = {
        "mapping":[], # genre names list
        "labels":[],  # int : genre label for each track
        "features":[] # features of each track
    }
    print("Starting feature extraction...")
    
    for i,genre_folder in enumerate(sorted(os.listdir(dataset_path))):
        genre_path = os.path.join(dataset_path,genre_folder)
        
        if os.path.isdir(genre_path):
            data["mapping"].append(genre_folder)
            print(f"\nProcessing genre: {genre_folder}")
            
            for filename in sorted(os.listdir(genre_path)):
                if filename.endswith(".wav"):
                    file_path = os.path.join(genre_path,filename)
                    
                    try:
                        signal,sr = librosa.load(file_path,sr=SAMPLE_RATE)
                        #signal segmentation
                        if len(signal)>=SAMPLES_PER_TRACK:
                            samples_per_segment = int(SAMPLES_PER_TRACK/NUM_SEGMENTS)
                            
                            for s in range(NUM_SEGMENTS):
                                starting_sample = s * samples_per_segment
                                ending_sample = starting_sample + samples_per_segment
                                segment = signal[starting_sample:ending_sample] #3 second chunk
                                
                                mfccs = librosa.feature.mfcc(y=segment,sr=sr,n_mfcc=NUM_MFCC,n_fft=N_FFT,hop_length=HOP_LENGTH)
                                mfccs_processed = np.mean(mfccs,axis=1)
                                
                                chroma = librosa.feature.chroma_stft(y=segment,sr=sr,n_fft=N_FFT,hop_length=HOP_LENGTH)
                                chroma_processed = np.mean(chroma,axis=1)
                                
                                spectral_centroid = librosa.feature.spectral_centroid(y=segment,sr=sr,n_fft=N_FFT,hop_length=HOP_LENGTH)
                                spectral_centroid_processed = np.mean(spectral_centroid)
                                
                                spectral_rolloff = librosa.feature.spectral_rolloff(y=segment,sr=sr,n_fft=N_FFT,hop_length=HOP_LENGTH)
                                spectral_rolloff_processed = np.mean(spectral_rolloff)
                                
                                zcr = librosa.feature.zero_crossing_rate(y=segment,hop_length=HOP_LENGTH)
                                zcr_processed = np.mean(zcr)
                                
                                feature_vector = np.hstack((mfccs_processed,chroma_processed,spectral_centroid_processed,spectral_rolloff_processed,zcr_processed))
                                
                                data["features"].append(feature_vector.tolist())
                                data["labels"].append(i)
                                         
                    except Exception as e:
                        print(f"Error loading files {file_path}: {e}")
                        continue
    
    print("Feature extraction done.")
    
if __name__ =="__main__":
    process_dataset(DATASET_PATH,JSON_PATH)