import os
import parselmouth
import pandas as pd
import numpy as np

def extract_intensity_features(wav_dir, output_file):
    files = [f for f in os.listdir(wav_dir) if f.endswith('.wav')]
    results = []
    
    print(f"Extracting INTENSITY features from {len(files)} files...")
    
    for idx, file in enumerate(files):
        filepath = os.path.join(wav_dir, file)
        try:
            sound = parselmouth.Sound(filepath)
            # Create Intensity object
            intensity = sound.to_intensity()
            
            # Get values
            # Intensity values are in dB
            # We can get the average over the whole duration
            mean_intensity = intensity.get_average()
            
            # To get SD, we need the array of values
            # The 'values' property of Intensity object is a numpy array of arrays (frames x 1)
            # intensity.values returns shape (1, n_frames)
            intensity_values = intensity.values[0]
            
            sd_intensity = np.std(intensity_values)
            max_intensity = np.max(intensity_values)
            min_intensity = np.min(intensity_values)
            
            results.append({
                "filename": file,
                "intensity_mean": mean_intensity,
                "intensity_sd": sd_intensity,
                "intensity_min": min_intensity,
                "intensity_max": max_intensity
            })
            
        except Exception as e:
            print(f"Error processing {file}: {e}")
            
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Intensity features saved to {output_file}")

if __name__ == "__main__":
    wav_folder = r"e:\Project\CMF\wavs"
    output_csv = r"e:\Project\CMF\features\features_intensity.csv"
    extract_intensity_features(wav_folder, output_csv)
