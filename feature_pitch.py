import os
import parselmouth
import pandas as pd
import numpy as np

def extract_pitch_features(wav_dir, output_file):
    files = [f for f in os.listdir(wav_dir) if f.endswith('.wav')]
    results = []
    
    print(f"Extracting PITCH features from {len(files)} files...")
    
    for idx, file in enumerate(files):
        filepath = os.path.join(wav_dir, file)
        try:
            sound = parselmouth.Sound(filepath)
            # Create a Pitch object (standard parameters)
            pitch = sound.to_pitch()
            
            # Get values excluding unvoiced frames (where pitch is 0/nan)
            pitch_values = pitch.selected_array['frequency']
            pitch_values = pitch_values[pitch_values != 0]
            
            if len(pitch_values) > 0:
                mean_f0 = np.mean(pitch_values)
                sd_f0 = np.std(pitch_values)
                min_f0 = np.min(pitch_values)
                max_f0 = np.max(pitch_values)
            else:
                mean_f0 = 0
                sd_f0 = 0
                min_f0 = 0
                max_f0 = 0
                
            results.append({
                "filename": file,
                "pitch_mean": mean_f0,
                "pitch_sd": sd_f0,
                "pitch_min": min_f0,
                "pitch_max": max_f0
            })
            
        except Exception as e:
            print(f"Error processing {file}: {e}")
            
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Pitch features saved to {output_file}")

if __name__ == "__main__":
    wav_folder = r"e:\Project\CMF\wavs"
    output_csv = r"e:\Project\CMF\features\features_pitch.csv"
    extract_pitch_features(wav_folder, output_csv)
