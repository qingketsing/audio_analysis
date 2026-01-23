import os
import parselmouth
import pandas as pd
import numpy as np

def extract_voice_quality_features(wav_dir, output_file):
    files = [f for f in os.listdir(wav_dir) if f.endswith('.wav')]
    results = []
    
    print(f"Extracting VOICE QUALITY (Jitter/Shimmer/HNR) features from {len(files)} files...")
    
    for idx, file in enumerate(files):
        filepath = os.path.join(wav_dir, file)
        try:
            sound = parselmouth.Sound(filepath)
            
            # For Jitter and Shimmer, we need Pulses (PointProcess)
            pitch = sound.to_pitch()
            # Use call with both Sound and Pitch to generate PointProcess using Cross-Correlation (cc)
            pulses = parselmouth.praat.call([sound, pitch], "To PointProcess (cc)")
            
            # Jitter (local)
            # shortest period=0.0001s, longest=0.02s (standard 50Hz-1000Hz range approx)
            # max_period_factor=1.3 is standard Praat default
            jitter = parselmouth.praat.call(pulses, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3)
            
            # Shimmer (local)
            # max_period_factor=1.3, max_amplitude_factor=1.6 standard Praat defaults
            shimmer = parselmouth.praat.call([sound, pulses], "Get shimmer (local)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6)
            
            # HNR (Harmonics-to-Noise Ratio)
            # time_step=0.01 (standard), min_pitch=75.0 (standard), silence_threshold=0.1, periods_per_window=1.0
            harmonicity = sound.to_harmonicity()
            hnr = parselmouth.praat.call(harmonicity, "Get mean", 0.0, 0.0)
            
            results.append({
                "filename": file,
                "jitter": jitter,
                "shimmer": shimmer,
                "hnr": hnr
            })
            
        except Exception as e:
            # If pitch detection fails or file is too short/silent
            print(f"Error processing {file}: {e}")
            results.append({
                "filename": file,
                "jitter": np.nan,
                "shimmer": np.nan,
                "hnr": np.nan
            })
            
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Voice Quality features saved to {output_file}")

if __name__ == "__main__":
    wav_folder = r"e:\Project\CMF\wavs"
    output_csv = r"e:\Project\CMF\features\features_voice_quality.csv"
    extract_voice_quality_features(wav_folder, output_csv)
