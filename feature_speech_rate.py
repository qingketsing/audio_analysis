import os
import parselmouth
import pandas as pd
import numpy as np
from parselmouth.praat import call
from scipy.signal import find_peaks

def extract_speech_rate_features(wav_dir, output_file):
    files = [f for f in os.listdir(wav_dir) if f.endswith('.wav')]
    results = []
    
    print(f"Extracting SPEECH RATE features from {len(files)} files...")
    
    # Parameters
    min_pause_dur = 0.3   # seconds
    
    for idx, file in enumerate(files):
        filepath = os.path.join(wav_dir, file)
        try:
            sound = parselmouth.Sound(filepath)
            original_duration = sound.get_total_duration()
            
            # Create Intensity object
            # minimum pitch 100Hz for intensity is standard for speech
            intensity = sound.to_intensity(100)
            
            # Get values (dB)
            # intensity.values is (1, n_frames)
            values = intensity.values[0]
            
            # Get time step
            dt = intensity.get_time_step()
            
            if len(values) == 0:
                raise ValueError("No intensity values")
                
            max_int = np.max(values)
            min_int = np.min(values)
            
            # Threshold: standard is 25dB below max
            # But if dynamic range is small, maybe less?
            # We stick to max - 25 for "silence" relative to peak speech.
            threshold = max_int - 25.0
            
            # Boolean array: True if Silent (< threshold)
            is_silent = values < threshold
            
            # Find consecutive silent frames
            # Pad with False to detect edges
            padded = np.concatenate(([False], is_silent, [False]))
            # Find changes
            diff = np.diff(padded.astype(int))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            
            total_pause_duration = 0.0
            n_pauses = 0
            
            for s, e in zip(starts, ends):
                # Duration in frames = e - s
                dur_sec = (e - s) * dt
                if dur_sec >= min_pause_dur:
                    total_pause_duration += dur_sec
                    n_pauses += 1
            
            phonation_time = original_duration - total_pause_duration
            
            # If phonation time is tiny, avoid div by zero
            if phonation_time < 0.01:
                phonation_time = 0.01 # minimal
                
            # Count Syllables (Peaks)
            # Find peaks in Intensity > threshold (voiced)
            # standard distance between syllables approx 150-200ms? 
            # 0.15s / dt
            dist_frames = int(0.15 / dt)
            if dist_frames < 1: dist_frames = 1
            
            # Peak must be > threshold - 2 (to be slightly robust) or just threshold
            # Using scipy find_peaks
            peaks, _ = find_peaks(values, height=threshold, distance=dist_frames)
            syllable_count = len(peaks)
            
            speech_rate = syllable_count / original_duration if original_duration > 0 else 0
            articulation_rate = syllable_count / phonation_time
            
            results.append({
                "filename": file,
                "syllable_count_est": syllable_count,
                "pause_count": n_pauses,
                "total_pause_duration": total_pause_duration,
                "phonation_time": phonation_time,
                "speech_rate_syl_per_sec": speech_rate,
                "articulation_rate_syl_per_sec": articulation_rate
            })

            
        except Exception as e:
            print(f"Error processing {file}: {e}")
            results.append({
                "filename": file,
                "syllable_count_est": 0,
                "pause_count": 0,
                "total_pause_duration": 0,
                "phonation_time": 0,
                "speech_rate_syl_per_sec": 0,
                "articulation_rate_syl_per_sec": 0
            })
            
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Speech Rate features saved to {output_file}")

if __name__ == "__main__":
    wav_folder = r"e:\Project\CMF\wavs"
    output_csv = r"e:\Project\CMF\features\features_speech_rate.csv"
    extract_speech_rate_features(wav_folder, output_csv)
