import os
import pandas as pd
import torch
import torchaudio
from transformers import pipeline

def extract_emotion_features(wav_dir, output_file):
    files = [f for f in os.listdir(wav_dir) if f.endswith('.wav')]
    results = []
    
    print(f"Loading Transformers Emotion Recognition Pipeline (superb/wav2vec2-base-superb-er)...")
    try:
        # Use top_k=None to get all probabilities
        classifier = pipeline("audio-classification", model="superb/wav2vec2-base-superb-er")
    except Exception as e:
        print(f"Failed to load transformers pipeline: {e}")
        return

    print(f"Extracting EMOTION features from {len(files)} files...")
    
    # Mapping to Arousal (A) and Valence (V) - Approximate centroids
    # Superb model classes: usually 'neu', 'hap', 'ang', 'sad'
    # Check labels at runtime.
    
    # Map (A, V) values (-1 to 1)
    emotion_map = {
        "neu":   {"A": 0.0,  "V": 0.0},
        "ang":   {"A": 0.8,  "V": -0.6},
        "hap":   {"A": 0.6,  "V": 0.6}, # Happiness
        "sad":   {"A": -0.5, "V": -0.6}
    }
    
    for idx, file in enumerate(files):
        filepath = os.path.join(wav_dir, file)
        try:
            # Pipeline is easy: pass filename
            preds = classifier(filepath, top_k=None)
            
            # preds is list of dicts [{'score': 0.9, 'label': 'neu'}, ...]
            
            arousal = 0.0
            valence = 0.0
            
            probs_dict = {}
            top_label = ""
            max_score = -1
            
            for p in preds:
                label = p['label']
                score = p['score']
                probs_dict[label] = score
                
                if score > max_score:
                    max_score = score
                    top_label = label
                
                # Weighted mapping
                if label in emotion_map:
                    arousal += score * emotion_map[label]["A"]
                    valence += score * emotion_map[label]["V"]
            
            results.append({
                "filename": file,
                "emotion_top": top_label,
                "prob_ang": probs_dict.get("ang", 0),
                "prob_hap": probs_dict.get("hap", 0),
                "prob_sad": probs_dict.get("sad", 0),
                "prob_neu": probs_dict.get("neu", 0),
                "arousal_est": arousal,
                "valence_est": valence
            })
            
            if (idx+1) % 10 == 0:
                print(f"Processed {idx+1}/{len(files)}")
            
        except Exception as e:
            print(f"Error processing {file}: {e}")
            
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Emotion features saved to {output_file}")


if __name__ == "__main__":
    wav_folder = r"e:\Project\CMF\wavs"
    output_csv = r"e:\Project\CMF\features\features_emotion.csv"
    extract_emotion_features(wav_folder, output_csv)
