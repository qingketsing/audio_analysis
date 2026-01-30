import os
import sys
import types
import importlib.machinery

# Avoid torchvision ops (nms) issues in this environment
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Minimal stub for torchvision to bypass missing nms
if "torchvision" not in sys.modules:
    def _make_module(name, is_pkg=False):
        mod = types.ModuleType(name)
        mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=is_pkg)
        if is_pkg:
            mod.__path__ = []
        sys.modules[name] = mod
        return mod

    tv_stub = _make_module("torchvision", is_pkg=True)
    transforms_mod = _make_module("torchvision.transforms", is_pkg=True)
    class _Interp:
        NEAREST = 0
        NEAREST_EXACT = 1
        BILINEAR = 2
        BICUBIC = 3
        LANCZOS = 4
        HAMMING = 5
        BOX = 6
    transforms_mod.InterpolationMode = _Interp

    v2_mod = _make_module("torchvision.transforms.v2", is_pkg=True)
    v2_functional = _make_module("torchvision.transforms.v2.functional")
    v2_mod.functional = v2_functional
    sys.modules["torchvision.transforms.functional"] = v2_functional  # alias

    datasets_mod = _make_module("torchvision.datasets", is_pkg=True)
    io_mod = _make_module("torchvision.io", is_pkg=True)
    models_mod = _make_module("torchvision.models", is_pkg=True)
    ops_mod = _make_module("torchvision.ops", is_pkg=True)
    utils_mod = _make_module("torchvision.utils", is_pkg=True)

    # Attach submodules as attributes on root stub
    tv_stub.transforms = transforms_mod
    tv_stub.datasets = datasets_mod
    tv_stub.io = io_mod
    tv_stub.models = models_mod
    tv_stub.ops = ops_mod
    tv_stub.utils = utils_mod

import pandas as pd

# Reduce allocator fragmentation for tight VRAM budgets
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torchaudio
import soundfile as sf
import numpy as np
from tqdm.auto import tqdm
from transformers.pipelines import pipeline

def extract_emotion_features(
    wav_dir,
    output_file,
    max_batch_size=1,
    chunk_size=16,
    gpu_mem_fraction=0.75,
    max_duration_seconds=20.0,
):
    files = sorted([f for f in os.listdir(wav_dir) if f.endswith('.wav')])
    if not files:
        print("No wav files found.")
        return

    print("Loading Transformers Emotion Recognition Pipeline (superb/wav2vec2-base-superb-er)...")

    if torch.cuda.is_available():
        device_id = 0
        print(f"Device selection: GPU (ID: {device_id})")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"GPU Compute Capability: {props.major}.{props.minor}")
        try:
            torch.cuda.set_per_process_memory_fraction(gpu_mem_fraction, device=device_id)
            print(f"GPU memory cap set to {gpu_mem_fraction*100:.0f}% of total")
        except Exception as e:
            print(f"Warning: failed to set GPU memory cap: {e}")
    else:
        device_id = -1
        print("CUDA not available, falling back to CPU. Inference will be slower.")

    # 加载 Pipeline
    try:
        classifier = pipeline("audio-classification", model="superb/wav2vec2-base-superb-er", device=device_id)
        device_desc = "GPU" if device_id >= 0 else "CPU"
        print(f"Model loaded on {device_desc} successfully")
    except Exception as e:
        print(f"Failed to load transformers pipeline: {e}")
        return

    print(f"Extracting EMOTION features from {len(files)} files...")
    
    emotion_map = {
        "neu":   {"A": 0.0,  "V": 0.0},
        "ang":   {"A": 0.8,  "V": -0.6},
        "hap":   {"A": 0.6,  "V": 0.6}, 
        "sad":   {"A": -0.5, "V": -0.6}
    }

    results = []

    total = len(files)
    print(f"Processing {total} files using batch_size up to {max_batch_size}, chunk_size {chunk_size}...")

    # 当前批大小会随 OOM 自动下降，并在后续批次沿用更小值
    current_bs = max_batch_size

    pbar = tqdm(total=total, desc="Emotion inference", unit="file")

    for start in range(0, total, chunk_size):
        batch_files = files[start:start + chunk_size]
        loaded = []

        for file in batch_files:
            path = os.path.join(wav_dir, file)
            try:
                audio_data, sample_rate = sf.read(path)
                audio_data = audio_data.astype(np.float32)
                if audio_data.ndim > 1:
                    audio_data = np.mean(audio_data, axis=1)
                if max_duration_seconds is not None:
                    max_len = int(max_duration_seconds * sample_rate)
                    if audio_data.shape[0] > max_len:
                        audio_data = audio_data[:max_len]
                if sample_rate != 16000:
                    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                    audio_data = resampler(torch.from_numpy(audio_data)).numpy()
                    sample_rate = 16000
                loaded.append({"file": file, "audio": {"array": audio_data, "sampling_rate": sample_rate}})
            except Exception as e:
                print(f"Skipping {file}: {e}")
                loaded.append({"file": file, "audio": None})

        # 控制每次送入模型的 batch_size，避免 OOM
        bs = current_bs
        while True:
            try:
                audio_inputs = [item["audio"] if item["audio"] is not None else None for item in loaded]
                preds_batch = classifier(audio_inputs, top_k=None, batch_size=bs)
                break
            except RuntimeError as e:
                if "CUDA out of memory" in str(e) and bs > 1 and device_id >= 0:
                    prev_bs = bs
                    bs = max(1, bs // 2)
                    current_bs = bs
                    torch.cuda.empty_cache()
                    print(f"CUDA OOM at batch_size={prev_bs} for files {start}-{start+len(batch_files)}; retrying with {bs}")
                    continue
                raise
            except Exception as e:
                print(f"Error processing files {start}-{start+len(batch_files)}: {e}")
                preds_batch = [[] for _ in batch_files]
                break

        for item, preds in zip(loaded, preds_batch):
            file = item["file"]
            if item["audio"] is None:
                results.append({
                    "filename": file,
                    "emotion_top": "",
                    "prob_ang": 0,
                    "prob_hap": 0,
                    "prob_sad": 0,
                    "prob_neu": 0,
                    "arousal_est": 0,
                    "valence_est": 0
                })
                continue
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

        processed = min(start + chunk_size, total)
        pbar.update(len(batch_files))
        if device_id >= 0:
            torch.cuda.empty_cache()

    pbar.close()

    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Emotion features saved to {output_file}")


if __name__ == "__main__":
    wav_folder = r"e:\Project\CMF\wavs"
    output_csv = r"e:\Project\CMF\features\features_emotion.csv"
    extract_emotion_features(wav_folder, output_csv)
