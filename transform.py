import os
from moviepy import VideoFileClip

def convert_videos_to_wav(video_dir, wav_dir):
    # Ensure usage of the correct directories
    if not os.path.exists(wav_dir):
        os.makedirs(wav_dir)
        print(f"Created directory: {wav_dir}")

    files = os.listdir(video_dir)
    mp4_files = [f for f in files if f.lower().endswith('.mp4')]
    
    total_files = len(mp4_files)
    print(f"Found {total_files} .mp4 files in {video_dir}")

    for idx, file in enumerate(mp4_files):
        video_path = os.path.join(video_dir, file)
        wav_filename = os.path.splitext(file)[0] + ".wav"
        wav_path = os.path.join(wav_dir, wav_filename)

        print(f"[{idx+1}/{total_files}] Processing {file} ...")

        try:
            # Using context manager is simpler if available, but for VideoFileClip explicit close is safer in loops
            video = None
            try:
                video = VideoFileClip(video_path)
                if video.audio:
                    # Write audio to wav
                    # codec='pcm_s16le' is standard for wav and good for parslemouth/speechbrain
                    video.audio.write_audiofile(wav_path, codec='pcm_s16le', logger=None)
                    print(f" -> Saved to {wav_filename}")
                else:
                    print(f" -> No audio stream found in {file}")
            finally:
                if video:
                    video.close()
                    
        except Exception as e:
            print(f" -> Error converting {file}: {e}")

if __name__ == "__main__":
    video_folder = r"e:\Project\CMF\vedios"
    wav_folder = r"e:\Project\CMF\wavs"
    
    convert_videos_to_wav(video_folder, wav_folder)
