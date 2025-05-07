import os
from datasets import load_dataset
from audio import processing
import torchaudio

# Path to raw audio and output folder
DATASET_PATH = "datasets/"
OUTPUT_PATH = "datasets/processed_audio/"

def main():
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # Load all audio files
    audio_data = load_dataset.load_audio_dataset(DATASET_PATH)

    print(f"[INFO] Loaded {len(audio_data)} audio files for processing.")

    for filename, waveform in audio_data:
        # Process audio
        waveform = processing.normalize_audio(waveform)
        waveform = processing.trim_silence(waveform)
        waveform = processing.resample_audio(waveform, orig_sr=44100, target_sr=16000)

        # Save processed audio
        save_path = os.path.join(OUTPUT_PATH, filename)
        torchaudio.save(save_path, waveform, 16000)
        print(f"[OK] Processed and saved: {filename}")

if __name__ == "__main__":
    main()
