import os
from typing import List, Tuple, Dict
from PIL import Image
import torchaudio
import pandas as pd
import torch

AUDIO_EXTENSIONS = ['.wav', '.mp3', '.flac']
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png']


# 🔹 Generic audio loader for a folder
def load_audio_dataset(path: str, sample_rate: int = 16000) -> List[Tuple[str, torch.Tensor]]:
    audio_data = []
    for file in os.listdir(path):
        if file.endswith(tuple(AUDIO_EXTENSIONS)):
            file_path = os.path.join(path, file)
            waveform, sr = torchaudio.load(file_path)
            if sr != sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
                waveform = resampler(waveform)
            audio_data.append((file, waveform))
    return audio_data


# ✅ 1. Load raw audio for processing
def load_raw_audio_dataset(base_path: str, sample_rate: int = 16000) -> List[Tuple[str, torch.Tensor]]:
    raw_audio_path = os.path.join(base_path, "audio", "raw")
    return load_audio_dataset(raw_audio_path, sample_rate)


# ✅ 2. Load STT dataset (paired audio-text)
def load_stt_dataset(base_path: str, csv_name: str = "metadata.csv") -> List[Tuple[torch.Tensor, str]]:
    stt_path = os.path.join(base_path, "audio", "stt")
    csv_path = os.path.join(stt_path, csv_name)

    df = pd.read_csv(csv_path)
    pairs = []

    for _, row in df.iterrows():
        audio_path = os.path.join(stt_path, row['filename'])
        waveform, _ = torchaudio.load(audio_path)
        pairs.append((waveform, row['text']))
    
    return pairs


# ✅ 3. Load voice cloning dataset (grouped by speaker folders)
def load_voice_cloning_dataset(base_path: str, sample_rate: int = 16000) -> Dict[str, List[torch.Tensor]]:
    cloning_path = os.path.join(base_path, "audio", "voice_cloning")
    speaker_data = {}

    for speaker in os.listdir(cloning_path):
        speaker_dir = os.path.join(cloning_path, speaker)
        if not os.path.isdir(speaker_dir):
            continue
        waveforms = []
        for file in os.listdir(speaker_dir):
            if file.endswith(tuple(AUDIO_EXTENSIONS)):
                audio_path = os.path.join(speaker_dir, file)
                waveform, sr = torchaudio.load(audio_path)
                if sr != sample_rate:
                    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
                    waveform = resampler(waveform)
                waveforms.append(waveform)
        speaker_data[speaker] = waveforms

    return speaker_data


# ✅ 4. Load image dataset
def load_image_dataset(base_path: str, image_size=(128, 128)) -> List[Tuple[str, Image.Image]]:
    image_dir = os.path.join(base_path, 'images')
    images = []

    for file in os.listdir(image_dir):
        if file.lower().endswith(tuple(IMAGE_EXTENSIONS)):
            img_path = os.path.join(image_dir, file)
            img = Image.open(img_path).convert('RGB')
            img = img.resize(image_size)
            images.append((file, img))

    return images
