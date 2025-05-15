import torch
import torchaudio
import os
from datasets import load_dataset  # assuming this is your module

def process_dataset(dataset_path: str, sample_rate: int = 16000):
    audio_files = load_dataset.load_audio_dataset(dataset_path, sample_rate)

    processed = []
    for filename, waveform, sr in audio_files:
        waveform = resample_audio(waveform, orig_sr=sr, target_sr=sample_rate)
        waveform = normalize_audio(waveform)
        waveform_segments = split_audio(waveform, segment_length=sample_rate * 5)  # 5s

        for i, segment in enumerate(waveform_segments):
            name = f"{filename}_part{i}.wav"
            saved = save_audio(segment, name)
            processed.append((name, saved))

    return processed

def normalize_audio(waveform: torch.Tensor) -> torch.Tensor:
    mean = waveform.mean()
    std = waveform.std()
    return (waveform - mean) / std

def resample_audio(waveform: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    if orig_sr == target_sr:
        return waveform
    resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
    return resampler(waveform)

def split_audio(waveform: torch.Tensor, segment_length: int) -> list:
    segments = []
    for i in range(0, len(waveform[0]), segment_length):
        segment = waveform[:, i:i + segment_length]
        if segment.shape[1] == segment_length:
            segments.append(segment)
    return segments

def save_audio(waveform: torch.Tensor, filename: str) -> str:
    path = os.path.join("processed_audio", filename)
    os.makedirs("processed_audio", exist_ok=True)
    torchaudio.save(path, waveform, 16000)  # Save at 16kHz
    return path
