import torch
import torchaudio
import os
from datasets import load_dataset

def process_dataset(dataset_path: str, sample_rate: int = 16000):
    audio_files = load_dataset.load_audio_dataset(dataset_path, sample_rate)

    processed = []
    for filename, waveform, sr in audio_files:
        waveform = normalize_audio(waveform)
        waveform = resample_audio(waveform, orig_sr=sr, target_sr=sample_rate)
        waveform = save_audio(waveform, filename)
        processed.append((filename, waveform))
    
    return processed

def normalize_audio(waveform: torch.Tensor) -> torch.Tensor:
    mean = waveform.mean()
    std = waveform.std()
    normailsed_waveform = (waveform - mean) / std
    return normailsed_waveform

def resample_audio(waveform: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    if orig_sr == target_sr:
        return waveform
    resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
    return resampler(waveform)

def split_audio(waveform: torch.Tensor, segment_length: int) -> list:
    segments = []
    for i in range(0, len(waveform), segment_length):
        segment = waveform[i:i + segment_length]
        if len(segment) == segment_length:
            segments.append(segment)
    return segments

def save_audio(waveform: torch.Tensor, filename: str, sample_rate) -> str:
    output_path = os.path.join("processed_audio", filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torchaudio.save(output_path, waveform, sample_rate=16000)
    return output_path