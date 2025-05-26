import torchaudio
import torchaudio.transforms as T

def load_audio(file_path, sample_rate=16000):
    waveform, sr = torchaudio.load(file_path)
    if sr != sample_rate:
        resampler = T.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)
    return waveform

def extract_mfcc(waveform, sample_rate=16000, n_mfcc=13):
    mfcc = T.MFCC(sample_rate=sample_rate, n_mfcc=n_mfcc)(waveform)
    return mfcc

def extract_mel(waveform, sample_rate=16000, n_mels=128):
    mel = T.MelSpectrogram(sample_rate=sample_rate)(waveform)
    return mel