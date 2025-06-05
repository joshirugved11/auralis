import torch
import torch.nn.functional as F
from auralislib.models.ctc_model import CTCmodel
from auralislib.audio.preprocessing import load_audio, extract_mfcc

# Character index map (example; customize to your model's vocab)
idx_to_char = {
    0: "_", 1: "a", 2: "b", 3: "c", 4: "d", 5: "e", 6: "f", 7: "g", 8: "h",
    9: "i", 10: "j", 11: "k", 12: "l", 13: "m", 14: "n", 15: "o", 16: "p",
    17: "q", 18: "r", 19: "s", 20: "t", 21: "u", 22: "v", 23: "w", 24: "x",
    25: "y", 26: "z", 27: " ", 28: "'"
}

BLANK_IDX = 0

def decode_ctc(preds, blank=BLANK_IDX, idx_to_char=None):
    pred_texts = []
    for pred in preds:
        collapsed = []
        previous = blank
        for p in pred:
            if p.item() != previous and p.item() != blank:
                collapsed.append(p.item())
            previous = p.item()
        if idx_to_char:
            pred_texts.append(''.join([idx_to_char[c] for c in collapsed]))
        else:
            pred_texts.append(collapsed)
    return pred_texts

def chunk_and_predict(audio_tensor, model, chunk_size=16000, overlap=4000):
    stride = chunk_size - overlap
    chunks = []
    for i in range(0, len(audio_tensor), stride):
        chunk = audio_tensor[i:i+chunk_size]
        if len(chunk) < chunk_size:
            chunk = torch.nn.functional.pad(chunk, (0, chunk_size - len(chunk)))
        chunks.append(chunk)

    texts = []
    for chunk in chunks:
        features = extract_mfcc(chunk).squeeze(0).transpose(0, 1).unsqueeze(0)
        with torch.no_grad():
            output = model(features)
            preds = torch.argmax(output, dim=2)
            decoded = decode_ctc(preds, idx_to_char=idx_to_char)
            texts.append(decoded[0])
    return ' '.join(texts)

def transcribe(path, use_chunking=True):
    waveform = load_audio(path)  # [1, T]
    model = CTCmodel(input_dim=13, output_dim=len(idx_to_char))  
    model.eval()

    if use_chunking:
        text = chunk_and_predict(waveform.squeeze(0), model)
    else:
        mfcc = extract_mfcc(waveform).squeeze(0).transpose(0, 1).unsqueeze(0)
        with torch.no_grad():
            output = model(mfcc)
            preds = torch.argmax(F.log_softmax(output, dim=-1), dim=-1)
            text = decode_ctc(preds, idx_to_char=idx_to_char)[0]

    return text
