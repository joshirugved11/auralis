from auralislib.models.ctc_model import CTCmodel
from auralislib.audio.preprocessing import load_audio, extract_mfcc
import torch
import torch.nn.functional as F

def transcribe(path):
    waveform = load_audio(path)
    mfcc = extract_mfcc(waveform).squeeze(0).transpose(0, 1).unsqueeze(0)
    
    model = CTCmodel
    model.eval()
    with torch.no_grad():
        output = model(mfcc)  
        probs = F.log_softmax(output, dim=-1)
        pred = torch.argmax(probs, dim=-1).squeeze(0).tolist()
        return decode_pred(pred)
    
def decode_pred(pred):
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    return ''.join([alphabet[i] for i in pred if i < len(alphabet)])  # Assuming pred contains indices of characters in the alphabet