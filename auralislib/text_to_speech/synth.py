from auralislib.models.tacotron_stub import TacotronStub
import torch

def text_to_tensor(text):
    vocab = "abcdefghijklmnopqrstuvwxyz "
    return torch.tensor([vocab.index(char) for char in text.lower() if char in vocab])

def synthesize(text, out_path=None):
    text_seq = text_to_tensor(text)
    model = TacotronStub()
    model.eval()
    with torch.no_grad():
        mel = model(text_seq)
    return mel