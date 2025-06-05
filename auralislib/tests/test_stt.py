import torch
import pytest
from auralislib.models.ctc_model import CTCmodel
from auralislib.STT.inference import decode_ctc, chunk_and_predict, transcribe
from unittest.mock import patch

def test_ctcmodel_forward_shape():
    model = CTCmodel(input_dim=13, output_dim=29)
    x = torch.randn(1, 10, 13)  # batch=1, seq_len=10, input_dim=13
    out = model(x)
    assert out.shape == (1, 10, 29), f"Unexpected output shape {out.shape}"

def test_decode_ctc_simple():
    # sequence with repeats and blanks
    preds = [torch.tensor([0,1,1,0,2,2,0,3])]
    idx_to_char = {0:"_", 1:"a", 2:"b", 3:"c"}
    decoded = decode_ctc(preds, blank=0, idx_to_char=idx_to_char)
    assert decoded[0] == "abc"

def test_chunking_logic(monkeypatch):
    # Patch model to return fixed output to avoid dependency on trained weights
    class DummyModel:
        def __call__(self, x):
            batch, seq_len, _ = x.shape
            return torch.randn(batch, seq_len, 29)  # dummy logits
    
    dummy_model = DummyModel()
    audio = torch.randn(35000)  # longer than chunk size 16000 with overlap 4000
    
    text = chunk_and_predict(audio, dummy_model)
    assert isinstance(text, str)
    # Number of chunks = ceil((35000 - 4000) / (16000 - 4000)) ~ 3
    # text should have 3 segments joined
