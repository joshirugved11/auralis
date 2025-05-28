import torch.nn as nn

class TacotronStub(nn.Module):
    def __init__(self, embedding_dim=256, vocab_size=1000, mel_dim=150):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, 512, batch_first=True)
        self.linear = nn.Linear(512, mel_dim)

    def forward(self, text_seq):
        embedded = self.embedding(text_seq)
        output, _ = self.lstm(embedded)
        mel = self.linear(output)
        return mel
         