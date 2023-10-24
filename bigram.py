import torch
import torch.nn as nn
from torch.nn import functional as F

#--- Hyperparameters ---#
context_length = 8
batch_size = 32
max_iters = 10_000
learning_rate = 1e-2
eval_interval = 0.1 * max_iters
eval_iters = 200
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

#---

torch.manual_seed(123)


# Text laden
faust = open("goethe_faust.txt", "r", encoding="utf-8").read()
faust_2 = open("goethe_faust_2.txt", "r", encoding="utf-8").read()

# Alle Zeichen als Variable und Vokabelgröße
chars = sorted(list(set(faust + faust_2)))
vocab_size = len(chars)

# Look-up Tabellen
stoi = {s:i for i,s in enumerate(chars)} # look-up Tabelle, die Zeichen Ziffern zuweist
itos = {i:s for s,i in stoi.items()} # look-up Tabelle, die Ziffern Zeichen zuweist
encode = lambda s: [stoi[l] for l in s] # Kodierungsfunktion
decode = lambda i: ''.join([itos[ix] for ix in i]) # Dekodierungsfunktion

# Texte tokenizieren
data_faust = torch.tensor(encode(faust), dtype = torch.long)
data_faust_2 = torch.tensor(encode(faust_2), dtype = torch.long)

# Train, val test splits
train_length = int(0.9 * len(data_faust))
data_train = data_faust[:train_length]
data_val = data_faust[train_length:]
data_test = data_faust_2

# Batch Funktion
def get_batch(split: str="train",
              batch_size: int=batch_size,
              device: str=device) -> tuple[torch.tensor, torch.tensor]:
    """Erzeugt Batches für das parallele Training"""
    data = data_train if split == "train" else data_val if split == "val" else data_test
    ix = torch.randint(len(data) - context_length, (batch_size,))
    x = torch.stack([data[i:i + context_length] for i in ix])
    y = torch.stack([data[i+1: i+context_length+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

class BigrammModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx shape and targets shape = batch_size, context_length
        logits = self.token_embedding_table(idx) # (batch_size, context_length, vocab_size) = (B, T, C)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits.view(B*T,C), targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            probs = F.softmax(logits[:,-1,:], dim=-1)
            idx_new = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_new), dim=1)
        return idx

@torch.no_grad()
def estimate_loss(model,
                  batch_size: int=batch_size,
                  eval_iters: int=eval_iters) -> dict:
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, batch_size)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    return out


def training(model,
             steps: int = max_iters,
             lr: float=learning_rate,
             batch_size: int=batch_size,
             eval_interval: int=eval_interval,
             device: str=device
             ):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    for i in range(steps):
        xb, yb = get_batch("train", batch_size, device)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True) # set gradients to 0
        loss.backward() # Backprop
        optimizer.step() # update parameters
        if i % eval_interval == 0:
            losses = estimate_loss(model, batch_size)
            print(f"Mittlerer Train loss nach {i} Epochen: {losses['train']} || Mittlerer Val loss: {losses['val']}")

model = BigrammModel(vocab_size)
m = model.to(device)

training(m)

print(decode(m.generate(torch.zeros((1,1), dtype=torch.long, device=device), 500)[0].tolist()))
