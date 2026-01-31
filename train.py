import os
os.system("cls")
print("import libs..")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import unicodedata
import csv



# -----------------------------
# 1️⃣ Hyperparameter
# -----------------------------
HIDDEN_SIZE = 128
BATCH_SIZE = 64
EPOCHS = 50
LR = 0.003
MAX_NAME_LEN = 20  # maximale Länge der generierten Namen
TEMPERATURE = 0.8

# -----------------------------
# 2️⃣ Prozeduraler Seed-Generator
# -----------------------------
TECHNO_SEEDS = ["flux", "knyx", "xen", "volt", "syn"]
METAL_SEEDS = ["iron", "ash", "void", "grim", "skull"]
POP_SEEDS = ["luna", "mira", "ely", "aria", "nova"]

def mutate(word):
    mods = [
        lambda w: w + random.choice(["x", "or", "ion", "a"]),
        lambda w: w.replace("i", "y"),
        lambda w: w.capitalize(),
        lambda w: w + random.choice(["a", "e", "u"]),
    ]
    return random.choice(mods)(word)

def generate_names(seeds, n=3000):
    names = set()
    while len(names) < n:
        base = random.choice(seeds)
        name = mutate(base)
        # Unicode-Normalisierung
        name = unicodedata.normalize('NFC', name)
        names.add(name)
    return list(names)

# -----------------------------
# 3️⃣ Trainingsdaten vorbereiten
# -----------------------------
if not os.path.exists("train_names.txt"):
    techno_names = generate_names(TECHNO_SEEDS, 4000)
    metal_names = generate_names(METAL_SEEDS, 3000)
    pop_names = generate_names(POP_SEEDS, 3000)
    all_names = techno_names + metal_names + pop_names
    with open("train_names.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(all_names))
else:
    with open("train_names.txt", encoding="utf-8") as f:
        all_names = [line.strip() for line in f if line.strip()]

print(f"Anzahl Trainingsnamen: {len(all_names)}")

# -----------------------------
# 4️⃣ Vokabular (inkl. Sonderzeichen)
# -----------------------------
chars = sorted(list(set("".join(all_names))))
chars.append("\n")  # Ende-Zeichen
char2idx = {c:i for i,c in enumerate(chars)}
idx2char = {i:c for i,c in enumerate(chars)}
vocab_size = len(chars)

# -----------------------------
# 5️⃣ Dataset
# -----------------------------
class NameDataset(Dataset):
    def __init__(self, names, char2idx, max_len=MAX_NAME_LEN):
        self.data = []
        for name in names:
            name = unicodedata.normalize('NFC', name.lower() + "\n")
            for i in range(1, len(name)):
                seq = name[:i]
                target = name[i]
                seq_idx = [char2idx[c] for c in seq]
                target_idx = char2idx[target]
                self.data.append((seq_idx, target_idx))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq, target = self.data[idx]
        seq_tensor = torch.tensor(seq, dtype=torch.long)
        target_tensor = torch.tensor(target, dtype=torch.long)
        return seq_tensor, target_tensor

dataset = NameDataset(all_names, char2idx)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda batch: (
    nn.utils.rnn.pad_sequence([x[0] for x in batch], batch_first=True),
    torch.tensor([x[1] for x in batch])
))

# -----------------------------
# 6️⃣ LSTM Modell
# -----------------------------
class NameLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size=HIDDEN_SIZE):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden

model = NameLSTM(vocab_size)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# -----------------------------
# 7️⃣ Training
# -----------------------------
print("Training startet...")
for epoch in range(EPOCHS):
    total_loss = 0
    for seqs, targets in dataloader:
        optimizer.zero_grad()
        outputs, _ = model(seqs)
        outputs = outputs[:, -1, :]
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(dataloader):.4f}")

# Modell speichern
torch.save(model.state_dict(), "name_lstm_unicode.pt")
print("Training abgeschlossen, Modell gespeichert!")

# -----------------------------
# 8️⃣ Name Generator
# -----------------------------
def generate_name(model, seed="", max_len=MAX_NAME_LEN, temperature=TEMPERATURE):
    model.eval()
    seed = unicodedata.normalize('NFC', seed.lower())
    input_ids = torch.tensor([[char2idx[c] for c in seed if c in char2idx]])
    hidden = None
    result = seed

    for _ in range(max_len):
        out, hidden = model(input_ids, hidden)
        logits = out[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        idx = torch.multinomial(probs, 1).item()
        char = idx2char[idx]
        if char == "\n":
            break
        result += char
        input_ids = torch.tensor([[idx]])
    return result.capitalize()

# -----------------------------
# 9️⃣ Nachträgliche Stilzuweisung
# -----------------------------
def assign_style(name):
    name_lower = name.lower()
    if any(c in name_lower for c in "xzqk"):
        return "TECHNO"
    elif any(c in name_lower for c in "grshv"):
        return "METAL"
    else:
        return "POP"

# -----------------------------
# 10️⃣ Test Generierung
# -----------------------------
#print("\nBeispiel generierter Namen mit Stil:")
#for _ in range(10):
#    seed = random.choice(["knx","flux","knoxx","xel","sig","björ"])
#    n = generate_name(model, seed=seed)
#    style = assign_style(n)
#    print(f"{n} → {style}")


# -----------------------------
# 11️⃣ Batch-Generierung mit variabler Länge
# -----------------------------
def generate_batch(model, num_names=10000, max_len=MAX_NAME_LEN, seeds=None, variable_length=False):
    """
    Generiert eine große Menge an neuen Namen.
    
    Args:
        model: trainiertes LSTM-Modell
        num_names: Anzahl der Namen
        max_len: maximale Länge der Namen
        seeds: optionale Seed-Liste für Inspiration
        variable_length: bool, ob Namen zufällige Länge <= max_len haben
    Returns:
        Liste von Tupeln: (Name, Stil)
    """
    generated = set()
    seeds = seeds or ["knx","flux","knoxx","xel","sig","björ","frey","eld","pad","heið"]
    
    while len(generated) < num_names:
        seed = random.choice(seeds)
        # Variable Länge, falls aktiviert
        current_max_len = random.randint(6, max_len) if variable_length else max_len
        name = generate_name(model, seed=seed, max_len=current_max_len)
        style = assign_style(name)
        generated.add((name, style))
    
    return list(generated)

# -----------------------------
# 12️⃣ Batch speichern
# -----------------------------
def save_batch_to_csv(batch, filename="generated_artists.txt"):
    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Name", "Stil"])
        for name, style in batch:
            writer.writerow([name, style])
    print(f"{len(batch)} Namen in {filename} gespeichert!")

# -----------------------------
# 13️⃣ Beispiel Batch-Generierung
# -----------------------------
# Flag zum Aktivieren variabler Namenlängen
variable_length_flag = True

print("\nStarte Batch-Generierung von 100 Namen...")
batch = generate_batch(model, num_names=100, variable_length=variable_length_flag)
save_batch_to_csv(batch)
print("Fertig!")









