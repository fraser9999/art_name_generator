import os
os.system("cls")
print("import libs..")

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import unicodedata
import csv

from datetime import datetime
from random import randrange

dir_path = os.path.dirname(os.path.realpath(__file__))

# -----------------------------
# 1️⃣ Hyperparameter / Einstellungen
# -----------------------------
HIDDEN_SIZE = 128
MAX_NAME_LEN = 20   # maximale Länge
TEMPERATURE = 0.8
VARIABLE_LENGTH_FLAG = True  # Zufällige Namenlängen aktivieren


MODEL_PATH = dir_path + "/" + "name_lstm_unicode.pt"  # Pfad zum gespeicherten Modell
CSV_OUTPUT = "generated_artists.csv"
NUM_NAMES = 100  # Anzahl generierter Namen

# -----------------------------
# 2️⃣ Vokabular definieren
#    Muss identisch sein wie beim Training
# -----------------------------
# Bei Unicode-Modellen: hier alle Zeichen aus dem Training eintragen
# Beispiel: einfache Demo-Zeichenliste, sollte beim Training genau aus all_names erstellt werden

trainfile = dir_path + "/" + "train_names.txt"

with open(trainfile, encoding="utf-8") as f:
    all_names = [line.strip() for line in f if line.strip()]

chars = sorted(list(set("".join(all_names))))
chars.append("\n")
char2idx = {c:i for i,c in enumerate(chars)}
idx2char = {i:c for i,c in enumerate(chars)}
vocab_size = len(chars)


# -----------------------------
# 3️⃣ LSTM-Modell definieren
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

# Modell initialisieren
model = NameLSTM(vocab_size)
# Modell laden
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()
print("Modell geladen!")

# -----------------------------
# 4️⃣ Name-Generierung
# -----------------------------
def generate_name(model, seed="", max_len=MAX_NAME_LEN, temperature=TEMPERATURE):
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
# 5️⃣ Stilzuweisung (nachträglich)
# -----------------------------

STYLE_RULES = {
    "TECHNO": {
        "chars": "xzqk",
        "suffixes": ["ion", "x", "or"],
        "score": 0,
        "min_length": 6
    },
    "METAL": {
        "chars": "grshvd",
        "suffixes": ["grim", "dark", "core"],
        "score": 0,
        "min_length": 6
    },
    "POP": {
        "chars": "aeiou",
        "suffixes": ["a", "y", "el"],
        "score": 0,
        "min_length": 4
    },
    "AMBIENT": {
        "chars": "aeiouhl",
         "suffixes": ["is", "um", "a", "en"],
         "score": 0,
         "min_length": 7
    },
    "COUNTRY": {
        "chars": "lnrytw",
         "suffixes": ["son", "ley", "ford", "man", "wood", "hill", "way", "ry"],
         "score": 0,
         "min_length": 5
    },
    "COUNTRY_OUTLAW": {
        "chars": "krnd",
         "suffixes": ["outlaw", "dust", "road"],
         "score": 0,
         "min_length": 6
    }




}

def assign_style(name):
    name = name.lower()

    #scores = {}

    scores = {
    "TECHNO": 7,
    "METAL": 3,
    "POP": 4,
    "AMBIENT": 2,
    "COUNTRY": 5,
    "COUNTRY_OUTLAW": 1 
}

    for style, rules in STYLE_RULES.items():
        score = 0
        score += sum(name.count(c) for c in rules["chars"])
        score += sum(2 for s in rules["suffixes"] if name.endswith(s))
        scores[style] = score

    return max(scores, key=scores.get)

    




#-old---------------------------------
#def old_assign_style(name):
#    name_lower = name.lower()
#    if any(c in name_lower for c in "xzqk"):
#        return "TECHNO"
#    elif any(c in name_lower for c in "grshv"):
#        return "METAL"
#    else:
#        return "POP"

# -----------------------------
# 6️⃣ Batch-Generierung
# -----------------------------
def generate_batch(model, num_names=100, max_len=MAX_NAME_LEN, seeds=None, variable_length=False, normalize=True):
    generated = set()
    
    if seeds== "" or seeds== None:
        seeds = ["knx","flux","knoxx","xel","sig","björ","frey","eld","pad","heið"]

    while len(generated) < num_names:
        seed = random.choice(seeds)
        current_max_len = random.randint(6, max_len) if variable_length else max_len

        # Random Temperature
        temp=randrange(6,15)
        temp=temp/10

        name = generate_name(model, seed=seed, max_len=current_max_len,temperature=temp)
        style = assign_style(name)

        # -- Normalize on/off
        name = normalize_name(name, enabled=normalize)


        generated.add((name, style))
    
    return list(generated)

# -----------------------------
# 7️⃣ Batch in CSV speichern
# -----------------------------
def save_batch_to_csv(batch, filename=CSV_OUTPUT, name_width=30):

    # set dir
    newpath = dir_path + "/" + "artists" 
    if not os.path.exists(newpath):
        os.makedirs(newpath)


    now = datetime.now()
    dt_string = now.strftime("%d%m%Y_%H%M%S")

    filename2 = os.path.splitext(filename)[0]
   

    file = newpath + "/" + filename2 + "_" + dt_string +".txt"
    clean_file = newpath + "/" + "clean_" + filename2 + "_" + dt_string +".txt"
    
    
    with open(file, "w", encoding="utf-8") as f:
        f.write(f"{'Name'.ljust(name_width)}Stil\n")
        f.write("-" * (name_width + 10) + "\n")
        for name, style in batch:
            f.write(f"{name.ljust(name_width)}{style}\n")

    f.close()

    with open(clean_file, "w", encoding="utf-8") as f2:
        for name, style in batch:
            f2.write(name)
            f2.write("\n")

    f2.close()

    print("")
    print(f"{len(batch)} Namen in {file} gespeichert!")

#-----------------------------------


#--------------------
# normalize ascii
#--------------------


def normalize_name(name: str, enabled: bool = True) -> str:
    """
    Wandelt Sonderzeichen & Umlaute in ASCII-nahe Buchstaben um.
    Beispiel: 'Tžúrn púmeah' -> 'Tzurn pumeah'
    """
    if not enabled:
        return name

    # 1️⃣ Unicode Normalisierung (NFKD trennt Akzente)
    normalized = unicodedata.normalize("NFKD", name)

    # 2️⃣ Diakritische Zeichen entfernen
    ascii_only = "".join(
        c for c in normalized
        if not unicodedata.combining(c)
    )

    # 3️⃣ Sonderfälle explizit ersetzen
    replacements = {
        "ð": "d", "Ð": "D",
        "þ": "th", "Þ": "Th",
        "ß": "ss",
        "æ": "ae", "Æ": "Ae",
        "œ": "oe", "Œ": "Oe",
        "ł": "l", "Ł": "L",
        "ø": "o", "Ø": "O",
        "đ": "d", "Đ": "D",
        "ž": "z", "Ž": "Z",
        "š": "s", "Š": "S",
        "č": "c", "Č": "C",
        "ć": "c", "Ć": "C"
    }

    for src, tgt in replacements.items():
        ascii_only = ascii_only.replace(src, tgt)

    return ascii_only


# -----------------------------
# 8️⃣ Batch generieren und speichern
# -----------------------------

# Flag zum Aktivieren variabler Namenlängen
VARIABLE_LENGTH_FLAG = True
NORMALIZE_UNICODE = True  # False = Original beibehalten
NORM_FLAG=False

while True:

    os.system("cls")
    
    print("")
    print("Fantasy Artist Names Generator v0.1a")
    print("\n\n")

    # Seed Gui ----------
    SEEDS=input("Seed Word: ")
    if SEEDS=="" or SEEDS==None:
        SEEDS=None

    # Norm Gui ----------
    nor = input("Normalize (Y/N) ")
    if nor=="" or nor==None:
        nor="Y"
    if nor=="y" or nor=="Y" or nor=="j" or nor=="J":
        NORM_FLAG=True
    else:
        NORM_FLAG=False

    # Anzahl Gui -----------
    anz = input("Anzahl Names (<1000) ")
    if anz=="" or anz==None:
       anz=100
    elif anz.isnumeric():
        anz=int(anz)
        if anz<1:
            anz=1
        if anz>1000:
            anz=1000
    else:
       anz=100
      
    # Set names amount
    NUM_NAMES=anz


    print(f"Starte Batch-Generierung von {NUM_NAMES} Namen...")
    batch = generate_batch(model, num_names=NUM_NAMES, max_len=MAX_NAME_LEN, seeds=SEEDS, variable_length=VARIABLE_LENGTH_FLAG, normalize=NORM_FLAG)
    
    save_batch_to_csv(batch)

    print("")
    print("Fertig!")
    print("")
 
    a=input("wait key")


