readme.md

```markdown
# AI Artist Name Generator

This project generates **unique, previously unknown artist names** using a
**character-level LSTM neural network**.
The system is designed to focus purely on **name creativity**, while
**music style classification is applied afterwards** using a flexible,
rule-based scoring system.

The generator supports:
- Unicode characters and special letters
- Optional normalization to ASCII
- Batch generation
- Adjustable name length
- Creative sampling using temperature
- Post-processing style assignment (Techno, Metal, Pop, Ambient, Country, etc.)

---

## Features

- 🔤 Character-level LSTM name generator
- 🎛 Adjustable sampling temperature (supports random temperature ranges)
- 🧠 No style bias during training (style added later)
- 🎶 Rule-based post style classification (extensible)
- 🌍 Unicode & special character support
- 🔁 Optional ASCII normalization (flag-controlled)
- 📦 Batch generation and CSV/TXT export
- ⚙️ Configurable name length

---

## Project Structure

```

.
├── train.py                 # Training script for the LSTM model
├── artistgen.py             # Batch name generation script
├── model.pt                 # Trained model (not included in repo)
├── style_rules.py           # Style scoring rules (optional)
├── README.md
├── requirements.txt

````

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/artist-name-generator.git
cd artist-name-generator
````

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # Linux / macOS
venv\Scripts\activate     # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Training the Model

Train the LSTM using your own dataset of names:

```bash
python train.py
```

The trained model will be saved as a `.pt` file and can later be loaded for
batch generation.

---

## Generating Artist Names

Run the standalone batch generator:

```bash
python artistgen.py
```

Typical options inside the script:

* Number of names to generate
* Minimum / maximum name length
* Random temperature range (e.g. 0.6 – 1.5)
* Unicode normalization ON/OFF
* Output file format

---

## Sampling Temperature

The **temperature parameter** controls creativity:

| Temperature | Effect                   |
| ----------- | ------------------------ |
| 0.4 – 0.6   | Conservative, safe names |
| 0.7 – 0.9   | Balanced                 |
| 1.0 – 1.2   | Creative                 |
| 1.3 – 1.5   | Experimental             |

The project supports **random temperature per name** for maximum diversity.

---

## Style Assignment (Post-Processing)

Music styles are assigned **after name generation** using a scoring system
based on:

* Character frequency
* Suffixes
* Name length
* Optional penalties or bonuses

Supported styles include:

* Techno
* Metal
* Pop
* Ambient
* Country

New styles can be added without retraining the model.

---

## Unicode Normalization

Special characters can be converted to ASCII equivalents:

```
Tžúrn púmeah → Tzurn pumeah
```

This behavior is controlled via a flag and can be disabled at any time.

---

## Design Philosophy

* Creativity first
* No dataset bias from music genres
* Transparent and deterministic post-classification
* Easily extensible without retraining

---

## License

MIT License

---

## Disclaimer

This project only **generates names**.
Name existence checks or trademark validation are intentionally out of scope.

````

---

## 📦 `requirements.txt`

```txt
torch>=1.12
````

👉 **That’s all you need**

* No extra NLP libraries
* No tokenizers
* No external datasets
* Fully offline capable

(Optional, only if you later add CLI or configs:)

```txt
numpy
```

---


