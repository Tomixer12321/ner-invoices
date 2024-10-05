import spacy
import pandas as pd
import os
import time
import torch
from spacy.training.example import Example

# Skontroluj, či je GPU dostupné cez PyTorch
print("GPU dostupné:", torch.cuda.is_available())

# Požiadavka na SpaCy, aby použil GPU, ak je dostupné
if not spacy.require_gpu():
    print("Nepodarilo sa inicializovať GPU, tréning prebehne na CPU.")

def train_spacy_ner(train_data_path, model_output_path, n_iter=10):
    start_time = time.time()
    print("Načítavam dáta...")
    
    # Načítaj tréningové dáta z CSV súboru
    train_df = pd.read_csv(train_data_path, low_memory=False)

    # Prepracuj dáta - odstráň prázdne riadky a uprav typy dát
    print("Prepracovávam dáta...")
    train_df = train_df.dropna(subset=['Text'])  # Odstráň riadky bez textu
    train_df['Tag'] = train_df['Tag'].fillna('Unknown')  # Prázdne tagy vyplň 'Unknown'
    train_df['Text'] = train_df['Text'].astype(str)  # Zaisti, aby bol 'Text' vo formáte stringu

    # **Limit dát na prvých 100 riadkov**
    train_df = train_df.head(100)

    # Inicializácia prázdneho NER modelu pre SpaCy
    print("Inicializujem SpaCy model...")
    nlp = spacy.blank("en")  # Prázdny model pre angličtinu
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")

    # Pridávaj labely (značky) do NER modelu
    for _, row in train_df.iterrows():
        ner.add_label(row['Tag'])

    # Príprava dát pre SpaCy (formát: tuple (text, entity))
    training_data = []
    for _, row in train_df.iterrows():
        text = row['Text']
        entities = [(0, len(text), row['Tag'])]  # Definuj entitu od začiatku do konca textu
        training_data.append((text, {"entities": entities}))

    # Vytvor list Example objektov pre tréning
    examples = []
    for text, annotations in training_data:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        examples.append(example)

    # Inicializácia optimizéra pred tréningom
    print("Inicializujem optimizér...")
    optimizer = nlp.initialize(lambda: examples)

    # Spusti tréning modelu
    print("Začínam tréning modelu...")
    for i in range(n_iter):
        print(f"Iterácia {i + 1}/{n_iter}")
        losses = {}
        for j, example in enumerate(examples):
            nlp.update([example], drop=0.35, sgd=optimizer, losses=losses)
            
            # Správa po každých 1000 príkladoch
            if j % 1000 == 0:
                print(f"Spracovaných {j} príkladov")

        print(f"Straty po {i + 1}. iterácii: {losses}")

    # Ukončenie tréningu a výpis času trvania
    print("Tréning modelu dokončený.")
    end_time = time.time()
    print(f"Tréning modelu trval {end_time - start_time:.2f} sekúnd.")

    # Uloženie natrenovaného modelu na disk
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)
    nlp.to_disk(model_output_path)
    print(f"Model bol uložený do {model_output_path}")

if __name__ == "__main__":
    # Cesta k tréningovým dátam
    train_data_path = os.path.join(os.path.dirname(__file__), '../data/train_data.csv')

    # Cesta pre uloženie modelu
    model_output_path = os.path.join(os.path.dirname(__file__), '../models/spacy_model')

    # Spustenie tréningu
    train_spacy_ner(train_data_path, model_output_path)
