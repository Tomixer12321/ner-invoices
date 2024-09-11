import spacy
import pandas as pd
import os
import time
from spacy.training.example import Example

def train_spacy_ner(train_data_path, model_output_path, n_iter=10):
    start_time = time.time()
    print("Načítavam dáta...")
    train_df = pd.read_csv(train_data_path, low_memory=False)

    # Prepracovávam dáta
    print("Prepracovávam dáta...")
    train_df = train_df.dropna(subset=['Text'])
    train_df['Tag'] = train_df['Tag'].fillna('Unknown')
    train_df['Text'] = train_df['Text'].astype(str)

    # Príprava dát pre SpaCy (potrebný formát je tuple (text, entity))
    training_data = []
    for _, row in train_df.iterrows():
        text = row['Text']
        entities = []
        # Tu predpokladám, že Tag obsahuje pozíciu alebo entitu, uprav to podľa štruktúry dát
        entities.append((0, len(text), row['Tag']))  # Pridaj pozíciu od 0 po dĺžku textu
        training_data.append((text, {"entities": entities})) 

    # Inicializácia prázdneho NER modelu v SpaCy
    print("Inicializujem SpaCy model...")
    nlp = spacy.blank("en")  # Prázdny model pre angličtinu
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")

    # Pridávame labely do NER
    for _, annotations in training_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # Tréning modelu
    print("Začínam tréning modelu...") 
    optimizer = nlp.begin_training()
    for i in range(n_iter):
        print(f"Iterácia {i + 1}/{n_iter}")
        losses = {}
        for text, annotations in training_data:
            example = Example.from_dict(nlp.make_doc(text), annotations)
            nlp.update([example], drop=0.35, sgd=optimizer, losses=losses)
        print(f"Straty po {i + 1}. iterácii: {losses}")

     

    print("Tréning modelu dokončený.")
    end_time = time.time()
    print(f"Tréning modelu trval {end_time - start_time:.2f} sekúnd.")

    # Uloženie modelu
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)
    nlp.to_disk(model_output_path)
    print(f"Model bol uložený do {model_output_path}")


if __name__ == "__main__":
    # Cesta k tréningovým dátam
    train_data_path = os.path.join(os.path.dirname(__file__), '../data/train_data.csv')
    # Cesta k priečinku, kde sa uloží model
    model_output_path = os.path.join(os.path.dirname(__file__), '../models/spacy_model')

    train_spacy_ner(train_data_path, model_output_path)
