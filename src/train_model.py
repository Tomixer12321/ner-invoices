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
    train_df = train_df.dropna(subset=['Text'])  # Remove rows with missing text
    train_df['Tag'] = train_df['Tag'].fillna('Unknown')  # Fill missing tags with 'Unknown'
    train_df['Text'] = train_df['Text'].astype(str)  # Ensure 'Text' column is of string type

    # **Limit the data to 100 rows** 
    train_df = train_df.head(100)  # This line limits the data to the first 100 rows

    # Inicializácia prázdneho NER modelu v SpaCy
    print("Inicializujem SpaCy model...")
    nlp = spacy.blank("en")  # Prázdny model pre angličtinu
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")

    # Pridávame labely do NER
    for _, row in train_df.iterrows():
        ner.add_label(row['Tag'])

    # Príprava dát pre SpaCy (potrebný formát je tuple (text, entity))
    training_data = []
    for _, row in train_df.iterrows():
        text = row['Text']
        entities = []
        entities.append((0, len(text), row['Tag']))  # Pridaj pozíciu od 0 po dĺžku textu
        training_data.append((text, {"entities": entities})) 

    # Create a list of Example objects
    examples = []
    for text, annotations in training_data:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        examples.append(example)

    # **Initialization of the pipeline before training**
    print("Inicializujem optimizér...")
    optimizer = nlp.initialize(lambda: examples)  # Initialize model parameters with Example objects

    # Tréning modelu
    print("Začínam tréning modelu...") 
    for i in range(n_iter):
        print(f"Iterácia {i + 1}/{n_iter}")
        losses = {}
        for j, example in enumerate(examples):
            nlp.update([example], drop=0.35, sgd=optimizer, losses=losses)
            
            # Print the message after every 1000 examples
            if j % 1000 == 0:
                print(f"Spracované {j} príkladov")
                
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
    train_data_path = os.path.join(os.path.dirname(__file__), '../data/train_data.csv')
    model_output_path = os.path.join(os.path.dirname(__file__), '../models/spacy_model')

    train_spacy_ner(train_data_path, model_output_path)
