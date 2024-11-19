import spacy
import pandas as pd
import os
import time
from spacy.training.example import Example
from tqdm import tqdm  # Progres bar

# Aktivácia GPU
spacy.require_gpu()

def evaluate_model(nlp, test_data_path):
    """Vyhodnotenie modelu na testovacích dátach."""
    test_df = pd.read_csv(test_data_path, nrows=3000)
    test_df = test_df.dropna(subset=['Text'])
    test_df['Text'] = test_df['Text'].astype(str)

    correct_predictions = 0
    total_entities = 0

    for _, row in test_df.iterrows():
        doc = nlp(row['Text'])
        predicted_entities = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
        
        # Očakávané entity z dát
        expected_entities = [(start, end, tag) for start, end, tag in zip(row['Col0'], row['Col1'], row['Tag'])]

        total_entities += len(expected_entities)

        # Porovnaj predikované entity s očakávanými tagmi
        for pred_entity in predicted_entities:
            if pred_entity in expected_entities:
                correct_predictions += 1

    accuracy = correct_predictions / total_entities if total_entities > 0 else 0
    return accuracy


def train_spacy_ner(train_data_path, test_data_path, model_output_path, n_iter=10):
    start_time = time.time()
    print("Načítavam tréningové dáta...")

    # Načítaj tréningové dáta s obmedzením na 3000 riadkov
    train_df = pd.read_csv(train_data_path, low_memory=False, nrows=3000)
    train_df = train_df.dropna(subset=['Text'])  # Odstráň riadky bez textu
    train_df['Tag'] = train_df['Tag'].fillna('Unknown')
    train_df['Text'] = train_df['Text'].astype(str)

    # Inicializácia SpaCy modelu
    print("Inicializujem SpaCy model...")
    nlp = spacy.blank("en")
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")

    # Pridanie labelov do NER modelu
    for _, row in train_df.iterrows():
        ner.add_label(row['Tag'])

    # Príprava tréningových dát
    training_data = []
    for _, row in train_df.iterrows():
        text = row['Text']
        entities = [(0, len(text), row['Tag'])]
        training_data.append((text, {"entities": entities}))

    examples = []
    for text, annotations in training_data:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        examples.append(example)

    # Inicializácia optimizéra
    print("Inicializujem optimizér...")
    optimizer = nlp.initialize(lambda: examples)

    # Tréning modelu s progres barom
    print("Začínam tréning modelu...")
    for i in range(n_iter):
        print(f"\nIterácia {i + 1}/{n_iter}")
        losses = {}

        # Progres bar na iterácie
        with tqdm(total=len(examples), desc=f"Iterácia {i + 1}") as pbar:
            for j, example in enumerate(examples):
                nlp.update([example], drop=0.35, sgd=optimizer, losses=losses)
                pbar.update(1)

        print(f"Straty po {i + 1}. iterácii: {losses}")

        # Evaluácia po každej iterácii
        print("Vyhodnocujem model...")
        accuracy = evaluate_model(nlp, test_data_path)
        print(f"Presnosť po {i + 1}. iterácii: {accuracy * 100:.2f}%")

    # Ukončenie tréningu
    print("Tréning modelu dokončený.")
    end_time = time.time()
    print(f"Tréning modelu trval {end_time - start_time:.2f} sekúnd.")

    # Uloženie modelu na disk
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)
    nlp.to_disk(model_output_path)
    print(f"Model bol uložený do {model_output_path}")

    return nlp


def predict_new_data(nlp, new_texts):
    """Predikcia na nových dátach."""
    for text in new_texts:
        doc = nlp(text)
        print(f"Text: {text}")
        for ent in doc.ents:
            print(f"Predikovaná entita: {ent.text} ({ent.label_})")


if __name__ == "__main__":
    # Cesty k tréningovým a testovacím dátam
    train_data_path = os.path.join(os.path.dirname(__file__), '../data/train_data.csv')
    test_data_path = os.path.join(os.path.dirname(__file__), '../data/test_data.csv')

    # Cesta pre uloženie modelu
    model_output_path = os.path.join(os.path.dirname(__file__), '../models/spacy_model')

    # Spustenie tréningu
    nlp = train_spacy_ner(train_data_path, test_data_path, model_output_path)

    # Načítanie modelu z disku
    nlp = spacy.load(model_output_path)

    # Vyhodnotenie modelu
    accuracy = evaluate_model(nlp, test_data_path)
    print(f"Presnosť modelu: {accuracy * 100:.2f}%")

    # Nové texty na predikciu
    new_texts = [
        "Apple is looking at buying U.K. startup for $1 billion",
        "Elon Musk is the CEO of Tesla"
    ]

    # Predikcia na nových textoch
    predict_new_data(nlp, new_texts)
