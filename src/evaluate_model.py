import spacy
import pandas as pd
import os
from tqdm import tqdm  # Progres bar

def evaluate_model(test_data_path, model_path):
    """Vyhodnotenie presnosti modelu na testovacích dátach."""
    # Načítanie modelu
    print("Načítavam model...")
    nlp = spacy.load(model_path)
    
    # Načítaj testovacie dáta
    print("Načítavam testovacie dáta...")
    test_df = pd.read_csv(test_data_path)
    test_df = test_df.dropna(subset=['Text'])  # Odstránenie prázdnych textov
    test_df['Text'] = test_df['Text'].astype(str)

    # Premenné na presnosť
    correct_predictions = 0
    total_entities = 0

    print("Vyhodnocujem...")
    # Progres bar pre iteráciu cez testovacie dáta
    with tqdm(total=len(test_df), desc="Vyhodnocovanie") as pbar:
        for _, row in test_df.iterrows():
            doc = nlp(row['Text'])
            predicted_entities = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
            expected_entity = row['Tag']

            # Vytlačiť predikované entity pre každý riadok
            # print(f"Text: {row['Text']}")
            # print(f"Predikované entity: {predicted_entities}")
            # print(f"Očakávaná entita: {expected_entity}")
            
            # Ak predikcia zodpovedá očakávanej entite, zvyšujeme počet správnych predikcií
            for pred_entity in predicted_entities:
                if pred_entity[2] == expected_entity:
                    correct_predictions += 1
                total_entities += 1

            pbar.update(1)

    # Výpočet presnosti
    accuracy = correct_predictions / total_entities if total_entities > 0 else 0
    print(f"Presnosť modelu: {accuracy * 100:.2f}%")
    return accuracy

if __name__ == "__main__":
    # Cesty k testovacím dátam a modelu
    test_data_path = os.path.join(os.path.dirname(__file__), '../data/custom_test_data.csv')
    model_path = os.path.join(os.path.dirname(__file__), '../models/spacy_model')

    # Spustenie evaluácie
    evaluate_model(test_data_path, model_path)
