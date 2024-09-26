import spacy
import pandas as pd
import os

def evaluate_model(test_data_path, model_path):
    nlp = spacy.load(model_path)
    
    # Načítaj testovacie dáta a obmedz na prvých 100 riadkov
    test_df = pd.read_csv(test_data_path)
    test_df = test_df.dropna(subset=['Text'])
    test_df['Text'] = test_df['Text'].astype(str)

    # Obmedzenie na 100 riadkov
    test_df = test_df.head(100)

    correct_predictions = 0
    total_entities = 0

    for _, row in test_df.iterrows():
        doc = nlp(row['Text'])
        predicted_entities = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
        print(f"Predicted entities for '{row['Text']}': {predicted_entities}")
        
        # Porovnanie predikovaných entít s pravými entitami
        total_entities += 1
        if predicted_entities and predicted_entities[0][2] == row['Tag']:
            correct_predictions += 1

    print(f"Accuracy: {correct_predictions / total_entities:.2f}")

if __name__ == "__main__":
    test_data_path = os.path.join(os.path.dirname(__file__), '../data/test_data.csv')
    model_path = os.path.join(os.path.dirname(__file__), '../models/spacy_model')
    evaluate_model(test_data_path, model_path)
