import pandas as pd
import joblib
from sklearn.metrics import classification_report
import os

def evaluate_model(test_data_path, model_path, vectorizer_path):
    test_df = pd.read_csv(test_data_path)
    X_test = test_df['Text']
    y_test = test_df['Tag']

    # Načítanie modelu a vectorizéra
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    
    # Transformácia testovacích textov
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Predikcia
    y_pred = model.predict(X_test_tfidf)

    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    # Dynamické cesty k dátam a modelu
    test_data_path = os.path.join(os.path.dirname(__file__), '../data/test_data.csv')
    model_path = os.path.join(os.path.dirname(__file__), '../models/model.pkl')
    vectorizer_path = os.path.join(os.path.dirname(__file__), '../models/vectorizer.pkl')

    evaluate_model(test_data_path, model_path, vectorizer_path)
