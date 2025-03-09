import pickle
import os

def load_model():
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

def predict_query(query):
    model, vectorizer = load_model()
    query_vectorized = vectorizer.transform([query])
    prediction = model.predict(query_vectorized)[0]
    if prediction == 1:
        return "Обнаружена SQL-инъекция!"
    else:
        return "Запрос безопасен."

print("✅ Модель успешно загружена!")