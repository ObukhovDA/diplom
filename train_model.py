import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import pickle

# === 1. Генерация данных ===
data = {
    'query': [
        "SELECT * FROM users;",
        "SELECT * FROM users WHERE id=1;",
        "INSERT INTO users (name, age) VALUES ('John', 25);",
        "UPDATE users SET password='1234' WHERE id=1;",
        "' OR 1=1 --",
        "'; DROP TABLE users; --",
        "' UNION SELECT null, null, null --",
        "SELECT password FROM users WHERE username='admin' AND password='' OR '1'='1';",
        "SELECT * FROM accounts WHERE username='' OR 'x'='x';",
        "INSERT INTO accounts (username, password) VALUES ('admin', 'password');"
    ],
    'label': [
        0, 0, 0, 0, 1, 1, 1, 1, 1, 0
    ]
}

# === 2. Создание DataFrame ===
df = pd.DataFrame(data)

# === 3. Векторизация текста ===
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['query'])
y = df['label']

# === 4. Обучение модели ===
model = RandomForestClassifier()
model.fit(X, y)

# === 5. Сохранение модели и векторизатора ===
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("Модель успешно обучена и сохранена!")
