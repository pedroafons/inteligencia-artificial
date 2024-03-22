import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

# Exemplo de dados de treinamento
data = {
    'texto': ["Este produto é incrível!", "Não gostei muito deste produto.", "O produto não atendeu às minhas expectativas.", "Uma belíssima porcaria!"],
    'sentimento': ['positivo', 'negativo', 'negativo', 'negativo']
}

df = pd.DataFrame(data)

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(df['texto'], df['sentimento'], test_size=0.2, random_state=42)

# Vetorização dos dados de texto
vectorizer = TfidfVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Treinamento do modelo LinearSVC
model = LinearSVC()
model.fit(X_train_vect, y_train)

# Avaliação do modelo
predictions = model.predict(X_test_vect)
print(classification_report(y_test, predictions))

# Exemplo de como classificar uma nova avaliação
nova_avaliacao = ["A qualidade deste produto é decepcionante."]
nova_avaliacao_vect = vectorizer.transform(nova_avaliacao)
sentimento = model.predict(nova_avaliacao_vect)[0]
print(f"Sentimento da nova avaliação: {sentimento}")
