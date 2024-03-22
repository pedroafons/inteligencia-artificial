import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

# Carregar os dados
data = pd.read_csv('wine_dataset_henri.csv')

# Separar os dados de entrada (X) e saída (y)
X = data.drop('style', axis=1)
y = data['style']

# Dividir os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar o classificador LinearSVC
clf = LinearSVC(random_state=42)

# Treinar o classificador
clf.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = clf.predict(X_test)

# Avaliar a precisão do modelo
accuracy = accuracy_score(y_test, y_pred)
print("A precisão do modelo é:", accuracy)

# Inicializar o DummyClassifier com a estratégia 'most_frequent'
dummy_clf = DummyClassifier(strategy="most_frequent")

# Treinar o DummyClassifier
dummy_clf.fit(X_train, y_train)

# Fazer previsões usando o DummyClassifier
y_pred_dummy = dummy_clf.predict(X_test)

# Avaliar a precisão do DummyClassifier
accuracy_dummy = accuracy_score(y_test, y_pred_dummy)
print("A precisão do DummyClassifier (most_frequent) é:", accuracy_dummy)
