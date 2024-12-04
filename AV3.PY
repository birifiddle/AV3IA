import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Carregando a base de dados
data = pd.read_csv('db.csv')

# 2. Tratamento de dados
data = data.dropna()  # Remove valores nulos

label_encoder = LabelEncoder()
for col in data.select_dtypes(include=['object']).columns:
    data[col] = label_encoder.fit_transform(data[col])

# 3. Separando variáveis independentes e a variável target
X = data.drop('Status', axis=1)  # Variáveis independentes
y = data['Status']  # Variável target

# 4. Dividindo os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 5. Normalizando os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. Treinando três modelos

# Modelo 1: Árvore de Decisão
model_dt = DecisionTreeClassifier(random_state=42)
model_dt.fit(X_train, y_train)
y_pred_dt = model_dt.predict(X_test)

# Modelo 2: Naive Bayes
model_nb = GaussianNB()
model_nb.fit(X_train, y_train)
y_pred_nb = model_nb.predict(X_test)

# Modelo 3: Rede Neural
model_nn = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)
model_nn.fit(X_train, y_train)
y_pred_nn = model_nn.predict(X_test)

# 7. Avaliação dos modelos individuais
def avaliar_modelo(y_true, y_pred, nome_modelo):
    print(f"\n{nome_modelo}")
    print("Acurácia:", accuracy_score(y_true, y_pred))
    print("Relatório de Classificação:\n", classification_report(y_true, y_pred))
    print("Matriz de Confusão:\n", confusion_matrix(y_true, y_pred))

# Avaliando cada modelo individualmente
avaliar_modelo(y_test, y_pred_dt, "Árvore de Decisão")
avaliar_modelo(y_test, y_pred_nb, "Naive Bayes")
avaliar_modelo(y_test, y_pred_nn, "Rede Neural")

# 8. Criando o comitê de classificadores (Voting Classifier)
comite = VotingClassifier(estimators=[
    ('decision_tree', model_dt),
    ('naive_bayes', model_nb),
    ('neural_network', model_nn)
], voting='hard')

# Treinando o comitê
comite.fit(X_train, y_train)
y_pred_comite = comite.predict(X_test)

# 9. Avaliação do comitê de classificadores
avaliar_modelo(y_test, y_pred_comite, "Comitê de Classificadores")
