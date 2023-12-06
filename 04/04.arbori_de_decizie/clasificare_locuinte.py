from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pandas as pd

data = pd.read_csv('dataset.csv')

label_encoder = LabelEncoder()
data['locatie'] = label_encoder.fit_transform(data['locatie'])

X_train, X_test, y_train, y_test = train_test_split(
    data[['numar_camere', 'suprafata_utila', 'locatie', 'an_constructie', 'pret']],
    data['target'],
    test_size=0.2,
    random_state=42
)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

prediction = model.predict(X_test)

accuratete = accuracy_score(y_test, prediction)
conf_matrix = confusion_matrix(y_test, prediction)
classification_rep = classification_report(y_test, prediction)

print(f"Acuratete: {accuratete}")
print(f"Matrice de confuzie:\n{conf_matrix}")
print(f"Raport de clasificare:\n{classification_rep}")

