import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("dataset.csv")

df = pd.DataFrame(data)

X = df[['Venit', 'Datorie', 'Scor_Credit']]
y = df['Aprobare']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)

predictions = knn.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"Precizia modeluluiL: {accuracy * 100} %")
print(f"Raportul detaliat al clasificarii: \n {classification_report(y_test, predictions, zero_division=1)}")

