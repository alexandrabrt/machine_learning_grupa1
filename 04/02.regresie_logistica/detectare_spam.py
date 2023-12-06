import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


data = pd.read_csv('dataset_spam_2.csv')

train_data, test_data, train_labels, test_labels = train_test_split(data['Text'], data['Label'],
                                                                    test_size=0.2, random_state=42)

vectorized = TfidfVectorizer(max_features=5000)
X_train = vectorized.fit_transform(train_data)
X_test = vectorized.transform(test_data)

model = LogisticRegression()
model.fit(X_train, train_labels)

predictions = model.predict(X_test)

acuratetea = accuracy_score(test_labels, predictions)
conf_matrix = confusion_matrix(test_labels, predictions)
raport_clasificare = classification_report(test_labels, predictions)

print(f"Acuratete: {acuratetea}")
print(f"Confusion Matrix: \n{conf_matrix}")
print(f"Raport de clasificare:\n{raport_clasificare}")



