# pasul 1 - incarcarea datelor
import csv
import os

import gensim
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score

data = []
with open('date.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            names = row
        else:
            data.append(row)
        line_count += 1

list_of_input = [data[i][0] for i in range(len(data))]
list_of_output = [data[i][1] for i in range(len(data))]
label_names = list(set(list_of_output))

print(list_of_input[:2])
print(label_names[:2])


# pasul 2 - impartirea date in date de antrenament si test
# reprezentare 1
np.random.seed(5)
no_of_samples = len(list_of_input)
indexes = [i for i in range(no_of_samples)]
trainSample = np.random.choice(indexes, int(0.8 * no_of_samples), replace=False)
testSample = [i for i in indexes if not i in trainSample]

training_input = [list_of_input[i] for i in trainSample]
training_output = [list_of_output[i] for i in trainSample]
test_inputs = [list_of_input[i] for i in testSample]
test_outputs = [list_of_output[i] for i in testSample]

print(training_input[:3])

# pasul 3: extragerea caracteristicilor

vectorizer = CountVectorizer()

train_features = vectorizer.fit_transform(training_input)
test_featrures = vectorizer.transform(test_inputs)

print(f"vocabular: {vectorizer.get_feature_names_out()[:10]}")
print(f"features: {train_features.toarray()[:3][:10]}")


# reprezentarea 2
vectorizer = TfidfVectorizer(max_features=50)


training_features = vectorizer.fit_transform(training_input)
test_features = vectorizer.transform(test_inputs)


print(f"vocabular reprezentare 2: {vectorizer.get_feature_names_out()[:10]}")
print(f"fearures reprezentare 2: {training_features.toarray()[:3]}")


# reprezentarea 3
crtDir = os.getcwd()
modelPath = os.path.join(crtDir, 'GoogleNews-vectors-negative300.bin')

word2vecModel300 = gensim.models.KeyedVectors.load_word2vec_format(modelPath, binary=True)

print(word2vecModel300.most_similar('support'))
print(f"vector for house: {word2vecModel300['house']}")


def feature_calcul(model, data):
    features = []
    phrases = [phrase.split() for phrase in data]
    for phrase in phrases:
        vectors = [model[word] for word in phrase if (len(word) > 2) and (word in model.key_to_index)]
        if len(vectors) == 0:
            result = [0.0] * model.vector_size
        else:
            result = np.sum(vectors, axis=0) / len(vectors)
        features.append(result)
    return features


training_features = feature_calcul(word2vecModel300, training_input)
test_features = feature_calcul(word2vecModel300, test_inputs)


# pasul 4 - antrenare model de invatare nesupervizata (clusteringul)

unsupervisedClassifier = KMeans(n_clusters=2, random_state=0, n_init=10)
unsupervisedClassifier.fit(training_features)


# pasul 5 - testare model

computedTestIndexes = unsupervisedClassifier.predict(test_features)
computedTestOutputs = [label_names[value] for value in computedTestIndexes]

# pasul 6 - calcul metrica de performanta

print(f'acc: {accuracy_score(test_outputs, computedTestOutputs)}')
