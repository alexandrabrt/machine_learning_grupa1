import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

breast = load_breast_cancer()
breast_data = breast.data
print(breast_data.shape)
breast_labels = breast.target
print(breast_labels.shape)
labels = np.reshape(breast_labels, (569, 1))
final_breast_data = np.concatenate([breast_data, labels], axis=1)
print(final_breast_data.shape)
breast_dataset = pd.DataFrame(final_breast_data)
features = breast.feature_names
print(features)
features_labels = np.append(features, 'label')
breast_dataset.columns = features_labels
print(breast_dataset.head())
breast_dataset['label'].replace(0, 'Benign', inplace=True)
breast_dataset['label'].replace(1, 'Malignant', inplace=True)
print(breast_dataset.tail())


x = breast_dataset.loc[:, features].values
x = StandardScaler().fit_transform(x)
print(x.shape)
print(np.mean(x), np.std(x))
feat_cols = [f"feature {i}" for i in range(x.shape[1])]
normalised_breast = pd.DataFrame(x, columns=feat_cols)
print(normalised_breast)

pca_breast = PCA(n_components=2)
principalComponents_breast = pca_breast.fit_transform(x)
principal_breast_Df = pd.DataFrame(data=principalComponents_breast, columns=['principal component 1', 'principal component 2'])
print(principal_breast_Df.tail())
print(f"Variatia explicata pe fiecare compoenenta principala: {pca_breast.explained_variance_ratio_}")


plt.figure(figsize=(10, 10))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel("Componenta principala 1", fontsize=20)
plt.ylabel('Componenta principala 2', fontsize=20)
plt.title('Analiza componentelor principale', fontsize=20)
targets = ['Benign', 'Malignant']
colors = ['r', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = breast_dataset['label'] == target
    plt.scatter(principal_breast_Df.loc[indicesToKeep, 'principal component 1']
               , principal_breast_Df.loc[indicesToKeep, 'principal component 2'], c = color, s = 50)
plt.legend(targets, prop={'size': 15})
plt.show()

