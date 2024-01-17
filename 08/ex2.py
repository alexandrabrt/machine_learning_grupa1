import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.src.datasets import cifar10
from sklearn.decomposition import PCA
import seaborn as sns

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print("Training data shape: ", x_train.shape)
print("Testing data shape: ", x_test.shape)

print(y_train.shape, y_test.shape)
classes = np.unique(y_train)
nClasses = len(classes)
print('Nr total de output', nClasses)
print('Output classes', classes)
plt.figure(figsize=[5, 5])
labels_dict = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}

plt.subplot(121)
curr_img = np.reshape(x_train[0], (32, 32, 3))
plt.imshow(curr_img)
print(plt.title("(Label: " + str(labels_dict[y_train[0][0]] + ")")))

plt.subplot(122)
curr_img = np.reshape(x_test[0], (32, 32, 3))
plt.imshow(curr_img)
print(plt.title(f"(Label: {labels_dict[y_test[0][0]]})"))
plt.show()

print(np.min(x_train), np.max(x_train))
x_train = x_train/255.0
print(np.min(x_train), np.max(x_train))
print(x_train.shape)
x_train_flat = x_train.reshape(-1, 3072)
feat_cols = [f"pixel {i}" for i in range(x_train_flat.shape[1])]
df_cifar = pd.DataFrame(x_train_flat, columns=feat_cols)
df_cifar['label'] = y_train
print(f"Marime dataframe: {df_cifar.shape}")
print(df_cifar.head())

pca_cifar = PCA(n_components=2)
principalComponents_cifar = pca_cifar.fit_transform(df_cifar.iloc[:, :-1])
principal_cifar_Df = pd.DataFrame(data=principalComponents_cifar, columns=['principal component 1', 'principal component 2'])
principal_cifar_Df['y'] = y_train
print(principal_cifar_Df.head())
print(f"Variatia pe fiecare componenta: {pca_cifar.explained_variance_ratio_}")

plt.figure(figsize=(16, 10))
sns.scatterplot(x='principal component 1', y='principal component 2',
                hue='y',
                palette=sns.color_palette('hls', 10),
                data=principal_cifar_Df,
                legend='full',
                alpha=0.3)
plt.show()
