import numpy as np
import pandas as pd
from keras.datasets import cifar10
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers import Dense
from keras import utils as np_utils
from keras.optimizers import RMSprop

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape)
print(y_train.shape)

classes = np.unique(y_train)
nClasses = len(classes)
print(nClasses)
print(classes)

x_train = x_train/255.0
print(np.min(x_train), np.max(x_train))
print(x_train.shape)
x_train_flat = x_train.reshape(-1, 3072)
feat_cols = [f"pixel {i}" for i in range(x_train_flat.shape[1])]
df_cifar = pd.DataFrame(x_train_flat, columns=feat_cols)
df_cifar['label'] = y_train
print(f"Marime dataframe: {df_cifar.shape}")
print(df_cifar.head())

x_test = x_test/255.0
x_test = x_test.reshape(-1, 32, 32, 3)
x_test_flat = x_test.reshape(-1, 3072)

pca = PCA(0.9)
pca.fit(x_test_flat)
PCA(copy=True, iterated_power='auto', n_components=0.9, random_state=None, svd_solver='auto', tol=0.0, whiten=False)
print(pca.n_components)

train_img_pca = pca.transform(x_train_flat)
test_img_pca = pca.transform(x_test_flat)

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

batch_size = 128
num_classes = 10
epoch = 20

model = Sequential()
model.add(Dense(1024, activation='relu', input_shape=(97, )))
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

print(model.summary())

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])
history = model.fit(train_img_pca, y_train, batch_size=batch_size, epochs=epoch, verbose=1,
                    validation_data=(test_img_pca, y_test))


model = Sequential()
model.add(Dense(1024, activation='relu', input_shape=(3072, )))
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train_flat, y_train, batch_size=batch_size, epochs=epoch, verbose=1,
                    validation_data=(x_test_flat, y_test))
