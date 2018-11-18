# -*- coding: utf-8 -*-
from pathlib import Path
from collections import Counter

import numpy as np

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

DATASET_PATH = "dataset/gray_scaled"

X_DATASET_PATH = Path(DATASET_PATH) / "hiragana.npz"
LABEL_DATASET_PATH = Path(DATASET_PATH) / "label.npz"

categorical_list = list("あいうえおかがきぎくぐけげこごさざしじすずせぜそぞただちぢつづてでとどなにぬねのはばぱひびぴふぶぷへべぺほぼぽまみむめもやゆよらりるれろわゐゑをん")

batch_size = 128
num_classes = len(categorical_list)
epochs = 12

# input image dimensions
img_rows, img_cols = 48, 48

x_data = np.load(X_DATASET_PATH)["arr_0"]
y_data = np.load(LABEL_DATASET_PATH)["arr_0"]

# 重みの算出
label_counter = Counter(y_data)
max_label_count = label_counter.most_common()[0][1]
label_weight = {}
for k,v in label_counter.items():
    label_weight[k] = max_label_count / v
#print(label_counter)
#print(label_weight)

if K.image_data_format() == 'channels_first':
    x_data = x_data.reshape(x_data.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_data = x_data.reshape(x_data.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_data = x_data.astype('float32')
x_data /= 255

print('x_data shape:', x_data.shape)

y_data = keras.utils.to_categorical(y_data, num_classes)

X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.25)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          class_weight=label_weight,
          validation_data=(X_test, y_test),
          shuffle=True)
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
model.save('./models/hiragana_cnn_model.h5')
