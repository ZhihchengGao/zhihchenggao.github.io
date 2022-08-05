---
layout: post
title: >
    Handwriting classification
author: Richard
tags: [AI, Python]
---

# Handwriting classification
## **Neuron Network**
Date: Aug.5, Python

<br>
<br>

```python

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

###
train_images.shape
train_labels.shape
###

def mnist_show(images, labels, num):
    class_names=['0','1','2','3','4','5','6', '7','8','9']
    if num >= len(labels):
        print("只有%s张图片，请输入0到%s之间的图片编号。"%(str(len(labels)), str(len(labels)-1)))
    else:
        image = images[num, :, :].reshape([28, 28])
        lable = labels[num]
        plt.title("Label: " + class_names[lable])
        sns.heatmap(image, cmap = 'gray')

mnist_show(train_images, train_labels, 1000)

train_images = train_images.reshape([-1,28,28,1])[:10000,:,:,:]
train_images = train_images/255.0
train_labels = train_labels[:10000]
test_images = test_images.reshape([-1,28,28,1])
test_images = test_images/255.0

##
train_images.shape
train_labels.shape
##

model = Sequential()
model.add(Conv2D(filters=4, kernel_size=(5, 5), strides=(1, 1), \
input_shape=(28, 28, 1), padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(filters=4, kernel_size=(3, 3), strides=(1, 1), \
padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(filters=4, kernel_size=(3, 3), strides=(1, 1), \
padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

model.summary()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images,
          train_labels,
          epochs = 20,
          validation_data = (test_images, test_labels))img = Image.open('../input/handwritten-digit-test-data/digits_test/7.png')


relm = img.resize((28, 28))
nm_arr = np.array(relm.convert('L'))
plt.imshow(nm_arr, 'gray')
pic_scale = nm_arr.reshape(1, 28, 28, 1)/255
predictions = model.predict(pic_scale)
idx = np.argmax(predictions, axis=1)
print(idx)

```
