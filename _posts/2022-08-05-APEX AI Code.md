---
layout: post
title: >
    APEX AI Code 
author: Richard
tags: [AI, Python]
---

# APEX AI CODE

## **Linear Regression: 预测房价数据**
Date: Aug.1, Python

```python
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import pandas as pd

data = pd.read_csv('pre-house.csv')

###
data.head(5)
###

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    data[['面积','房间数','卫生间数','所在楼层','总楼层','地铁数','距离地铁距离',]],
    data['价格'],
    test_size=0.30)

from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train, y_train)

###
linreg.score(X_test, y_test)
###

y_pred = linreg.predict(X_test)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
x, y = pd.Series(y_test, name="label"), pd.Series(y_pred, name="prediction")

sns.regplot(x = x,y = y)

```

<br>
<br>

## **Logistic Regression:商品评价**
Date: Aug.2, Python

```python
import os
import jieba
import numpy as np
import pandas as pd

reviewData = pd.read_csv("online_shopping_10_cats_mini.csv")
reviewData = reviewData[["review", "label"]]

###
reviewData.head(5)
###

from sklearn.model_selection import  train_test_split
X_train,X_test,y_train,y_test = train_test_split(
    reviewData['review'],
    reviewData['label'],
    test_size=0.30)

def chinese_cut(review_features):
    def chinese_word_cut(mytext):
        return " ".join(jieba.cut(mytext))
    if isinstance(review_features, pd.Series):
        cut_review_features = review_features.apply(chinese_word_cut)
        print("中文评论的词与词之间将使用空格分隔。下面展示的是其中5条评论：")
        display(cut_review_features.head(5))

    elif isinstance(review_features, str):
        cut_review_features = chinese_word_cut(review_features)
        print("中文评论的词与词之间将使用空格分隔。下面展示的是中文分词后的评论：\n%s"%cut_review_features)

    else:
        print("输入数据错误！请重新输入!")
        return None
    return cut_review_features

X_train = chinese_cut(X_train)
X_test = chinese_cut(X_test)

from sklearn.feature_extraction.text import  CountVectorizer

vect = CountVectorizer(max_features = 10000)
vect.fit(X_train)
X_train_count = vect.transform(X_train)
X_test_count = vect.transform(X_test)

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train_count, y_train)

logreg.score(X_test_count, y_test)

s = input('评论：')
s_cut = chinese_cut(s)
s_count = vect.transform([s_cut])
result = logreg.predict(s_count)

if result[0] == 0:
    print('\n \n \n回应：\n有问题可以联系客服')
else:
    print('\n \n \n回应：\n谢谢你')

```

<br>
<br>


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

<br>
<br>
<br>
<br>
<br>
<script src="https://giscus.app/client.js"
        data-repo="ZhihchengGao/zhihchenggao.github.io"
        data-repo-id="R_kgDOHw_0jQ"
        data-category="Announcements"
        data-category-id="DIC_kwDOHw_0jc4CQnM5"
        data-mapping="pathname"
        data-strict="0"
        data-reactions-enabled="1"
        data-emit-metadata="0"
        data-input-position="bottom"
        data-theme="preferred_color_scheme"
        data-lang="en"
        crossorigin="anonymous"
        async>
</script>