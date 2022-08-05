---
layout: post
title: >
    APEX AI Code 
author: Richard
tags: [Page]
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

```Python
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