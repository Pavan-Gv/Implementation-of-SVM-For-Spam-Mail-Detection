# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1.Import the required library packages.
2. Import the dataset to operate on.
3. Split the dataset into required segments.
4. Predict the required output.
5. Run the programme.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: G Venkata Pavan Kumar
RegisterNumber: 212221240013
*/
import cv2
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("spam.csv",encoding='latin-1')
data.head()
data.info()
data.isnull().sum()
x=data["v1"].values
y=data["v2"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extractiaon.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
![ml 1](https://user-images.githubusercontent.com/94827772/173217467-9faabb90-4d8b-45af-86e7-760e800677f0.png)

![ml 2](https://user-images.githubusercontent.com/94827772/173217466-67e86818-d947-43ea-b0ff-eaaae9f07307.png)

![ml 3](https://user-images.githubusercontent.com/94827772/173217465-388c7afb-4241-49bb-86f0-87d45d2d98ff.png)

![ml 4](https://user-images.githubusercontent.com/94827772/173217462-dfeec9d5-55f1-47e3-b5b7-7241fad5f0c7.png)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
