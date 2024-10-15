#importing the lib
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

#To Read the File
df=pd.read_csv(r"D:\ML\Heart Disease Model\archive (2)\heart.csv")
print(df)

#Data Preprocessing
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


df['Sex'] = le.fit_transform(df['Sex'])
df['ChestPainType'] = le.fit_transform(df['ChestPainType'])
df['RestingECG'] = le.fit_transform(df['RestingECG'])
df['ExerciseAngina'] = le.fit_transform(df['ExerciseAngina'])
df['ST_Slope'] = le.fit_transform(df['ST_Slope'])
print(df)

#Data Splitting
x=df.drop(columns=['HeartDisease'])
y=df['HeartDisease']
print("Input",x)
print("Output",y)

#Data Training
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=12)

print("DF",df.shape)
print("x_train",x_train.shape)
print("x_test",x_test.shape)
print("y_train",y_train.shape)
print("y_test",y_test.shape)

#train the data
##Naive Bayes Algorithm
from sklearn.naive_bayes import GaussianNB
NB=GaussianNB()
NB.fit(x_train,y_train)
y_pred=NB.predict(x_test)
print("y_pred",y_pred)
print("y_test",y_test)
from sklearn.metrics import accuracy_score
print("GaussianNB Accuracy Score:", accuracy_score(y_pred,y_test))

##Support Vector Machine
from sklearn.svm import SVC
svc=SVC(kernel='linear',random_state=42)
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
print("SVC Accuracy Score :", accuracy_score(y_test,y_pred))

##Logistic Regression
from sklearn.linear_model import LogisticRegression
LR=LogisticRegression()
LR.fit(x_train,y_train)
y_pred_LR=LR.predict(x_test)
print("Logistic Regression Accuracy Score :", accuracy_score(y_pred_LR,y_test))

##Random Forest
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
rfc.fit(x_train,y_train)
y_pred=rfc.predict(x_test)
print("Random Forest Classifier Accuracy Score :",accuracy_score(y_pred,y_test))





