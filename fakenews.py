#FAKE NEWS DETECTION

import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import nltk
nltk.download('stopwords')
#print(stopwords.words('english')) ---workssss

news_dataset=pd.read_csv("train_old.csv")
print(news_dataset.shape)
print(news_dataset.head())

#counting number of missing columns
print(news_dataset.isnull().sum())
news_dataset=news_dataset.fillna(" ")

#separating the class(either real or fake) and the rest of the table
X=news_dataset.drop(columns='class',axis=1)
Y=news_dataset['class']
print(X)

news_dataset['content']=news_dataset['title']+news_dataset['subject']
print(news_dataset['content'])


port_stem=PorterStemmer()
def stemming(content):
    stemmed_content=re.sub('[^a-zA-z]',' ',content)
    stemmed_content=stemmed_content.lower()
    stemmed_content=stemmed_content.split()
    stemmed_content=[port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content=' '.join(stemmed_content)
    return stemmed_content

news_dataset['content']=news_dataset['content'].apply(stemming)
#print(news_dataset['content'])

X,Y=news_dataset['content'].values,news_dataset['class'].value


vectorizer=TfidVectorizer()
vectorizer.fit(X)
X=vectorizer.transformer(X)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)
model=LogisticRegression()
model.fit(X_train,Y_train)

#accuracy score on the training data
X_train_prediction=model.predict(X_train)
accuracy=accuracy_score(X_train_prediction,Y_test)
print("ACCURACY OF THE MODEL:",accuracy)

#PREDICTION SYSTEM
X_new=X_test[0]
prediction=model.predict(X_new)
print(prediction)
if (prediction[0]==1):
    print('FAKE!')
else:
    print('REAL :-)')

