import nltk
import re
import string
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import word_tokenize, sent_tokenize
from wordcloud import WordCloud, STOPWORDS


nltk.download('stopwords')

Dataset1 = pd.read_csv("D:/news.csv")

Dataset1["Article"] = Dataset1["title"] + Dataset1["text"]
Dataset1.sample(frac = 1) 


Dataset1 = Dataset1.loc[:,['Article','label']]
Dataset1 = Dataset1.dropna()

Dataset2true = pd.read_csv("D:/True.csv")
Dataset2fake = pd.read_csv("D:/Fake.csv")

Dataset2true['label']= 'REAL'
Dataset2fake['label']= 'FAKE'

Dataset2 = pd.concat([Dataset2true, Dataset2fake])
Dataset2["Article"] = Dataset2["title"] + Dataset2["text"]
Dataset2.sample(frac = 1) #Shuffle 100%
Dataset2 = Dataset2.loc[:,['Article','label']]

Dataset3 = pd.read_csv("D:/data.csv")

Dataset3["Article"] = Dataset3["Headline"] + Dataset3["Body"]
Dataset3["label"] = Dataset3["Label"]
Dataset3.sample(frac = 1) 

Dataset3.label[Dataset3.label == 1 ] = 'REAL'
Dataset3.label[Dataset3.label == 0 ] = 'FAKE'

Dataset3 = Dataset3.loc[:,['Article','label']]
Dataset3 = Dataset3.dropna()

allset = [Dataset1, Dataset2, Dataset3]
news = pd.concat(allset)

x_train,x_test,y_train,y_test = train_test_split(news['Article'], news['label'], test_size=0.2)


y_train=y_train.astype('str')
y_test=y_test.astype('str')

pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', LogisticRegression())])


Logisticmodel = pipe.fit(x_train, y_train)
prediction = Logisticmodel.predict(x_test)

print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))
Logisticmodel_accuracy = round(accuracy_score(y_test, prediction)*100,2)

print(classification_report(y_test, prediction))

print(confusion_matrix(y_test, prediction))


with open('model.pickle', 'wb') as handle:
    pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
