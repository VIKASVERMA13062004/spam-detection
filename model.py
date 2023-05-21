import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle

df=pd.read_csv('emails.csv')
df.columns.str.match("Unnamed")
df=df.loc[:,~df.columns.str.match("Unnamed")]
df=df.dropna(axis=0)

X=df['text']
y=df['spam']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)


clf=Pipeline([
    ('vectorizer',CountVectorizer()),
    ('nb',MultinomialNB())
])

clf.fit(X_train,y_train)

pickle.dump(clf, open('model.pkl','wb'))