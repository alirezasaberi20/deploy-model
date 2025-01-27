import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import BernoulliNB
import joblib
import sklearn

df = pd.read_csv('/kaggle/input/sms-spam-collection-dataset/spam.csv' ,encoding='ISO-8859-1')
df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'], axis=1, inplace=True)
df.columns = ['label', 'Text']

X = df['Text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=20)

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

classifier = BernoulliNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

model_ = joblib.load('/kaggle/working/classifier.joblib')
vectori = joblib.load('/kaggle/working/vectorizer.joblib')