import string

import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('spam_ham_dataset.csv')

# nltk.download('stopwords')

# print(df)



data = df.where((pd.notnull(df)), '')

# print(data.head(11))

# print(data.info())

# print(data.shape)

data.loc[data['label'] == 'spam', 'label'] = 0
data.loc[data['label'] == 'ham', 'label'] = 1
X = data['text']
Y = data['label']


# print(X)
# print("Spam\n")
# print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state= 3)

# print(X.shape)
# print(X_train.shape)
# print(X_test.shape)

# print(Y.shape)
# print(Y_train.shape)
# print(Y_test.shape)

feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)

X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

# print(X_train)
# print(X_train_features)

model = LogisticRegression()

model.fit(X_train_features, Y_train)

prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)

print(f"Accuracy on training data: {accuracy_on_training_data}")


prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)

print(f"Accuracy on test data: {accuracy_on_test_data}")


input_your_mail = [input("Input Your Email:").strip()]


input_data_features = feature_extraction.transform(input_your_mail)

prediction = model.predict(input_data_features)

if prediction[0] == 1:
    print("Ham Mail")
else:
    print("Spam Mail")