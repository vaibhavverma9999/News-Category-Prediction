import pickle

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

n1 = LabelEncoder()
df = pd.read_csv('bbc-text.csv')

X = df['text']
X = np.array(X)
Y = df['category']
Y = np.array(Y)

tfidf = TfidfVectorizer(lowercase=False, analyzer='word', stop_words='english', ngram_range=(1,3), use_idf=True)
X1 = tfidf.fit_transform(X)
n1 = LabelEncoder()

Y1 = n1.fit_transform(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X1, Y1, test_size=0.3, random_state=0)

lr = LogisticRegression(solver='lbfgs', multi_class='multinomial')
lr.fit(X_train, Y_train)
test = lr.predict(X_test)
acc = 0
l = np.size(test, 0)
for i in range(l):
    acc = acc + abs(test[i]-Y_test[i])
print("Accuracy using Logistic Regression is: %f"%((1-acc/l)*100))

save_tdidf = open("tdidf.pkl", "wb")
pickle.dump(tfidf, save_tdidf)
save_tdidf.close()

save_tdidf = open("tdidf.pkl", "rb")
tdidf = pickle.load(save_tdidf)
save_tdidf.close()

save_classifier = open("logistic.pickle","wb")
pickle.dump(lr, save_classifier)
save_classifier.close()

classifier_f = open("logistic.pickle", "rb")
clf2 = pickle.load(classifier_f)
classifier_f.close()

t = "Congress leader Nana Patole has been elected as the Speaker of Maharashtra Legislative Assembly after Bharatiya Janata Party (BJP) candidate Kisan Kathore withdrew his nomination on Sunday."
t = np.array(t).reshape(-1,1)
t = tfidf.transform(t[0])

print(n1.inverse_transform(clf2.predict(t)))