import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import csv

print("Loading train data")

news_train = list(csv.reader(open('news_train.txt', 'rt', encoding="utf8"), delimiter='\t'))
news_train_data = []
news_train_data_target = []

for x in news_train:
    news_train_data.append(x[2])
    news_train_data_target.append(x[0])

print("training")

text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),
])
text_clf = text_clf.fit(news_train_data, news_train_data_target)

print("Loading test data...")

news_test = list(csv.reader(open('news_test.txt', 'rt', encoding="utf8"), delimiter='\t'))
news_test_data = []
news_test_data_target = []
for x in news_test:
    news_test_data.append(x[1])
   
print("testing")

docs_test = news_test_data
predicted = text_clf.predict(docs_test)
print("outputing to final_output.txt")
fh = open("final_output.txt", 'w')
for item in predicted:
  fh.write("%s\n" % item)
