import numpy
import urllib
import scipy.optimize
import random
from collections import defaultdict
import string
from sklearn import linear_model

def parseData(fname):
  with open(fname, encoding='utf-8') as fh:
    for line in fh:
      yield eval(line)

### Just the first 5000 reviews

data = list(parseData("beer_50000.json"))[:5000]

### How many unique words are there?

wordCount = defaultdict(int)
for d in data:
  for w in d['review/text'].split():
    wordCount[w] += 1

### Ignore capitalization and remove punctuation

wordCount = defaultdict(int)
punctuation = set(string.punctuation)
for d in data:
  r = ''.join([c for c in d['review/text'].lower() if not c in punctuation])
  for w in r.split():
    wordCount[w] += 1


### Just take the most popular words...

wordCount = defaultdict(int)
punctuation = set(string.punctuation)
for d in data:
  r = ''.join([c for c in d['review/text'].lower() if not c in punctuation])
  for w in r.split():
    wordCount[w] += 1

counts = [(wordCount[w], w) for w in wordCount]
counts.sort()
counts.reverse()

words = [x[1] for x in counts[:1000]]

### Sentiment analysis

wordId = dict(zip(words, range(len(words))))
wordSet = set(words)

def feature(datum):
  feat = [0]*len(words)
  r = ''.join([c for c in datum['review/text'].lower() if not c in punctuation])
  for w in r.split():
    if w in words:
      feat[wordId[w]] += 1
  feat.append(1) #offset
  return feat

X = [feature(d) for d in data]
y = [d['review/overall'] for d in data]

#No regularization
theta,residuals,rank,s = numpy.linalg.lstsq(X, y)

print(residuals)

#With regularization
#clf = linear_model.Ridge(1.0, fit_intercept=False)
#clf.fit(X, y)
#theta = clf.coef_
#predictions = clf.predict(X)
