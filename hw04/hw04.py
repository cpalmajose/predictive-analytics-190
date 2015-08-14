'''
Created on May 26, 2015

@author: Chris A
'''
import numpy
import string
import sys
from math import log10
from collections import defaultdict
from sklearn import linear_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

fname = "beer_50000.json"

def parseData(fname):
    with open(fname, encoding='utf-8') as fh:
        for line in fh:
            yield eval(line)

# Unigram
def unigram(datam):
    result = defaultdict(int)
    p = set(string.punctuation)
    for d in datam:
        r = ''.join([c for c in d['review/text'].lower() if not c in p])
        for w in r.split():
            result[w] += 1
            
    return result

# Bigram
def bigram(datam):
    result = defaultdict(int)
    p = set(string.punctuation)
    for d in datam:
        r = ''.join([c for c in d['review/text'].lower() if not c in p])
        r = r.split()
        for i in range(len(r) - 1):
            v = ''.join([r[i], ' ', r[i+1]])
            result[v] += 1
            
    return result
            

def findMax(data, prev_last_keys):
    m = -1000000    
    retkey = None
    
    for key in data.keys():
        if key in prev_last_keys:
            continue
        
        check = data[key]
        if check > m:
            retkey = key
            m = check
            
    return retkey

def findMin(data, prev_last_keys):
    m = 1000000   
    retkey = None
    
    for key in data.keys():
        if key in prev_last_keys:
            continue
        
        check = data[key]
        if check < m:
            retkey = key
            m = check
            
    return retkey

def feature(datum, words, wordIds):
    feat = [0]*len(words)
    l_bg = bigram([datum])
    for item in l_bg:
        if item in words:
            feat[wordIds[item]] += 1
    feat.append(1)
    return feat

def feature2(datum, words, wordIds):
    feat = [0]*len(words)
    l_bg = bigram([datum])
    l_ug = unigram([datum])
    l_ug_bg = list(l_bg.keys()) + list(l_ug.keys())
    
    for item in l_ug_bg:
        if item in words:
            feat[wordIds[item]] += 1
    feat.append(1)
    return feat

# TASKS
def t1(data):
    bg = bigram(data)
    
    print("5 most frequent bi-grams:")
    prev_keys = []
    for i in range(5):
        r = findMax(bg, prev_keys)
        if r:
            print(r + ": " + str(bg[r]))
        else:
            print("t1: findmax err")
            sys.exit()
            
        prev_keys.append(r)
        
    bg_counts = [(bg[w], w) for w in bg]
    bg_counts.sort()
    bg_counts.reverse()
    
    print("-------------\ntask 2: residuals")
    bg_words = [x[1] for x in bg_counts[:1000]] 
    bg_wordId = dict(zip(bg_words, range(len(bg_words))))
    
    X = [feature(d, bg_words, bg_wordId) for d in data]
    y = [d['review/overall'] for d in data]
    
    theta, residuals, rank, s = numpy.linalg.lstsq(X,y)
    print( "Bigram residuals: " + str(residuals))
    
    '''
    clf = linear_model.Ridge(1.0, fit_intercept=False)
    clf.fit(X,y)
    #theta = clf.coef_
    pred = clf.predict(X)
    print(pred)
    '''
    
    
    print("-------------\ntask 3: residuals")
    
    # unigram
    ug = unigram(data)
    ug_counts = [(ug[w], w) for w in ug]
    ug_counts.sort()
    ug_counts.reverse()
    
    # unigram + bigram
    ug_bg_counts = ug_counts[:1000] + bg_counts[:1000]
    ug_bg_counts.sort()
    ug_bg_counts.reverse()
    
    ug_bg_words = [x[1] for x in ug_bg_counts[:1000]]
    ug_bg_wordId = dict(zip(ug_bg_words, range(len(ug_bg_words))))
    
    X = [feature2(d, ug_bg_words, ug_bg_wordId) for d in data]
    y = [d['review/overall'] for d in data]
    
    theta, residuals, rank, s = numpy.linalg.lstsq(X,y)
    print(residuals)
    
    print("------\nTask 4")
    tmp = []
    for key in ug_bg_wordId.keys():
        tmp.append( (theta[int(ug_bg_wordId[key])], key)   )
    
    tmp.sort()
    
    print("NEG")
    for i in range(5):
        print(str(tmp[i][1]) + ": " + str(tmp[i][0]))
        
    tmp.reverse()
    print("POS")
    for i in range(5):
        print(str(tmp[i][1]) + ": " + str(tmp[i][0]))
    
    
    '''
    with open('r4.txt', encoding='utf-8', mode='w') as fh:
        for i in range(len(ug_bg_words)):
            fh.write(ug_bg_words[i][1] + ": " + str(ug_bg_words[i][0]) + '\n')
            
        print("Done")
    '''
    
def t6(data):
    print("Task 6")
    p = set(string.punctuation)
    words = ['foam', 'smell', 'banana', 'lactic', 'tart']
    totals = []
    word_idf = []
   
    # IDF (closer to 0, more frequent appears in all documents)
    for word in words: 
        cnt = 0
        for d in data:
            cnt = 0
            r = ''.join([c for c in d['review/text'].lower() if not c in p])
            for w in r.split():
                if word == w:
                    cnt = 1
            totals.append(cnt)
              
        print("IDF(" + word + ") : " + str(log10(len(data)/sum(totals))))
        word_idf.append(log10(len(data)/sum(totals)))
        totals.clear()
    
        
    #tdidf
    index = 0
    for word in words: 
        cnt = 0
        r = ''.join([c for c in data[0]['review/text'].lower() if not c in p])
        for w in r.split():
            if word == w:
                cnt += 1
            
        print("TDIDF(" + word + ") : " + str(cnt * word_idf[index]))
        index += 1
        
    print("-------------\nTASK 7")
    '''
    ug1 = unigram([data[0]])
    ug2 = unigram([data[1]])
    ug1 = set(ug1.keys())
    ug2 = set(ug2.keys())
    ug12 = ug1.difference(ug2)
    
    ug_count1 = defaultdict(int)
    ug_count2 = defaultdict(int)
    word_idf = defaultdict(int)
    '''
    
    docs = []
    for d in data:
        r = ''.join([c for c in d['review/text'].lower() if not c in p])
        docs.append(r)
       
    tfidf_v = TfidfVectorizer()
    tfidf_mat = tfidf_v.fit_transform(tuple(docs))
    
    r = cosine_similarity(tfidf_mat[0:1], tfidf_mat)
    r = r[0]
    print(r[1])  
    
    print("---------\nTask 8")
    index = 1
    m = r[1]
    for i in range(2,5000):
        if r[i] > m:
            index = i
            m = r[i]
            
    print(r[index])
    print("BEER ID: " + data[index]['beer/beerId'])
    print("USER : " + data[index]['user/profileName'])
    
data = list(parseData(fname))[:5000]

t6(data)    