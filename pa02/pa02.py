'''
Data Structure:
beer/name
beer/beerId
beer/brewerId
beer/ABV
beer/style
review/apearance
review/aroma
review/palate
review/taste
review/overall
review/time
review/profileName
review/text
'''
import sys
import datetime
import random
import pickle
import re 
from collections import defaultdict

from sklearn import datasets
import numpy
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestRegressor


import plotly.plotly as py
from plotly.graph_objs import *

py.sign_in("calmajos", "61yw68mizx")

# Regular Expression to parse data
A = re.compile(r'^(.+?(?=:)): (.+)$')
B = re.compile(r'^.+?(?=:): ')

FNAME = 'beeradvocate.txt'
TEST_FNAME = 'test.p'
VALID_FNAME = 'valid.p'

def parseData(filename, test_fname, valid_fname):
    i = []
    okay = True
    ctr = 0
    print("PARSING DATA")
    with open('beeradvocate.txt', encoding='utf-8', mode='r') as fh:
        d = {}
        for line in fh:
            #linnum += 1
            nl = line.rstrip('\n')
            r = A.search(nl)
            r2 = B.search(nl)
            
            if r and okay:
                result = re.sub(r'\s+', " ", r.group(2))
                d[r.group(1)] = result
            elif r2 and okay:
                #print("BAD VALUE at " + str(linnum) + line.rstrip('\n'))
                okay = False
                d = {}
            else:
                if okay:  
                    i.append(d)
                    d = {}  
                    ctr += 1
                    if ctr % 10000 == 0:
                        print(ctr)  
                elif not nl:
                    okay = True
            
            if len(i) == 100000:
                break;
    
    random.shuffle(i)
    testData = i[:75000]
    vData = i[75000:]
    print("LENGTH OF TEST DATA: " + str(len(testData)))
    print("LENGTH OF VALIDTION DATA: " + str(len(vData)))
    with open(test_fname, mode='wb') as fp:
        pickle.dump(testData, fp)  
    with open(valid_fname, mode='wb') as fp:
        pickle.dump(vData, fp)    
    print("Parse Complete")
    
def openData(fname):
    with open(fname, mode='rb') as fh:
        data = pickle.load(fh)
        
    return data

# Parses first one million data entries
# Writes pickle files for test data and validation data

#parseData(FNAME, TEST_FNAME, VALID_FNAME)

# Opens the data files 

def feature(data, field=None):
    feature = [1]
    if field:
        feature.append(data[field])
    return feature

#parseData(FNAME, TEST_FNAME, VALID_FNAME)

def feature2(data, field1, field2, field3=None):
    feature = [1]
    feature.append(float(data[field1]))
    
    if field2 == 'review/text':
        feature.append(float(len(data[field2])))
    else:
        feature.append(float(data[field2]))
        
    if field3:
        feature.append(float(data[field3]))
        
    return feature

def createDict(ylist):
    x = []
    y = []
    xDict = defaultdict(int)
    for i in ylist:
        xDict[i] += 1
    
    l = list(xDict.keys())
    l.sort()
    
    for key in l:
        x.append(key)
        y.append(xDict[key])
        
    return (x,y)

def generateBarGraph(x_y_pts, names, xAxis, yAxis, gname):
    scatters = []
    i = 0
    for x, y in x_y_pts:
        name=names[i]
        i += 1
        s =  Bar(x=x,y=y, name=name)
        scatters.append(s)
        
    data = Data(scatters)
    layout = Layout(
        title=gname,
        xaxis=XAxis(title=xAxis),
        yaxis=YAxis(title=yAxis)            
        )
    fig = Figure(data=data,layout=layout)
    plot_url = py.plot(fig, fname=gname)

def explore(testData, validData):
    
    plot_names = ['overall', 'appearance', 'aroma', 'taste']
    x_y_list = []
    y_taste = [float(d['review/taste']) for d in testData]
    
    X = [feature(d) for d in testData]
    y = [float(d['review/overall']) for d in testData]
    
    theta,residuals,rank, s = numpy.linalg.lstsq(X, y)
    x_y_list.append(createDict(y))
    
    clf = linear_model.LinearRegression(fit_intercept=False)
    clf.fit(X,y)
    X2 = [feature(d) for d in validData]
    y2 = [float(d['review/overall']) for d in validData]
    y_p2 = clf.predict(X2)
    print("MSE: (TEST) " + str(mean_squared_error(y2, y_p2)))
    
    y_p = [theta[0]] * len(y)
    #print("TEST: " + str(mean_squared_error(y,y)) + " | SHOULD BE ZERO")
    print("OVERALL overall: " + str(theta))
    print("MSE: " + str(mean_squared_error(y_taste,y_p)))
    y.sort()
    print("MIN: " + str(y[0]) + "| MAX: " + str(y[-1]))
    print("---------")
    
    y = [float(d['review/appearance']) for d in testData]
    theta,residuals,rank, s = numpy.linalg.lstsq(X, y)
    x_y_list.append(createDict(y))
    
    clf = linear_model.LinearRegression(fit_intercept=False)
    clf.fit(X,y)
    X2 = [feature(d) for d in validData]
    y2 = [float(d['review/appearance']) for d in validData]
    y_p2 = clf.predict(X2)
    print("MSE: (TEST) " + str(mean_squared_error(y2, y_p2)))
    
    y_p = [theta[0]] * len(y)
    print("OVERALL APPEARANCE: " + str(theta))
    print("MSE: " + str(mean_squared_error(y_taste,y_p)))
    y.sort()
    print("MIN: " + str(y[0]) + "| MAX: " + str(y[-1]))
    print("---------")
    
    y = [float(d['review/aroma']) for d in testData]
    theta,residuals,rank, s = numpy.linalg.lstsq(X, y)
    y_p = [theta[0]] * len(y)
    x_y_list.append(createDict(y))
    
    clf = linear_model.LinearRegression(fit_intercept=False)
    clf.fit(X,y)
    X2 = [feature(d) for d in validData]
    y2 = [float(d['review/aroma']) for d in validData]
    y_p2 = clf.predict(X2)
    print("MSE: (TEST) " + str(mean_squared_error(y2, y_p2)))
    
    print("OVERALL AROMA: " + str(theta))
    print("MSE: " + str(mean_squared_error(y_taste,y_p)))
    y.sort()
    print("MIN: " + str(y[0]) + "| MAX: " + str(y[-1]))
    print("---------")
    
    y = [int(d['review/time']) for d in testData]
    theta,residuals,rank, s = numpy.linalg.lstsq(X, y)
    
    y_p = [theta[0]] * len(y)
    print("OVERALL TIME: " + str(theta))
    print("MSE: " + str(mean_squared_error(y_taste,y_p)))
    y.sort()
    print("MIN: " + str(y[0]) + "| MAX: " + str(y[-1]))
    print("---------")
    
    y = [float(d['review/taste']) for d in testData]
    theta,residuals,rank, s = numpy.linalg.lstsq(X, y)
    x_y_list.append(createDict(y))
    
    clf = linear_model.LinearRegression(fit_intercept=False)
    clf.fit(X,y)
    X2 = [feature(d) for d in validData]
    y2 = [float(d['review/taste']) for d in validData]
    y_p2 = clf.predict(X2)
    print("MSE: (TEST) " + str(mean_squared_error(y2, y_p2)))
    
    y_p = [theta[0]] * len(y)
    print("OVERALL TASTE: " + str(theta))
    print("MSE: " + str(mean_squared_error(y_taste,y_p)))
    y.sort()
    print("MIN: " + str(y[0]) + "| MAX: " + str(y[-1]))
    print("---------")
    
    y = [len(d['review/text'].split()) for d in testData]
    theta,residuals,rank, s = numpy.linalg.lstsq(X, y)
    
    clf = linear_model.LinearRegression(fit_intercept=False)
    clf.fit(X,y)
    X2 = [feature(d) for d in validData]
    y2 = [len(d['review/text'].split()) for d in validData]
    y_p2 = clf.predict(X2)
    print("MSE: (TEST) " + str(mean_squared_error(y2, y_p2)))
    
    y_p = [theta[0]] * len(y)
    print("OVERALL TEXT LENGTH: " + str(theta))
    print("MSE: " + str(mean_squared_error(y_taste,y_p)))
    y.sort()
    print("MIN: " + str(y[0]) + "| MAX: " + str(y[-1]))
    print("---------")
    
    y = [float(d['beer/ABV']) for d in testData]
    theta,residuals,rank, s = numpy.linalg.lstsq(X, y)
    
    clf = linear_model.LinearRegression(fit_intercept=False)
    clf.fit(X,y)
    X2 = [feature(d) for d in validData]
    y2 = [len(d['review/text'].split()) for d in validData]
    y_p2 = clf.predict(X2)
    print("MSE: (TEST) " + str(mean_squared_error(y2, y_p2)))
    
    y_p = [theta[0]] * len(y)
    print("OVERALL ABV: " + str(theta))
    print("MSE: " + str(mean_squared_error(y_taste,y_p)))
    y.sort()
    print("MIN: " + str(y[0]) + "| MAX: " + str(y[-1]))
    print("---------")
    
    # plot stuff
    generateBarGraph(x_y_list, plot_names, 'Rating', 'Counts', 'Counts Graph')

def explore2(testData, validData):
    print("ABV and APPEARANCE")
    X = [feature2(d, 'beer/ABV', 'review/appearance') for d in testData]
    y = [float(d['review/taste']) for d in testData]
    
    #clf = linear_model.LinearRegression(fit_intercept=False)
    clf = RandomForestRegressor()
    clf.fit(X, y)
    
    
    test_predictions = clf.predict(X)
    print("MSE (TRAIN): " + str(mean_squared_error(test_predictions, y)))
    
    X = [feature2(d, 'beer/ABV', 'review/appearance') for d in validData]
    y = [float(d['review/taste']) for d in validData]
    test_predictions = clf.predict(X)
    print("MSE (TEST): " + str(mean_squared_error(test_predictions, y)))
    print("SCORE: " + str(clf.score(X,y)))
    
    print("-----")
    print("ABV and AROMA")
    X = [feature2(d, 'beer/ABV', 'review/aroma') for d in testData]
    y = [float(d['review/taste']) for d in testData]
    
    #clf = linear_model.LinearRegression(fit_intercept=False)
    clf = RandomForestRegressor()
    clf.fit(X, y)
    
    
    test_predictions = clf.predict(X)
    print("MSE (TRAIN): " + str(mean_squared_error(test_predictions, y)))
    
    X = [feature2(d, 'beer/ABV', 'review/aroma') for d in validData]
    y = [float(d['review/taste']) for d in validData]
    test_predictions = clf.predict(X)
    print("MSE (TEST): " + str(mean_squared_error(test_predictions, y)))
    print("SCORE: " + str(clf.score(X,y)))
    
    print("-----")
    print("ABV and Text Length")
    X = [feature2(d, 'beer/ABV', 'review/text') for d in testData]
    y = [float(d['review/taste']) for d in testData]
    
    #clf = linear_model.LinearRegression(fit_intercept=False)
    clf = RandomForestRegressor()
    clf.fit(X, y)
    
    
    test_predictions = clf.predict(X)
    print("MSE (TRAIN): " + str(mean_squared_error(test_predictions, y)))
    
    X = [feature2(d, 'beer/ABV', 'review/text') for d in validData]
    y = [float(d['review/taste']) for d in validData]
    test_predictions = clf.predict(X)
    print("MSE (TEST): " + str(mean_squared_error(test_predictions, y)))
    print("SCORE: " + str(clf.score(X,y)))
    
    print("-----")
    print("ABV and Palate")
    X = [feature2(d, 'beer/ABV', 'review/palate') for d in testData]
    y = [float(d['review/taste']) for d in testData]
    
    #clf = linear_model.LinearRegression(fit_intercept=False)
    clf = RandomForestRegressor()
    clf.fit(X, y)
    
    
    test_predictions = clf.predict(X)
    print("MSE (TRAIN): " + str(mean_squared_error(test_predictions, y)))
    
    X = [feature2(d, 'beer/ABV', 'review/palate') for d in validData]
    y = [float(d['review/taste']) for d in validData]
    test_predictions = clf.predict(X)
    print("MSE (TEST): " + str(mean_squared_error(y, test_predictions)))
    print("SCORE: " + str(clf.score(X,y)))
    
    print("---")
    print("ABV, Aroma, Palate")
    X = [feature2(d, 'beer/ABV', 'review/palate', 'review/aroma') for d in testData]
    y = [float(d['review/taste']) for d in testData]
    
    #clf = linear_model.LinearRegression(fit_intercept=False)
    clf = RandomForestRegressor()
    clf.fit(X, y)
   
    
    test_predictions = clf.predict(X)
    print("MSE (TRAIN): " + str(mean_squared_error(test_predictions, y)))
    
    X = [feature2(d, 'beer/ABV', 'review/palate', 'review/aroma') for d in validData]
    y = [float(d['review/taste']) for d in validData]
    test_predictions = clf.predict(X)
    print("MSE (TEST): " + str(mean_squared_error(y, test_predictions)))
    print("SCORE: " + str(clf.score(X,y)))
    
    #---------------------------------------------------------------
    
def explore3(testData, validData):    
   
    user_to_review = defaultdict(list)
    beer_to_review = defaultdict(list)
    
    vuser_to_review = defaultdict(list)
    vbeer_to_review = defaultdict(list)
    
    for item in testData:
        l = feature2(item, 'beer/ABV', 'review/palate', 'review/aroma')
        l.append(float(item['review/taste']))
        user_to_review[item['review/profileName']].append(l)
        beer_to_review[item['beer/beerId']].append(l)
        
    for item in validData:
        l = feature2(item, 'beer/ABV', 'review/palate', 'review/aroma')
        l.append(float(item['review/taste']))
        vuser_to_review[item['review/profileName']].append(l)
        vbeer_to_review[item['beer/beerId']].append(l)
        
    tv_user = []
    tv_beer = []
    for i in user_to_review.keys():
        if i in vuser_to_review.keys():
            tv_user.append(i)
            
    for i in beer_to_review.keys():
        if i in vbeer_to_review.keys():
            tv_beer.append(i)
            
    for user in tv_user:
        all_stuff = user_to_review[user]
        X = [d[:-1] for d in all_stuff]
        y = [d[-1] for d in all_stuff]
        clf = linear_model.LinearRegression(fit_intercept=False)
        clf.fit(X,y)
        
        recommend = []
        for beerId in vbeer_to_review.keys():
            nX = [d[:-1] for d in vbeer_to_review[beerId]]
            ny = [float(d[-1]) for d in vbeer_to_review[beerId]]
            if clf.score(nX,ny) > 0.63:
                recommend.append(beerId)
                
                p = clf.predict(nX)
                print("MSE: " + str(mean_squared_error(p, ny)))
                for i,j in zip(p,ny):
                    print(str(i) + " : " + str(j))
                
                
        print(len(recommend)/len(vbeer_to_review.keys()))
        for bid in recommend:
            print("BID: " + bid + str(vbeer_to_review[bid]))
        sys.exit()
    
    print("CREATING LOGISTIC REGESSION")
    
    X = [feature2(d, 'beer/ABV', 'review/palate', 'review/aroma') for d in testData]
    y = [float(d['review/taste']) for d in testData]
    
    clf = linear_model.LogisticRegression(fit_intercept=False, C=1000)
    clf.fit(X,y)
    
    X = [feature2(d, 'beer/ABV', 'review/palate', 'review/aroma') for d in validData]
    y = [float(d['review/taste']) for d in validData]
    #y_p = clf.predict(X)
    #yy_p = clf.decision_function(X)
    '''
    l = 0
    for i,j in zip(y_p, y):
        print(yy_p[l])
        l += 1
        print(str(i) + " | " + str(j))
    '''
def explore4(testData, validData): 
    
    xTrain=[0.3336, 0.24259, 0.22789, 0.44112,0.177228]
    yTrain=[0.31485, 0.5017, 0.5231, 0.09405, 0.636021]
    names=['ABV/APPEARANCE', 'ABV/AROMA', 'ABV/PALATE', 'ABV/TEXT_LENGTH', 'ABV/AROMA/PALATE']
     
    data = Data([Bar(x=names, y=xTrain)])
    layout = Layout(title='MSE (Test set)',
                    yaxis=YAxis(title="MSE"))
    fig = Figure(data=data, layout=layout)
    plot_url = py.plot(fig, fname='MSE')
   
    data = Data([Bar(x=names, y=yTrain)])
   
    layout = Layout(title='Scores',
                    yaxis=YAxis(title="Scores"))
    fig = Figure(data=data, layout=layout)
    plot_url = py.plot(fig, fname='Scores')
    
'''
for i in range(5):
    d = testData[i]['review/time']
    print(datetime.datetime.fromtimestamp(int(d)).strftime('%I:%M %p (%H:%M)'))
'''
    
testData = openData(TEST_FNAME)
validData = openData(VALID_FNAME)

#explore(testData, validData)
#explore2(testData, validData)
explore3(testData, validData)
#explore4(testData, validData)
