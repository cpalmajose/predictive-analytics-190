'''


'''
import gzip
from _collections import defaultdict
import numpy


def readGz(f):
    for l in gzip.open(f):
        yield eval(l)

# User Ratings
UserRatings = defaultdict(list)    

# Helpful Ratings
allHelpful = []
userHelpful = defaultdict(list)

# Lists for linear regression (part 4.c)
X = []
y = []

# get data
for l in readGz('train.json.gz'):
    user,item,rating = l['reviewerID'],l['itemID'],l['rating']
    UserRatings[user].append((item, rating))
    allHelpful.append(l['helpful'])
    userHelpful[user].append(l['helpful'])
    
    X.append([1,len(l['reviewText']),float(l['rating'])])
    y.append(float(l['helpful']['nHelpful']) / float(l['helpful']['outOf']))
    #y.append(float(l['helpful']['nHelpful']))

def task01():
    cnt = 0
    totalRating = 0
    for user in UserRatings.keys():
        for item in UserRatings[user]:
            cnt = cnt + 1
            totalRating = totalRating + item[1]
        
    averageRating = float(totalRating) / float(cnt) 
    print ("Average Rating: " + str(averageRating))
    
    cnt = 0
    mse = 0
    with open('labeled_Rating.txt', encoding='utf-8', mode='r') as fh:
        for line in fh:
            cnt = cnt + 1
            items = line.split()
            #mse = mse + ((averageRating - float(items[2])) ** 2)
            mse = mse + ((float(items[2]) - averageRating) ** 2)
            
    print ("MSE: " + str(mse/float(cnt)))
    
    
def task02():
    
    # help calculate alpha
    alpha = 0
    alpha_t = 0
    alpha_cnt = 0
    
    theta = 1
    
    # counter for items
    len_U_i = {}
    totals_U_i = {}
    bias_u = {}
    bias_i = {}
    
    # iterative procedure 
    for user in UserRatings.keys():
        
        len_I_u = 0
        I_u_total = 0
        bias_u[user] = 0
        
        for items in UserRatings[user]:
            
            # set item bias to zero if not in dict
            if items[0] not in bias_i.keys():
                bias_i[items[0]] = 0
                
            # calculate alpha
            alpha_cnt = alpha_cnt + 1
            alpha_t = alpha_t + (items[1] - (bias_u[user] + bias_i[items[0]]))
            alpha = alpha_t / alpha_cnt
            
            # calculate bias_u
            len_I_u = len_I_u + 1
            I_u_total = I_u_total + (items[1] - (alpha + bias_i[items[0]]))
            bias_u[user] = I_u_total / float(theta + len_I_u)
            
            # calculate bias_i
            if items[0] not in len_U_i.keys():
                len_U_i[items[0]] = 0
                totals_U_i[items[0]] = 0
                
            len_U_i[items[0]] = len_U_i[items[0]] + 1
            totals_U_i[items[0]] = totals_U_i[items[0]] + (items[1] - (alpha + bias_u[user]))
            bias_i[items[0]] = totals_U_i[items[0]] / float(theta + len_U_i[items[0]])
    
    
    
    print("BIAS for User - U566105319: " + str(bias_u['U566105319']))
    print("BIAS for item - I102776733: " + str(bias_i['I102776733']))
    
    # calculate mse
    cnt = 0
    mse = 0
    nids_u = 0
    nids_i = 0
    
    #test01 = set()
    #test02 = set()
    
    #for user in UserRatings.keys():
    #    test01.add(user)
    
    with open('labeled_Rating.txt', encoding='utf-8', mode='r') as fh:
        for line in fh:
            cnt = cnt + 1
            items = line.split()
            b_u = 0
            b_i = 0
            if items[0] in bias_u.keys():
                b_u = bias_u[items[0]]
            else: 
                nids_u = nids_u + 1
                
            if items[1] in bias_i.keys():
                b_i = bias_i[items[1]]
            else: 
                nids_i = nids_i + 1
            
            #test
            #test02.add(items[0])
                
            mse = mse + ((float(items[2]) - (alpha + b_u + b_i) ) ** 2)
    
            
    print("MSE: " + str(float(mse)/float(cnt)))
    print("Users not in dataset: " + str(nids_u))
    print("Items not in dataset: " + str(nids_i))
    print("Len user in training set: " + str(len(UserRatings.keys())))
    print("Number users in test set: " + str(cnt))
    
def task03():
    
    USER01 = 'U229891973'
    USER02 = 'U622491081'
    user01_set = set()
    user02_set = set()
    
    for items in UserRatings[USER01]:
        user01_set.add(items[0])
    for items in UserRatings[USER02]:
        user02_set.add(items[0])
        
    user01_02_intersect = user01_set.intersection(user02_set)
    user01_02_union = user01_set.union(user02_set)
    
    print("INTERSECT LENGTH: " + str(len(user01_02_intersect)))
    print("UNION LENGTH: " + str(len(user01_02_union)))
    
    userList = []
    jaccard_simularity = 0.0
   
    
    for user in UserRatings:
        # clear sets
        user01_set.clear()
        user01_02_intersect.clear()
        user01_02_union.clear()
        
        for items in UserRatings[user]:
            user01_set.add(items[0])
        
        user01_02_intersect = user01_set.intersection(user02_set)
        user01_02_union = user01_set.union(user02_set)
        
        result = float(len(user01_02_intersect)) / float(len(user01_02_union))
        if result > jaccard_simularity:
            userList.clear()
            userList.append(user)
            jaccard_simularity = result
        elif result == jaccard_simularity:
            userList.append(user)
        else:
            #do nothing
            pass 
    
    print("Jaccard Simularity: " + str(jaccard_simularity))
    print("Users Related: ")
    print(userList)
    
def task04():
    
    print("Task 04")
    #part a
    averageRate = sum([x['nHelpful'] for x in allHelpful]) * 1.0 / sum([x['outOf'] for x in allHelpful])
    print("AVERAGE RATE: " + str(averageRate))
    
    #part b
    absolute_error = 0
    mean_squared_error = 0
    predictions = []
    truth = []
    
    
    # create truth and prediction list
    with open('labeled_Helpful.txt', encoding='utf-8', mode='r') as fh:
        for line in fh:
            data = line.split()
            truth.append( float(data[3]) )
            predictions.append(averageRate * float(data[2]))
            
    #calculate mse and absolute error
    for p,t in zip(predictions, truth):
        mean_squared_error = mean_squared_error + ((t - p) ** 2)
        absolute_error = absolute_error + abs(t - p)
    
    
    mean_squared_error = mean_squared_error / float(len(truth))
    
    print("MSE: " + str(mean_squared_error))
    print("AE : " + str(absolute_error))
    print("---------------------")
    
    
    # part 4.c (Linear Regression)
    theta,residuals,rank,s = numpy.linalg.lstsq(X, y)
    
    #print("Theta: " + str(theta))
    #print("Residuals: " + str(residuals))
    print("Alpha: " + str(theta[0]))
    print("B_1: " + str(theta[1]))
    print("B_2: " + str(theta[2]))
    print("---------------------")
    
    alpha = theta[0]
    
    # part 4.d Using features to calculate mse and absolute error from guessing nhelpful
    d = []
    for l in readGz('helpful.json.gz'):
        d.append((float(l['helpful']['outOf']), len(l['reviewText']), float(l['rating'])))
        
    mse = 0
    ae = 0
    cnt = 0
    for i,j in zip(truth,d):
        '''
        if cnt < 5:
            cnt = cnt + 1
            print(cnt)
            print("T: " + str(i))
            print("P: " + str( (alpha * j[0]) + (theta[1] * j[1]) + (theta[2] * j[2]) ))
'''
        mse = mse + ((i - ((alpha * j[0]) + (theta[1] * j[1]) + (theta[2] * j[2]))) ** 2)
        ae = ae + (i - ((alpha * j[0]) + (theta[1] * j[1]) + (theta[2] * j[2])))
        
    mse = mse / len(truth)
    print("MSE: " + str(mse))
    print("AE: " + str(ae))
        
#task01()
#task02()
#task03()
task04()