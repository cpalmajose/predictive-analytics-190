'''
    
    TODO: Comment About assignment

'''

import numpy
import scipy.optimize
import random
from sklearn.decomposition import PCA
from collections import defaultdict

fname = "beer_50000.json"

def parseData(fname):
    with open(fname, encoding='utf-8') as fh:
        for line in fh:
            yield eval(line)
    
centroids = [[0,0,0,0,1],[0,0,0,1,0]]
data = list(parseData(fname))    
X = [[x['review/overall'], x['review/taste'], x['review/aroma'], x['review/appearance'], x['review/palate']] for x in data]

def argmin(vector, centroid):
    result = 0
    for i in range(len(vector)):
        result = result + ((vector[i] - centroid[i]) ** 2)
    return result

def newCentroids(c_indexes, data):
    
    c0 = [0] * 5
    c1 = [0] * 5
    
    # Add points based on indexes
    for index in c_indexes[0]:
        for i in range(len(c0)):
            c0[i] = c0[i] + data[index][i]
    
    for index in c_indexes[1]:
        for i in range(len(c1)):
            c1[i] = c1[i] + data[index][i]
            
    # divide by number of points
    c0_len = len(c_indexes[0])
    c1_len = len(c_indexes[1])
    for i in range(len(c0)):
        c0[i],c1[i] = (c0[i]/c0_len, c1[i]/c1_len)
        
    return [c0,c1]
            
    

def avgNewMeanCentorid(centroid, count):
    new_c = centroid
    for i in range(len(centroid)):
        new_c[i] = new_c[i] / count
    return new_c

def kMeanCluster(data, centroids):
    ctr = 0
    y = []
    
    #centroid indexes of data
    ci_old = [ [], [] ]
    ci_new = [ [], [] ]
    
    #centroid center values
    cc_new = [centroids[0], centroids[1]]
    
    while True:

        ctr = ctr + 1
        if ctr % 10000 == 0:
            print("ITERATION: " + str(ctr))
            print("C0       : " + str(cc_new[0]))
            print("C1       : " + str(cc_new[1]))

        # Add X to centroid based on distance of the centroid
        for i in range(len(data)):
            k = 0
            min = 10000000000
            for j in range(len(cc_new)):
                result = argmin(data[i], cc_new[j])
                if result < min:
                    min = result
                    k = j
            
            # Append to output y
            y.append(k)   
            # Append to specific centorid
            ci_new[k].append(i)
            
        # Compare if the old values are not equal to new values
        if ci_new == ci_old:
            break;
        
        y.clear()
        # calculate new centroid centers
        cc_new = newCentroids(ci_new, data)
        
        # Move new indexes to old indexes
        ci_old = ci_new
        ci_new = [[],[]]
    
          
    print(len(ci_new[0]))
    print(len(ci_new[1]))
    print(len(y))
    print(cc_new[0])
    print(cc_new[1])
    
def p5(centroids, data):
    p=[[],[]]
    
    for i in range(len(data)):
        k = 0
        min = 100000000
        for j in range(len(centroids)):
            result = argmin(data[i], centroids[j])
            if result < min:
                min = result
                k = j
        
        p[k].append(i)
        
    print(len(p[0]))
    print(len(p[1]))
    
    return p
    
def p6(points, centroids, data):
    r = 0
    for i in range(len(points)):
        result = 0
        for index in points[i]:
            for j in range(len(data[index])):
                result = result + (data[index][j] - centroids[i][j])
            r = r + result
            result = 0
        print("Reconstruct error: " + str(i) + " | " + str(r))
        r = 0

c1 = [4.17993, 4.23675, 4.14107, 4.08866, 4.12518]
c2 = [3.09862, 3.06899, 3.14020, 3.38222, 3.11332]
c =[c1,c2]
#points = p5(c,X)
#p6(points,c,X)

import networkx as nx
import matplotlib.pyplot as plt

def tp1():
    
    edges = set()
    nodes = set()
    
    with open('egonet.txt', encoding='utf-8') as fh:
        for edge in fh:
            x,y = edge.split()
            x,y = int(x),int(y)
            edges.add((x,y))
            edges.add((y,x))
            nodes.add(x)
            nodes.add(y)

    #print("E: " + str(len(edges)))
    #print("N: " + str(len(nodes)))
    #print(nodes)
    
    ### Find all 3 and 4-cliques in the graph ###
    print("Getting cliques3/4")
    cliques3 = set()
    cliques4 = set()
    for n1 in nodes:
        for n2 in nodes:
            if not ((n1,n2) in edges): continue
            for n3 in nodes:
                if not ((n1,n3) in edges): continue
                if not ((n2,n3) in edges): continue
                clique = [n1,n2,n3]
                clique.sort()
                cliques3.add(tuple(clique))
                for n4 in nodes:
                    if not ((n1,n4) in edges): continue
                    if not ((n2,n4) in edges): continue
                    if not ((n3,n4) in edges): continue
                    clique = [n1,n2,n3,n4]
                    clique.sort()
                    cliques4.add(tuple(clique))
                    
    
    #Clique Percolation
    communities = set()
    result_communities = set()
    done = False
    for cliques in cliques4:
        communities.add(cliques)
        result_communities.add(cliques)
    
    print("Clique Percolation")
    #passctr = 0
    while not done:
        #passctr = passctr + 1
        #if passctr % 50 == 0:
        #    print("Passctr: " + str(passctr))
        #    print("result_community: " + str(len(result_communities)))
        
        done = True
        pass1 = False
        for community1 in result_communities:
            for community2 in result_communities:
                I = set(community1)
                J = set(community2)
                for c3 in cliques3:
                    c3s = set(c3)
                    if community1 is not community2 and c3s.issubset(I) and c3s.issubset(J):
                        done = False
                        result_communities.remove(community1)
                        result_communities.remove(community2)
                        newSet = I.union(J)
                        result_communities.add(tuple(newSet))
                        pass1 = True
                        done = False
                        break
                    
                if pass1: break;
            if pass1: break;
                    
    print("Total Communities: " + str(len(result_communities)))
    for t in result_communities:
        print("Community_len: " + str(len(t)))
        print("Nodes: \n" + str(t))
        
                    
''' 
    G = nx.Graph()
    for e in edges:
        G.add_edge(e[0], e[1])
    nx.draw(G)
    plt.show()
    plt.clf()
'''
    
tp1()