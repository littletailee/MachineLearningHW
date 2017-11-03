# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 16:15:45 2017

@author: jwzhang1996
"""

import scipy.io as spio
import numpy as np
import pandas as pd
import random
import pylab as pl
import seaborn as sns
import matplotlib.pyplot as plt     

classifiers = []
for i in range(10):
    for j in range(i + 1, 10):
        classifiers.append([i,j])
        
dat = spio.loadmat('hw1data.mat', squeeze_me = True)
X = dat['X']
Y = dat['Y']

class Perceptron():
    def __init__(self, T, trainprop):
        self.T = T
        self.trainprop = trainprop
    
    def per0(self, X, Y, w, i, k):
        #perceptron V0: if there is a mistake, correct it immediately
        test = Y[i] * (np.dot(w[k], X[i]))
        if test <= 0:
            w[k] += Y[i] * X[i]
        
    def per1(self, X, Y, w, i, k):
        #perceptron V1: find the minimum value of y*<w,x>, if it's negative, correct it
        test = np.multiply(Y, (np.dot(X,w[k])))
        if min(test) <= 0:
            i = np.where(test == min(test))[0][0]
            w[k] += Y[i] * X[i]

    def test(self, classifiers = classifiers):
        testX = self.X[self.trainnum:]
        testY = self.Y[self.trainnum:]
        err = 0
        for i in range(self.N - self.trainnum):
            count = [0] * 10
            for k in range(len(classifiers)): 
                if np.dot(self.w[k], testX[i]) > 0:
                    count[classifiers[k][0]] += 1
                else:
                    count[classifiers[k][1]] += 1
            label = count.index(max(count))
            if label != testY[i]:
                err += 1
        return err        
        
    def traindataset(self, X, Y):
        [self.N, self.D] = X.shape       
        item = [i for i in range(self.N)]
        random.shuffle(item)
        self.X = np.column_stack([X[item],np.ones(self.N)])
        self.D += 1
        self.Y = Y[item]      
        self.trainnum = int(self.trainprop * self.N)
        self.trainX = self.X[0:self.trainnum]
        self.trainY = self.Y[0:self.trainnum]
    
    def fit(self, X, Y, classifiers, per = per0):  
        self.traindataset(X, Y)
        self.w = pd.DataFrame([np.zeros(45)] * self.D)
        for k in range(len(classifiers)):
            sub_trainX = np.concatenate([self.trainX[self.trainY == classifiers[k][0]], self.trainX[self.trainY == classifiers[k][1]]],axis = 0)
            sub_trainY = np.concatenate([[1] * len(self.trainY[self.trainY == classifiers[k][0]]), [-1] * self.trainY[self.trainY == classifiers[k][1]]],axis = 0)
            l = len(sub_trainY)
            item = [i for i in range(l)]
            random.shuffle(item)
            sub_trainX = sub_trainX[item]
            sub_trainY = sub_trainY[item]
            for t in range(self.T):
                i = t % l
                self.per0(sub_trainX, sub_trainY, self.w, i, k)

split = [int(i*0.3*len(Y)) for i in range(1, 11)] #different T
trainratio = [i/10 for i in range(8,10)] #different train size
sns.set()

def errortest(split, trainratio, Perceptron, method):
    err = pd.DataFrame([np.zeros(len(split))] * len(trainratio))
    for i in range(len(split)):
        for j in range(len(trainratio)):
            V0 = Perceptron(split[i], trainratio[j])
            if method == 0:
                per = V0.per0
            elif method == 1:
                per = V0.per1
            V0.fit(X, Y, classifiers, per)
            err[i][j] = V0.test(classifiers)
    return err

err0 = errortest(split, trainratio, Perceptron, 0) #Perceptron V0
err1 = errortest(split, trainratio, Perceptron, 1) #Perceptron V1

#plot
N = len(Y)
plt.scatter(split, [1-i/(N * (1 - trainratio[0])) for i in err0.iloc[0]])
plt.scatter(split, [1-i/(N * (1 - trainratio[1])) for i in err0.iloc[1]])
plot1, = plt.plot(split, [1-i/(N * (1 - trainratio[0])) for i in err0.iloc[0]], label='80% train data')
plot2, = plt.plot(split, [1-i/(N * (1 - trainratio[1])) for i in err0.iloc[1]], label='90% train data')
plt.scatter(split, [1-i/(N * (1 - trainratio[0])) for i in err1.iloc[0]])
plt.scatter(split, [1-i/(N * (1 - trainratio[1])) for i in err1.iloc[1]])
plot3, = plt.plot(split, [1-i/(N * (1 - trainratio[0])) for i in err1.iloc[0]], label='80% train data')
plot4, = plt.plot(split, [1-i/(N * (1 - trainratio[1])) for i in err1.iloc[1]], label='90% train data')
plt.legend([plot1, plot2, plot3, plot4], ["V0 80% train data", "V0 90% train data", "V1 80% train data", "V1 90% train data"])
pl.title('Plot of Accuracy Rate of Perceptron')# give plot a title
pl.xlabel('Number of Passes(T)')# make axis labels
pl.ylabel('Accuracy Rate')

#kernel
def polynomial_kernel(x, y, p = 5):
    return np.dot(x, y) ** p

def pv2test(testY, classifiers, fx):
    err = 0
    l = len(testY)
    for i in range(l):
        count = [0] * 10
        for k in range(45): 
            if fx.iloc[i][k] > 0:
                count[classifiers[k][0]] += 1
            else:
                count[classifiers[k][1]] += 1
        label = count.index(max(count))
        if label != testY[i]:
            err += 1
    return err

trainnum = int(0.9 * N)    
trainX = X[0:trainnum]
trainY = Y[0:trainnum]
testX = X[trainnum:]
testY = Y[trainnum:]
count = []
for p in range(5,10):    
    fx = pd.DataFrame([np.zeros(45)] * len(testY))
    for k in range(45):
        sub_trainX = np.concatenate([trainX[trainY == classifiers[k][0]], trainX[trainY == classifiers[k][1]]],axis = 0)
        sub_trainY = np.concatenate([[1] * len(trainY[trainY == classifiers[k][0]]), [-1] * len(trainY[trainY == classifiers[k][1]])],axis = 0)
        l = len(sub_trainY)
        alpha = np.zeros(l)
        item = [i for i in range(l)]
        random.shuffle(item)
        sub_trainX = sub_trainX[item]
        sub_trainY = sub_trainY[item]
        K = np.dot(sub_trainX, sub_trainX.T) ** p
        for t in range(10000):
            i = t % l
            test = np.sign(np.sum(K[:,i] * alpha * sub_trainY))* sub_trainY[i]
            if test <= 0 :
                alpha[i] += 1
        for j in range(N - trainnum):      #for every test point
            s = 0
            kernel = [polynomial_kernel(sub_trainX[i], testX[j], p) for i in range(l)]
            s = np.sign(np.sum(kernel * alpha * sub_trainY))
            fx[k][j] = s
    count.append(pv2test(testY, classifiers, fx))
pos = [i*45-1 for i in range(1,6)]
count0 = [count[i] for i in pos]
plt.scatter(split, [1-i/(N - trainnum) for i in count0])
plot1, = plt.plot(split, [1-i/(N - trainnum) for i in count0], label='Kernel Perceptron V0')
pl.title('Plot of Accuracy Rate of Kernel Perceptron V0(90% Training Data, T=10000)')# give plot a title
pl.xlabel('Number of Order')# make axis labels
pl.ylabel('Accuracy Rate')
    
        
