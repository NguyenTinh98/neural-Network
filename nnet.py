from math import *
import numpy as np



def sizeOfWeight(x, y):
    return x*y/2

def signmod(x):
    return 1 / (1+exp(-x))

def randomMatrix(row, col):
    return np.random.rand(row, col)


def dotMatrix(x, y):
    x = np.array(x)
    y = np.array(y)
    return sum(x*y)

class nnet:
    a = [[]]
    weights = []
    x = []
    dts = [[]]
    h = []
    hlNodes = []
    h0 = []
    def __init__(self, dts, nodes, h0):
        self.dts = dts.copy()
        self.hlNodes.append(len(dts[0]))
        for i in range (len(nodes)):
            self.hlNodes.append(nodes[i])
        self.hlNodes.append(1)
        self.h0 = h0.copy()
        self.a[0] = dts.copy() #a[i]: layer i


    #initialize weights // done   
    def initWeights(self):
        for i in range(len (self.hlNodes)-1):
            temp = self.hlNodes[i]+1
            if i == 0:
                temp = self.hlNodes[i]
            self.weights.append(randomMatrix( self.hlNodes[i+1],temp))


    def initActives(self):
        for i in range(1, len(self.hlNodes)):
            temp = self.hlNodes[i]+1
            t = [1] * temp
            tmp = []
            for i in range (len(dts)):
                tmp.append(t)
            self.a.append(tmp.copy())




    def updateAs(self):
        for i in range (1, len(self.hlNodes)):
            for j in range (len(self.dts)):
                for k in range (1, len(self.a[i][j])):
                    self.a[i][j][k] = signmod(dotMatrix(self.weights[i-1][k-1],self.a[i-1][j]))


    def updateHw(self):
        for i in range (len(dts)):
            self.h[i]=self.a[len(self.hlNodes)-1][i][1]


    def forwardProp(self):
        return True
    

    def backProp(self):
        self.updateAs()
        for i in range (len(dts)):
            sigmaL += 

    
    def calDiff(self):
        return True

    def cost(self):
        cost = 0
        for i in range (len(self.dts)):
            cost += self.h0[i]*log(self.h[i])+(1-h0[i])*log(1-h[i])
        cost /= -len(dts)
        return cost


    

'''
a = [[1], [2]]
x = [[2], [3]]
a = np.array(a)
x = np.array(x)
print (a, x)
print (a.T.dot(x))
print (signmod(a,x))'''
dts = [[1,2,3,4],[1,2,5,4]]
nodes = [3, 2]
h = [2,3]
mynnet = nnet(dts, nodes, h)
mynnet.initWeights()
mynnet.initActives()
mynnet.updateAs()
print (mynnet.weights)
print (mynnet.a)
print(len(mynnet.a[1][0]))
print(mynnet.weights[0][0])
print(dotMatrix([1,2,3],[1,2,1]))