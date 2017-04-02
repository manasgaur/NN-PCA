from math import exp
from random import seed
import random
from random import randrange
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
''' Initialize the network '''
def init_network(numinputs, numhiddenlayer, numoutputs):
    net=list()
    # hidden layer
    HiddenLayer = [{'weights':[np.random.uniform() for i in range(numinputs+1)]} for i in range(numhiddenlayer)]
    net.append(HiddenLayer)
    OutputLayer = [{'weights':[np.random.uniform() for i in range(numhiddenlayer+1)]} for i in range(numoutputs)]
    net.append(OutputLayer)
    return net

''' Transfer activation function for the neuron '''

def transferFunction(actn):
    return 1.0/(1.0+exp(-actn))

'''Calculation of activation of neuron from an input '''

def activateFunction(wts, inp):
    activaN = wts[-1]
    for i in range(len(wts)-1):
        activaN+=wts[i]*inp[i]
    return activaN

''' Forward Propagation of the network '''

def forwardPropagationNet(netK, row):
    inps = row
    for layer in netK:
        newinps=[]
        for neuron in layer:
            activaN = activateFunction(neuron['weights'], inps)
            neuron['output'] = transferFunction(activaN)
            newinps.append(neuron['output'])
        inps = newinps
    return inps


''' Derivative of the Transfer Function '''
def derivativeTransferFunction(op):
    return (op)*(1-op)

''' Backward Propagation of the network '''

def backwardPropagationNet(netK, observed):
    for i in reversed(range(len(netK))):
        layer = netK[i]
        errorlst = list()
        if i != len(netK)-1:
            for j in range(len(layer)):
                e = 0.0
                for neuron in netK[i+1]:
                    e += (neuron['weights'][j] * neuron['delta'])
                errorlst.append(e)
        else:
            for k in range(len(layer)):
                neuron = layer[k]
                errorlst.append(observed[k] - neuron['output'])
        for p in range(len(layer)):
            neuron = layer[p]
            neuron['delta'] = errorlst[p] * derivativeTransferFunction(neuron['output'])


''' Update Weights of the Network '''
def updateWeightsFunction(netK, row, learningRate):
    for i in range(len(netK)):
        inp = row[:-1]
        if i != 0:
            inp = [neuron['output'] for neuron in netK[i-1]]
        for neuron in netK[i]:
            for j in range(len(inp)):
                neuron['weights'][j] += learningRate * neuron['delta'] * inp[j]
            neuron['weights'][-1] += learningRate * neuron['delta']


''' Training the Network.'''

def trainingNetwork(netK, trainData, learningRate, epochs, numoutputs):
    print("Learning Rate of the network = %.3f" % (learningRate))
    #epocherror = 0
    for epoch in range(epochs):
        errorsum = 0
        for row in trainData:
            outputs = forwardPropagationNet(netK,row)
            observed = [0 for i in range(numoutputs)]
            observed[row[-1]] = 1
            errorsum -= sum([(observed[i]-outputs[i])**2 for i in range(len(observed))])
            backwardPropagationNet(netK,observed)
            updateWeightsFunction(netK,row,learningRate)
        print('Number_of_Epoch=%d, trainingError=%.3f' % (epoch,errorsum))
        #epocherror=errorsum
    #print("Training Accuracy of the Model after Training is %.3f" % (100+epocherror))


''' Cross Validation function '''
def splitByCrossValidation(data,nFolds):
    dataSplit = list()
    copyData = data
    foldNumber = int(len(data) / nFolds)
    for i in range(nFolds):
        fold = list()
        while len(fold) < foldNumber :
            idx = randrange(len(copyData))
            fold.append(copyData.pop(idx))
        dataSplit.append(fold)
    return dataSplit

''' Prediction Function '''
def predict(netK,row):
    out = forwardPropagationNet(netK,row)
    return out.index(max(out))

''' Accuracy Metric '''

def accuracyCalculation(A,P):
    hit = 0
    for i in range(len(P)):
        if A[i] == P[i]:
            hit+=1
    return (hit/float(len(A))) * 100.0


''' Main Function '''

if __name__ == '__main__':
    seed(1)
    f=open("/Users/manasgaur/Downloads/SoftComputing3/Homework_Three_Data/SC_DJI_PHANTOM_SET_ONE",'r')
    f2=open("/Users/manasgaur/Downloads/SoftComputing3/Homework_Three_Data/SC_DJI_PHANTOM_SET_TWO",'r')
    temp=[]
    temp2=[]
    for line in f.readlines():
        temp.append((line.split('\n')[0].replace('\'','')))
    for line in f2.readlines():
        temp2.append((line.split('\n')[0].replace('\'','')))
    print  "Number of tuples in the data (data is still of string type)",len(temp+temp2)
    data1=[]
    data2=[]
    for i in range(len(temp)):
        x=[]
        terms = temp[i].split(' ')[:-1]
        for t in range(len(terms)):
            tmp = float(terms[t])
            x.append(tmp)
        data1.append(x)
    for i in range(len(temp2)):
        x=[]
        terms = temp2[i].split(' ')[:-1]
        for t in range(len(terms)):
            tmp = float(terms[t])
            x.append(tmp)
        data2.append(x)
    data= data1+data2
    newdata = data
    dataset=[]
    ''' Preparing the data'''
    for i in range(len(newdata)):
        ele=newdata[i][0]
        newlst=newdata[i][1:]
        if ele == 1.0:
            newlst.append(1)
        if ele == -1.0 or ele == 0.0:
            newlst.append(0)
        dataset.append(newlst)
    D=[]; L=[];
    for i in range(len(dataset)):
        L.append(dataset[i][-1])
        D.append(dataset[i][:-2])
    ''' We are performing dimensionality reduction on this dataset'''

    X = np.asarray(D)
    pca = PCA(n_components=24)
    X1 = pca.fit_transform(X)
    newdataset=[]
    for i in range(len(X1)):
        x = []
        for j in range(len(X1[0])):
            x.append(X1[i][j])
        x.append(L[i])
        newdataset.append(x)
    dataset = newdataset
    print('Data Loaded and Prepared.')
    numFolds = 2
    print('Now we are performing CrossValidation')
    dataSplit = splitByCrossValidation(dataset,numFolds)
    resultscore = list()
    trainset=list()
    testset=list()
    for fld in dataSplit:
        trainset = list(dataSplit)
        trainset.remove(fld)
        trainset = sum(trainset,[])
        testset =list()
        for r in fld:
            copyrow = list(r)
            testset.append(copyrow)
            copyrow[-1] = None
        numinputs = len(trainset[0])-1
        print("Number of the features of the dataset %d" %(numinputs))
        numoutputs = len(set(row[-1] for row in trainset))
        print("Number of the labels of the dataset %d" %(numoutputs))
        print("Initializing the Network .......")
        ''' Passing the number of features, number of hidden neurons and number of outputs'''
        netK = init_network(numinputs,2,numoutputs)
        print("Network Initialized. Training of the Network, begins ....")
        ''' Passing data, network, learning rate, epochs and number of outputs'''
        trainingNetwork(netK, trainset, 0.1, 300, numoutputs)
        print('Printing the output of each layer of the network')
        for layer in netK:
            print(layer)
        predictions = list()
        for row in testset:
            pred = predict(netK, row)
            predictions.append(pred)
        actualval = [row[-1] for row in fld]
        accuracy = accuracyCalculation(actualval,predictions)
        resultscore.append(accuracy)
    print("Accuracy of the Model is %.3f" %(sum(resultscore)/float(len(resultscore))))
    print('Jobs Ends!!')


