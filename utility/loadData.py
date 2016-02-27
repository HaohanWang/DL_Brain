__author__ = 'haohanwang'

import numpy as np

def load_data():
    print 'Loading Data...',
    datatext = np.loadtxt(open("../geneExpressionData/ge.csv","rb"),delimiter=",")
    train = []
    dev = []
    test = []
    data = {}
    for i in range(len(datatext[:,0])):
        k = datatext[i,0]
        data[k] = datatext[i,1:]
    print '...',
    text = [line.strip() for line in open('../labels/train.txt')]
    trainLabel = []
    for line in text:
        items = line.split()
        p1 = int(items[0])
        p2 = int(items[1])
        train.append(data[p1]*data[p2])
        trainLabel.append(int(items[2]))
    print '...',
    text = [line.strip() for line in open('../labels/dev.txt')]
    devLabel = []
    for line in text:
        items = line.split()
        p1 = int(items[0])
        p2 = int(items[1])
        dev.append(data[p1]*data[p2])
        devLabel.append(int(items[2]))
    print '...',
    text = [line.strip() for line in open('../labels/test.txt')]
    testLabel = []
    for line in text:
        items = line.split()
        p1 = int(items[0])
        p2 = int(items[1])
        test.append(data[p1]*data[p2])
        testLabel.append(int(items[2]))
    go_trainLabel = [int(line.strip()) for line in open('../labels/train_go.txt')]
    go_devLabel = [int(line.strip()) for line in open('../labels/dev_go.txt')]
    go_testLabel = [int(line.strip()) for line in open('../labels/test_go.txt')]

    mf_train_label = np.loadtxt('../labels/MF_train_int.txt', delimiter='\t').astype(int)
    cc_train_label = np.loadtxt('../labels/CC_train_int.txt', delimiter='\t').astype(int)
    bp_train_label = np.loadtxt('../labels/BP_train_int.txt', delimiter='\t').astype(int)


    print 'Done'
    return (train,trainLabel, go_trainLabel, mf_train_label, cc_train_label, bp_train_label), (dev,devLabel, go_devLabel),(test,testLabel, go_testLabel)

def load_extracted(folderName):
    result = []
    fileName = ['train', 'dev', 'test']
    for fn in fileName:
        data = np.loadtxt(open('../extractedInformation/'+folderName+fn+'.csv'), delimiter=',')
        result.append((data[:,:-1], data[:,-1]))
    return result
