__author__ = 'haohanwang'


import pickle
import numpy as np
import numpy
from utility import loadData

def gm(v, num):
    l = []
    for i in range(num):
        l.append(v)
    return np.mat(l)

params = pickle.load(open('../model/sda80.pkl'))

m = []
for param in params:
    m.append(param.get_value(True))

l = len(params)/2
train, dev, test = loadData.load_data()

new_train = np.mat(train[0])
for i in range(l):
    new_train = new_train*np.mat(m[i])
new_train = new_train/sum(sum(new_train))

new_dev = np.mat(dev[0])
for i in range(l):
    new_dev = new_dev*np.mat(m[i])
new_dev = new_dev/sum(sum(new_dev))

new_test = np.mat(test[0])
for i in range(l):
    new_test = new_test*np.mat(m[i])
new_test = new_test/sum(sum(new_test))

print new_train.shape
train_label = np.reshape(np.array(train[1]), (20000,1))
dev_label = np.reshape(np.array(dev[1]), (20000,1))
test_label = np.reshape(np.array(test[1]), (20000,1))

new_train = np.append(new_train, train_label, 1)
new_dev = np.append(new_dev, dev_label, 1)
new_test = np.append(new_test, test_label, 1)

numpy.savetxt('../extractedInformation/sda80/train.csv', new_train, delimiter=",")
numpy.savetxt("../extractedInformation/sda80/dev.csv", new_dev, delimiter=",")
numpy.savetxt('../extractedInformation/sda80/test.csv', new_test, delimiter=",")
