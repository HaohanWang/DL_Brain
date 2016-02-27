__author__ = 'haohanwang'

from utility import loadData
import numpy as np

from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest

train, dev, test = loadData.load_data()

old_train = np.array(train[0])
old_test = np.array(test[0])
old_dev = np.array(dev[0])
m = min([np.min(old_train), np.min(old_test), np.min(old_dev)])
old_train -= m
old_test -= m
old_dev -= m

for count in [20, 40, 60, 80, 100]:
    selector = SelectKBest(chi2, k=count)
    # print np.min(old_train)
    # print np.min(old_test)
    # print np.min(old_dev)
    selector.fit(old_train, train[1])
    new_train = selector.transform(old_train)
    new_dev = selector.transform(old_dev)
    new_test = selector.transform(old_test)
    f = open('../extractedInformation/chi'+str(count)+'/train.csv', 'w')
    for i in range(len(new_train)):
        n = [str(k) for k in new_train[i]]
        f.writelines(','.join(n)+','+str(train[1][i])+'\n')
    f.close()

    f = open('../extractedInformation/chi'+str(count)+'/dev.csv', 'w')
    for i in range(len(new_dev)):
        n = [str(k) for k in new_dev[i]]
        f.writelines(','.join(n)+','+str(dev[1][i])+'\n')
    f.close()

    f = open('../extractedInformation/chi'+str(count)+'/test.csv', 'w')
    for i in range(len(new_test)):
        n = [str(k) for k in new_test[i]]
        f.writelines(','.join(n)+','+str(test[1][i])+'\n')
    f.close()