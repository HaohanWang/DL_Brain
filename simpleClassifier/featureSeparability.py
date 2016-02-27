__author__ = 'haohanwang'

from utility import loadData
import numpy as np

train, dev, test = loadData.load_data()

old_train = np.array(train[0])
old_test = np.array(test[0])
old_dev = np.array(dev[0])

mean1 = np.mean(old_train[:10000,:], axis=0)
std1 = np.std(old_train[:10000,:], axis=0)
mean2 = np.mean(old_train[10000:,:], axis=0)
std2 = np.std(old_train[:10000,:], axis=0)

fs = np.abs(mean1 - mean2)/(std1 + std2)

inds = np.argsort(fs)

l = inds.shape[0]

for count in [20, 40, 60, 80, 100]:
    indice = inds[l-count:]
    new_train = old_train[:, indice]
    new_dev = old_dev[:, indice]
    new_test = old_test[:, indice]
    f = open('../extractedInformation/fs'+str(count)+'/train.csv', 'w')
    for i in range(len(new_train)):
        n = [str(k) for k in new_train[i]]
        f.writelines(','.join(n)+','+str(train[1][i])+'\n')
    f.close()

    f = open('../extractedInformation/fs'+str(count)+'/dev.csv', 'w')
    for i in range(len(new_dev)):
        n = [str(k) for k in new_dev[i]]
        f.writelines(','.join(n)+','+str(dev[1][i])+'\n')
    f.close()

    f = open('../extractedInformation/fs'+str(count)+'/test.csv', 'w')
    for i in range(len(new_test)):
        n = [str(k) for k in new_test[i]]
        f.writelines(','.join(n)+','+str(test[1][i])+'\n')
    f.close()
