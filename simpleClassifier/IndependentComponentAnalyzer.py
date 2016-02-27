__author__ = 'haohanwang'

from utility import loadData
import numpy as np

from sklearn.decomposition import FastICA

train, dev, test = loadData.load_data()

for count in [20, 40, 60, 80, 100]:
    pca = FastICA(n_components=count, whiten=True)
    new_train = pca.fit_transform(np.array(train[0]))
    new_dev = pca.fit_transform(np.array(dev[0]))
    new_test = pca.fit_transform(np.array(test[0]))

    f = open('../extractedInformation/ica'+str(count)+'/train.csv', 'w')
    for i in range(len(new_train)):
        n = [str(k) for k in new_train[i]]
        f.writelines(','.join(n)+','+str(train[1][i])+'\n')
    f.close()

    f = open('../extractedInformation/ica'+str(count)+'/dev.csv', 'w')
    for i in range(len(new_dev)):
        n = [str(k) for k in new_dev[i]]
        f.writelines(','.join(n)+','+str(dev[1][i])+'\n')
    f.close()

    f = open('../extractedInformation/ica'+str(count)+'/test.csv', 'w')
    for i in range(len(new_test)):
        n = [str(k) for k in new_test[i]]
        f.writelines(','.join(n)+','+str(test[1][i])+'\n')
    f.close()