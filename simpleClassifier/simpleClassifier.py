__author__ = 'haohanwang'

from utility import loadData
import numpy as np

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

train, dev, test = loadData.load_data()

for n in [20, 40, 60, 80, 100]:
    train, dev, test = loadData.load_extracted(folderName='fs'+str(n)+'/')
    classifier = SVC()
    classifier.fit(train[0],train[1])
    print '====================='
    r = classifier.predict(dev[0])
    print accuracy_score(dev[1],r)
    r = classifier.predict(test[0])
    print accuracy_score(test[1],r)
    print '====================='

# train, dev, test = loadData.load_data()
#
# trData = np.array(train[0])
# trLabel = train[1]
# teData = np.array(test[0])
# teLabel = test[1]
#
# print trData.shape
# print teData.shape
#
# trData = np.sum(trData, axis=1).reshape([20000, 1])
# teData = np.sum(teData, axis=1).reshape([20000, 1])
#
# print trData.shape
# print teData.shape
#
# classifier = GaussianNB()
# classifier.fit(trData, trLabel)
# r = classifier.predict(teData)
# print accuracy_score(teLabel, r)