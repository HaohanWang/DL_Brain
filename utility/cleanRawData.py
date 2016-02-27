
import sys
import os
import numpy as np

p = '../rawData/'
count = 0
for r1, d1, f1 in os.walk(p):
    for ds in d1:
        count += 1
        for i in range(10):
            rd = np.loadtxt(p + ds + '/'+ str(i)+'.txt', delimiter=',')
            data = rd[:,3:]
            fileName = '../data/subj_'+str(count)+'_story_'+str(i+1)+'.csv'
            np.savetxt(fileName, data, delimiter=',')

