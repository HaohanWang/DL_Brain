f = open('../data/label.csv', 'w')

for i in range(10):
    for j in range(10):
        line = 'subj_'+str(i+1)+'_story_'+str(j+1)+','
        line = line + '1\n' if j < 5 else line + '0\n'
        f.writelines(line)
f.close()