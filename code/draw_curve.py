import numpy as np
import matplotlib.pyplot as plt
import sys, os
import pdb

closs = []
acc = []
lr = []

with open(sys.argv[1]) as f:
    for line in f:
        if 'Class' in line:
            closs.append(float(line.rstrip('\n').split(': ')[1]))
        elif 'Acc' in line:
            acc.append(float(line.rstrip('%\n').split(': ')[1]))
        elif 'current lr' in line:
            lr.append(float(line.rstrip('\n').split('= ')[1]))

lr = np.asarray(lr)
closs = np.asarray(closs)
acc = np.asarray(acc)

ep = np.arange(len(closs))*50
plt.switch_backend('agg')
fig = plt.figure()

plt.plot(ep, closs/max(closs), 'b', label='closs')
plt.plot(ep, acc/100, 'r', label='accuracy')
plt.plot(ep, lr*1000, 'g', label='lr')

plt.legend(loc='upper left')
plt.xlabel('Epoch')
plt.title('Training curve')

fig.savefig(os.path.join('curve', sys.argv[1].split('/')[-1].replace('log', 'png')))
