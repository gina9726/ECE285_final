import h5py
import numpy as np
import pdb

f = h5py.File("data_img_fc7.h5", "r")
train = np.array(f['images_train'])
test = np.array(f['images_test'])
f.close()

N = train.shape[0]
d = 5
p = int(np.round(N/d))
print('N=%d, d=%d, p=%d' % (N, d, p))
part = zip(range(0, N, p), range(p, N+1, p))
for i in range(len(part)):
    train_part = train[part[i][0]:part[i][1]]
    print('%d ~ %d are saved to img_fc7_train_part%d.npy' % (part[i][0], part[i][1], i+1))
    np.save('img_fc7_train_part%d.npy' % (i+1), train_part)

np.save('img_fc7_test.npy', test)
