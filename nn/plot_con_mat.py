import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt


conf_arr = np.load("con_mat_v3.npy")
conf_arr = np.delete(conf_arr, [0,4,8], 0)
conf_arr = np.delete(conf_arr, [0,4,8], 1)

n_sum = np.sum(conf_arr, axis = 1)
n_sum = n_sum.reshape((n_sum.shape[0],1))
n_sum = np.matlib.repmat(n_sum, 1, conf_arr.shape[1])
n_sum = n_sum.astype(np.float32)
conf_arr = conf_arr / n_sum
conf_arr = np.around(conf_arr, decimals=2)
label_dict = {
    'nba_neg':'0',
    'nba_dunk':'1',
    'nba_jumpshot':'2',
    'nba_layup':'3',
    'tennis_neg':'4',
    'tennis_forehand':'5',
    'tennis_backhand':'6',
    'tennis_serve':'7',
    'baseball_neg':'8',
    'baseball_hit':'9',
    'baseball_pitch':'10',
    'baseball_stolen_base':'11'
}

norm_conf = []
for i in conf_arr:
    a = 0
    tmp_arr = []
    a = sum(i, 0)
    for j in i:
        tmp_arr.append(float(j)/float(a))
    norm_conf.append(tmp_arr)

fig = plt.figure()
plt.clf()
ax = fig.add_subplot(111)
ax.set_aspect(1)
res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet, 
                interpolation='nearest')

width, height = conf_arr.shape

for x in xrange(width):
    for y in xrange(height):
        ax.annotate(str(conf_arr[x][y]), xy=(y, x), 
                    horizontalalignment='center',
                    verticalalignment='center')

cb = fig.colorbar(res)
# alphabet = ['bd', 'bj','bl','tf','tb','ts','bh','bp','bsb']
alphabet = ['bd', 'bj','bl','tf','tb','ts', 'bh','bp','bsb']
plt.xticks(range(width), alphabet)
plt.yticks(range(height), alphabet)
# plt.show()
plt.savefig('confusion_matrix_v3_without_neg.png', format='png')
