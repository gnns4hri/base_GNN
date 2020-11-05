import sys
import math
import pickle
from os import listdir
from os.path import isfile, join

data_path = '.'
if len(sys.argv) > 1:
    data_path = sys.argv[1]


filenames = [ data_path+'/'+f for f in listdir(data_path) if isfile(join(data_path, f)) ]

best_score = 1000000000
best_data = None
best_file = None

for f in filenames:
    if f.endswith('loss'):
        data = pickle.load(open(f, 'rb'))
        if data[0] < best_score:
            best_score = data[0]
            best_data = data
            best_file = f
        print (f, data)
print('===========================')
print('Squared', data[0], 0.015)
print('Non-squared', math.sqrt(data[0]), math.sqrt(0.015))
print(best_file, best_data)
prms = best_file[:-4] + 'prms'
data = pickle.load(open(prms, 'rb'))
print('layers:', data[1])
print('feats:', data[2])
print('num_hidden:', data[3])
print('classess', data[4])
print('heads', data[5])
print('drop', data[7])
print('att_drop', data[8])
print('alpha', data[9])
print('residual', data[10])

