from AddMat import GetS2ME
from DrawAll import PoltCountor
import numpy as np
import h5py

ME = np.load('NNmyarray_24.npy', allow_pickle=True)

with h5py.File(f'NN_result_24.h5', 'r') as f:
    nodes = f['nodes'][:]
    U_total = f['U'][:]

MPUT, us, _ = GetS2ME(nodes, ME, U_total, [])

# UTRAIN = np.load('nu.npy')
# UTRAIN_ALL = np.squeeze(UTRAIN.reshape(-1,1), axis=None)
# MPU, us, _ = GetS2ME(nodes, ME, UTRAIN_ALL, [])

PoltCountor(ME, MPUT[:,1], 'NN')

# error = np.abs(MPUT[:, 1] - MPU[:, 1])
# PoltCountor(ME, error, 'ERROR')

