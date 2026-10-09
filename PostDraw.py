from AddMat import GetS2ME
from DrawAll import PoltCountor
import numpy as np
import h5py

ME = np.load('NMMmyarray.npy', allow_pickle=True)

with h5py.File(f'NMM_result.h5', 'r') as f:
    MPE = f['MPE'][:]
    CE = f['CE'][:]
    nodes = f['nodes'][:]
    U_total = f['U'][:]

MPUT, us, _ = GetS2ME(nodes, ME, U_total, [])

UTRAIN = np.load('NNnu.npy')
UTRAIN_ALL = np.squeeze(UTRAIN.reshape(-1,1), axis=None)
MPU, us, _ = GetS2ME(nodes, ME, UTRAIN_ALL, [])

PoltCountor(ME, MPU[:,1], 'NN')

error = np.abs(MPUT[:, 1] - MPU[:, 1])
PoltCountor(ME, error, 'ERROR')

