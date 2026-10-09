import numpy as np
from ShapeF import generate_structured_quad_mesh
from CutMesh import CutMEMesh
from AddMat import AddMat, GetS2ME
from DrawAll import PlotMesh, PoltCountor
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve  #稀疏矩阵求解器
import h5py

polygon = [
    [0, 0],
    [2, 0],
    [2, 2],
    [0, 2]
]

angle = np.deg2rad(75)
crack = np.array([[1+np.cos(angle)/2,1+np.sin(angle)/2,1-np.cos(angle)/2,1-np.sin(angle)/2],
                  [1-np.cos(angle)/2,1+np.sin(angle)/2,1+np.cos(angle)/2,1-np.sin(angle)/2]])

meshsize = 30 # 30

material = np.array([1, 0.2, 200]) # E, v, den

# 约束面 xy->01,minmax->01,x/y/xy->012,val
BC = np.array([[1, 0, 1, 0],
               [1, 1, 1, 1],
               [0, 0, 0, 0],
               [0, 1, 0, 0]])

# Gravity = np.array([0, 10*material[2]])
Gravity = np.array([0, 0])

# 荷载点 x,y,fx,fy
NC = np.array([])

# 测量点 x, y
MP = np.array([])

###################################################
# 前处理
# 生成网格
nodes_origin, elements, para = generate_structured_quad_mesh(polygon, meshsize)

# 生成流形单元
nodes, ME, BCdict, MPL, MPE, CrackE2N, CrackMat = CutMEMesh(nodes_origin, elements, crack, material, BC, NC, MP, None, para)

# 网格图
# PlotMesh(ME)

# 静力矩阵
K, F = AddMat(nodes, ME, BCdict, Gravity)

# 求解
K = csr_matrix(K)
U_total = spsolve(K, F) 

engry = 0.5*U_total.T @ K @ U_total - F.T @ U_total
print(engry)

# 保存数据用于训练与对比
CE = np.zeros((len(ME),3))
for i, me in enumerate(ME):
    CE[i] = np.array([me.Area,me.gxyz[0],me.gxyz[1]])

with h5py.File('NMM_result.h5', 'w') as f:
    f.create_dataset('nodes', data=nodes)
    f.create_dataset('MPE', data=MPE)
    f.create_dataset('CE', data=CE)
    f.create_dataset('U', data=U_total)
    f.create_dataset('gravity', data=Gravity)
    f.create_dataset('bbcc', data=BC)
    f.create_dataset('mat', data=material)

np.save('NMMmyarray.npy', ME)

# 后处理
# uu, us, MPu = GetS2ME(nodes, ME, U_total, MPL)

# # 云图
# PoltCountor(ME, uu[:,1], 'NMM')
