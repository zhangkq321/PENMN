import numpy as np
from ShapeF import generate_structured_quad_mesh
from CutMesh import CutMEMesh
from AddMat import AddMat, GetS2ME
from DrawAll import PlotMesh, PoltCountor
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve  #稀疏矩阵求解器
from GetCrack import CalCrack
import h5py

def savedata(ME, nodes, MPE, Dnew, Gravity, BC, material, step):
    # engry = 0.5*Dnew.T @ K @ Dnew - F.T @ Dnew
    # print(engry)

    # 保存数据用于训练与对比
    CE = np.zeros((len(ME),3))
    for i, me in enumerate(ME):
        area_val = me.Area[0] if hasattr(me.Area, '__len__') and len(me.Area) == 1 else me.Area
        CE[i] = np.array([area_val,me.gxyz[0],me.gxyz[1]])

    with h5py.File(f'NMM_result_{step}.h5', 'w') as f:
        f.create_dataset('nodes', data=nodes)
        f.create_dataset('MPE', data=MPE)
        f.create_dataset('CE', data=CE)
        f.create_dataset('U', data=Dnew)
        f.create_dataset('gravity', data=Gravity)
        f.create_dataset('bbcc', data=BC)
        f.create_dataset('mat', data=material)
        f.create_dataset('CrackE2N', data=CrackE2N)
        f.create_dataset('CrackMat', data=CrackMat)
        f.create_dataset('crack', data=crack)

    np.save(f'NMMmyarray_{step}.npy', ME)

polygon = [
    [0, 0],
    [1, 0],
    [1, 2],
    [0, 2]
]

crack = np.array([[0,1,0.25,1]])

meshsize = 33 # 13 # 33 

material = np.array([80e9, 0.3, 100]) # E, v, den

# 约束面 xy->01,minmax->01,x/y/xy->012,val
BC = np.array([[1, 0, 1, -0.01],[1, 0, 0, 0],
               [1, 1, 1, 0.01],[1,1,0,0]])

# Gravity = np.array([0, 10*material[2]])
Gravity = np.array([0, 0])

# 荷载点 x,y,fx,fy
NC = np.array([])

# 测量点 x, y
MP = np.array([])

sigm_t = 0.001
nsteps = 25 # 10 #25
###################################################
# 前处理
# 生成网格
nodes_origin, elements, para = generate_structured_quad_mesh(polygon, meshsize)

# 生成流形单元
nodes, ME, BCdict, MPL, MPE, CrackE2N, CrackMat = CutMEMesh(nodes_origin, elements, crack, material, BC, NC, MP, None, para)

# 网格图
PlotMesh(ME)

cn = len(CrackE2N)
import copy
BCdict_new = copy.deepcopy(BCdict)

for step in range(nsteps):
    print(f"{step+1}-th time step")

    for key, bbcc in BCdict_new.items():
        bbcc['val'] = (1+step) * BCdict[key]['val']

    # 静力矩阵
    K, F = AddMat(nodes, ME, BCdict_new, Gravity)

    # 求解
    K = csr_matrix(K)
    Dnew = spsolve(K, F) 

    savedata(ME, nodes, MPE, Dnew, Gravity, BC, material, step)

    cno = cn
    nodes, MPE, ME, CrackE2N, CrackMat, cn, Dnew = CalCrack(crack, material, nodes, MPE, ME, Dnew, sigm_t, CrackE2N, CrackMat, cn)
    # if cn > cno:
    #     # 后处理
    #     uu, us, MPu = GetS2ME(nodes, ME, Dnew, MPL)
    #     sigma_x = us[:, 0]
    #     sigma_y = us[:, 1]
    #     tau_xy = us[:, 2]
    #     # von Mises应力公式: sqrt(σ_x² - σ_x·σ_y + σ_y² + 3·τ_xy²)
    #     von_mises = np.sqrt(sigma_x**2 - sigma_x*sigma_y + sigma_y**2 + 3*tau_xy**2)
    #     PoltCountor(ME, von_mises, step)


# 后处理
# uu, us, MPu = GetS2ME(nodes, ME, U_total, MPL)

# 云图
# PoltCountor(ME, uu[:,0], 'NMM')
