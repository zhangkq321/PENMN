import numpy as np
from ShapeF import strain_matrix, Elastic, NFunction2D

def AddMat(nodes, ME, BCdict, Gravity):
    # 组装刚度矩阵
    nn = nodes.shape[0]
    K = np.zeros((2*nn, 2*nn))
    F = np.zeros((2*nn, 1))

    for me in ME:
        area = me.Area
        gxy = me.gxyz
        
        nodecoord = nodes[me.dof,:]
        B = strain_matrix(gxy, nodecoord)
        D = Elastic(me.material[0], me.material[1])
        Ke = B.T @ D @ B * area

        DOF = np.dstack((2 * me.dof, 2 * me.dof + 1)).ravel()
        K[DOF[:, None], DOF] += Ke

        # 形函数矩阵
        N = NFunction2D(gxy, nodecoord)
        F[DOF[:, None], 0] += N.T @ Gravity[:,None] * area
        
        if me.PointLoad is not None:
            cxy = me.PointLoad[0:2]
            N = NFunction2D(cxy, nodecoord)
            load = me.PointLoad[2:4][:,None]
            F[DOF[:, None], 0] += N.T @ load

    # 约束
    penalty = me.material[0] * 1e6
    for bbcc in BCdict.values():
        row_indices = bbcc['Index']
        dofs = bbcc['dof']
        val = bbcc['val']
        for i in row_indices:
            for d in dofs:
                K[2*i + d, 2*i + d] *= penalty
                F[2*i + d, 0] = val * K[2*i + d, 2*i + d] 

    return K, F

# 插值到流形
def GetS2ME(nodes, ME, U, MPL):

    uu = np.zeros((len(ME), 2))
    us = np.zeros((len(ME), 3))

    index = 0
    for me in ME:
        gxy = me.gxyz
        nodecoord = nodes[me.dof,:]
        N = NFunction2D(gxy, nodecoord)   
        B = strain_matrix(gxy, nodecoord)   

        DOF = np.dstack((2 * me.dof, 2 * me.dof + 1)).ravel()  

        D = Elastic(me.material[0], me.material[1])

        uu[index,:] = N @ U[DOF]
        us[index,:] = D @ B @ U[DOF]

        # for i, nod in enumerate(me.points):
        #     N = NFunction2D(nod, nodecoord)  
        #     me.points[i,:] += N @ U[DOF]

        index += 1
    
    if len(MPL) != 0:
        MPu = np.zeros((MPL.shape[0], 2))
        for index, mp in enumerate(MPL):
            me = ME[int(mp[2])]
            gxy = me.gxyz
            nodecoord = nodes[me.dof,:]
            N = NFunction2D(gxy, nodecoord)   
            DOF = np.dstack((2 * me.dof, 2 * me.dof + 1)).ravel()  
            MPu[index,:] = N @ U[DOF]
    else:
        MPu = []

    return uu, us, MPu