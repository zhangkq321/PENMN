import numpy as np

def lagrange_basis(coord):
    """
    计算4节点矩形单元的形函数值和导数
    参数:
        coord: 局部坐标 [xi, eta]
    返回:
        N: 形函数向量 (4,)
        dNdxi: 形函数对局部坐标的导数矩阵 (4, 2)
    """
    xi = coord[0]
    eta = coord[1]
    
    # 形函数向量
    N = 0.25 * np.array([
        (1 - xi) * (1 - eta),  # N1
        (1 + xi) * (1 - eta),  # N2
        (1 + xi) * (1 + eta),  # N3
        (1 - xi) * (1 + eta)   # N4
    ])
    
    # 形函数对局部坐标的导数矩阵 [dN/dxi, dN/deta]
    dNdxi = 0.25 * np.array([
        [-(1 - eta), -(1 - xi)],    # dN1/dxi, dN1/deta
        [ 1 - eta , -(1 + xi)],    # dN2/dxi, dN2/deta
        [ 1 + eta ,   1 + xi ],    # dN3/dxi, dN3/deta
        [-(1 + eta),   1 - xi ]     # dN4/dxi, dN4/deta
    ])
    
    return N, dNdxi

def NFunction2D(gxy, nodes):
    """
    在全局坐标下计算三角形单元的重心坐标形函数
    """
    # 计算单元边界坐标
    MaxXPP = np.max(nodes[:, 0])
    MinXPP = np.min(nodes[:, 0])
    MaxYPP = np.max(nodes[:, 1])
    MinYPP = np.min(nodes[:, 1])
    
    # 计算局部坐标（从全局坐标变换到[-1,1]标准单元）
    LocalCoord = np.zeros(2)
    LocalCoord[0] = -1 + (gxy[0] - MinXPP) / (MaxXPP - MinXPP) * 2
    LocalCoord[1] = -1 + (gxy[1] - MinYPP) / (MaxYPP - MinYPP) * 2
    
    # 计算各节点的形函数值
    N1D, _ = lagrange_basis(LocalCoord)
    
    ndim = 2  # 空间维数
    
    # 构建形函数矩阵，使用Kronecker积
    N = np.kron(N1D.reshape(-1, 1), np.eye(ndim))
    
    return N.T

def strain_matrix(gxy, nodes):

    # 计算单元边界坐标
    MaxXPP = np.max(nodes[:, 0])
    MinXPP = np.min(nodes[:, 0])
    MaxYPP = np.max(nodes[:, 1])
    MinYPP = np.min(nodes[:, 1])
    
    # 计算局部坐标（从全局坐标变换到[-1,1]标准单元）
    LocalCoord = np.zeros(2)
    LocalCoord[0] = -1 + (gxy[0] - MinXPP) / (MaxXPP - MinXPP) * 2
    LocalCoord[1] = -1 + (gxy[1] - MinYPP) / (MaxYPP - MinYPP) * 2
    
    # 计算形函数导数
    _, dNdxi = lagrange_basis(LocalCoord)
    
    # 雅可比矩阵 J = d(x,y)/d(xi,eta)
    J = nodes.T @ dNdxi  # 等价于 xPPs' * dNdxi (MATLAB)
    
    # 计算全局坐标下的形函数导数 dN/dx = (dN/dxi) * inv(J)
    # 注意：MATLAB中的 dNdx = dNdxi / J 等价于 dNdx = dNdxi * inv(J)
    try:
        J_inv = np.linalg.inv(J)
        dNdx = dNdxi @ J_inv
    except np.linalg.LinAlgError:
        raise ValueError("雅可比矩阵奇异，无法计算逆矩阵")
    
    # 初始化B矩阵 [εx, εy, γxy]^T = B * u
    n_nodes = nodes.shape[0]
    B = np.zeros((3, 2 * n_nodes))
    
    # 构建B矩阵
    for inode in range(n_nodes):
        # 每个节点的位移分量在B矩阵中的位置
        # 节点inode的x位移对应列 2*inode
        # 节点inode的y位移对应列 2*inode + 1
        B[:, 2 * inode : 2 * inode + 2] = np.array([
            [dNdx[inode, 0], 0],                 # ∂Ni/∂x, 0
            [0, dNdx[inode, 1]],                 # 0, ∂Ni/∂y  
            [dNdx[inode, 1], dNdx[inode, 0]]     # ∂Ni/∂y, ∂Ni/∂x
        ])
    
    return B

def Elastic(E, poisson):

    # Plane Strain state

    Daux1 = ( E * (1. - poisson) )/( ( 1.+poisson )*( 1.-2.*poisson ) )
    Daux2 = poisson/( 1.-poisson )
    Daux3 = ( 1.-2.*poisson)/( 2.*( 1.-poisson ))

    D = np.array([ [   1. , Daux2,   0. ],
                   [ Daux2,   1. ,   0. ],
                   [   0. ,   0. , Daux3] ])

    D = Daux1 * D
    
    return D

###############################
def generate_structured_quad_mesh(polygon, meshsize):
    """
    生成结构化四边形网格
    
    参数:
    x_min, x_max : x轴范围
    y_min, y_max : y轴范围
    nx : x方向单元数量
    ny : y方向单元数量
    
    返回:
    nodes : (n_nodes, 2) 数组, 每行 [x, y] 坐标
    elements : (n_elements, 4) 数组, 每行 [n1, n2, n3, n4] 节点索引
    """
    polygon = np.array(polygon)
    x_min = np.min(polygon[:,0])
    x_max = np.max(polygon[:,0])
    y_min = np.min(polygon[:,1])
    y_max = np.max(polygon[:,1])
    nx = ny = meshsize
    
    para = np.array([x_min, x_max, y_min, y_max, nx, ny])

    # 1. 生成节点坐标
    x = np.linspace(x_min, x_max, nx + 1)
    y = np.linspace(y_min, y_max, ny + 1)
    xx, yy = np.meshgrid(x, y, indexing='xy')
    nodes = np.column_stack((xx.ravel(), yy.ravel()))
    
    # 2. 生成单元连接关系
    elements = np.zeros((nx * ny, 4), dtype=int)
    elem_idx = 0
    for j in range(ny):      # 遍历y方向（行）
        for i in range(nx):  # 遍历x方向（列）
            # 计算当前单元左下角节点的全局索引
            n_start = j * (nx + 1) + i
            # 逆时针顺序: 左下 → 右下 → 右上 → 左上
            elements[elem_idx] = [
                n_start,          # 左下 (i, j)
                n_start + 1,      # 右下 (i+1, j)
                n_start + nx + 2, # 右上 (i+1, j+1)
                n_start + nx + 1  # 左上 (i, j+1)
            ]
            elem_idx += 1
    
    return nodes, elements, para
