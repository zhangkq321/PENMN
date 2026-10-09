import numpy as np
from collections import defaultdict
from itertools import chain
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import split, unary_union
import pandas as pd

# 寻找每一个节点所在的单元
def get_elements_per_node(elements, target_node_indices):
    # 将elements转换为NumPy数组
    elements_array = np.asarray(elements)
    num_elements = elements_array.shape[0]

    # 构建节点到单元索引的映射
    node_to_elements = defaultdict(list)
    for elem_idx in range(num_elements):
        for node in elements_array[elem_idx]:
            node_to_elements[node].append(elem_idx)

    # 将列表转换为NumPy数组
    for node in node_to_elements:
        node_to_elements[node] = np.array(node_to_elements[node], dtype=np.int64)

    # 为每个目标节点提取结果
    result = []
    for node in target_node_indices:
        result.append(node_to_elements.get(node, np.array([], dtype=np.int64)))

    return result

# 线段交点计算
def is_line_intersect_joints(p1, p2, joints):

    def segments_intersect(a1, a2, b1, b2):
        def on_segment(p, q, r):
            if (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and
                min(p[1], r[1]) <= q[1] <= max(p[1], r[1])):
                return True
            return False

        def orientation(p, q, r):
            val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
            if val == 0:
                return 0
            return 1 if val > 0 else 2

        o1 = orientation(a1, a2, b1)
        o2 = orientation(a1, a2, b2)
        o3 = orientation(b1, b2, a1)
        o4 = orientation(b1, b2, a2)

        if o1 != o2 and o3 != o4:
            return True

        if o1 == 0 and on_segment(a1, b1, a2):
            return True
        if o2 == 0 and on_segment(a1, b2, a2):
            return True
        if o3 == 0 and on_segment(b1, a1, b2):
            return True
        if o4 == 0 and on_segment(b1, a2, b2):
            return True

        return False

    for joint in joints:
        j_p1 = np.array([joint[0], joint[1]])
        j_p2 = np.array([joint[2], joint[3]])
        if segments_intersect(p1, p2, j_p1, j_p2):
            return True
    return False

# 离散线段
def discretize_segment(p1, p2, dd):
    """
    将两点之间的线段按固定距离 dd 离散化为有限个点。
    
    参数:
    - p1, p2: 二维点，形状为 (2,) 的 NumPy 数组、列表或元组。
    - dd: 每个点之间的固定距离。
    
    返回:
    - points: 形状为 (n, 2) 的 NumPy 数组，表示线段上的离散点。
    """
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    
    delta = p2 - p1
    L = np.linalg.norm(delta)
    
    # 处理两点重合的情况
    if np.isclose(L, 0):
        return p1.reshape(1, -1)
    
    unit_vec = delta / L  # 单位方向向量
    
    # 生成 t 值（沿线段的距离）
    t_values = np.arange(0, L + 1e-8, dd)
    t_values = t_values[t_values <= L]  # 确保不超过线段长度

    # 计算每个点
    points = p1 + t_values.reshape(-1, 1) * unit_vec

    # 确保终点也被包含（如果最后一点不是终点）
    if not np.allclose(points[-1], p2, atol=1e-8):
        points = np.vstack([points, p2])

    return points

def find_intersecting_cells(crack, para, dd):
    """
    快速查找裂缝穿过的所有单元（利用结构化网格特性）
    
    参数:
    crack : 裂缝线段列表，每个元素为[x1, y1, x2, y2]
    x_min, x_max, y_min, y_max : 网格边界
    nx, ny : x/y方向单元数量
    dd : 离散化步长
    
    返回:
    result : 数组，每行 [裂缝段索引, 单元索引]
    """
    x_min, x_max, y_min, y_max, nx, ny = para

    # 1. 预计算网格坐标
    x_coords = np.linspace(x_min, x_max, nx + 1)
    y_coords = np.linspace(y_min, y_max, ny + 1)
    
    # 2. 预计算单元尺寸（用于边界处理）
    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny
    
    result = []
    
    # 3. 处理每条裂缝线段
    for seg_idx, seg in enumerate(crack):
        p1 = np.array([seg[0], seg[1]])
        p2 = np.array([seg[2], seg[3]])
        
        # 离散化线段
        points = discretize_segment(p1, p2, dd)
        
        # 存储当前线段经过的单元
        seg_cells = []
        
        # 4. 快速定位每个离散点所在的单元
        for pt in points:
            px, py = pt
            
            # 检查点是否在网格内
            if (px < x_min - 1e-6 or px > x_max + 1e-6 or 
                py < y_min - 1e-6 or py > y_max + 1e-6):
                continue
            
            # 处理边界点（防止越界）
            px = np.clip(px, x_min, x_max - 1e-9)
            py = np.clip(py, y_min, y_max - 1e-9)
            
            # 计算单元索引（核心优化）
            i = int((px - x_min) / dx)  # x方向单元索引
            j = int((py - y_min) / dy)  # y方向单元索引
            
            # 防止边界溢出
            i = min(max(i, 0), nx - 1)
            j = min(max(j, 0), ny - 1)
            
            # 计算全局单元索引（行优先）
            cell_idx = j * nx + i
            seg_cells.append(cell_idx)
        
        # 5. 去除重复单元（保持顺序）
        unique_cells = []
        seen = set()
        for cell in seg_cells:
            if cell not in seen:
                seen.add(cell)
                unique_cells.append(cell)
        
        # 6. 记录结果
        for cell in unique_cells:
            result.append([seg_idx, cell])
    
    return np.array(result)

def FindBC(BC, nodes):
    BCdict = {}
    # 处理约束
    for i, bbcc in enumerate(BC):
        col_data = nodes[:, int(bbcc[0])]
        MorM = int(bbcc[1])
        if MorM == 0:
            mm = col_data.min()
        elif MorM == 1:
            mm = col_data.max()
        # 节点索引
        row_indices = np.flatnonzero(col_data == mm)
        xyz = int(bbcc[2])
        if xyz == 0 or xyz == 1:
            dofs = [xyz]
        elif xyz == 2:
            dofs = [0, 1]

        BCdict[i] = {'Index':row_indices, 'dof':dofs, 'val':bbcc[3]}
        
    return BCdict

def remove_unused_nodes(nodes, elements, elements2):
    """
    删除未使用的节点，并更新 elements 和 elements2 的索引。
    """
    used_indices = np.unique(elements)
    new_nodes = nodes[used_indices]
    
    # 定义通用的索引映射函数
    def map_indices(old_elements):
        flat = old_elements.flatten()
        new_flat = np.searchsorted(used_indices, flat)
        return new_flat.reshape(old_elements.shape)
    
    # 注意elements2的一些节点可能已经被删除，这将导致其索引错误，但是存在这样节点的单元并不会被使用，因此这里可以忽略这一问题
    new_elements = map_indices(elements)
    new_elements2 = map_indices(elements2)

    return new_nodes, new_elements, new_elements2

def quad_area_shoelace(nodes, element):
    """
    使用鞋带公式计算四边形单元面积
    
    参数:
    nodes : (n_nodes, 2) 节点坐标数组
    element : (4,) 单元的4个节点索引（按逆时针顺序）
    
    返回:
    area : 单元面积（标量）
    """
    # 获取四边形的四个节点坐标
    coords = nodes[element]  # shape (4, 2)
    
    # 鞋带公式实现
    x = coords[:, 0]
    y = coords[:, 1]
    
    # 计算: 0.5 * |sum(x_i * y_{i+1} - x_{i+1} * y_i)|
    # 注意: 最后一个点连接到第一个点 (i+1 mod 4)
    area = 0.5 * abs(
        x[0]*y[1] + x[1]*y[2] + x[2]*y[3] + x[3]*y[0] -
        (y[0]*x[1] + y[1]*x[2] + y[2]*x[3] + y[3]*x[0])
    )
    
    return area

def quad_area_vectorized(nodes, elements):
    """
    向量化计算所有四边形单元的面积（高效批量处理）
    
    参数:
    nodes : (n_nodes, 2) 节点坐标数组
    elements : (n_elements, 4) 单元连接数组
    
    返回:
    areas : (n_elements,) 所有单元的面积数组
    """
    # 获取所有单元的节点坐标
    coords = nodes[elements]  # shape (n_elements, 4, 2)
    
    # 提取x和y坐标
    x = coords[..., 0]  # shape (n_elements, 4)
    y = coords[..., 1]  # shape (n_elements, 4)
    
    # 向量化鞋带公式
    # 计算 x_i * y_{i+1}
    x_i_y_ip1 = x * np.roll(y, -1, axis=1)
    # 计算 y_i * x_{i+1}
    y_i_x_ip1 = y * np.roll(x, -1, axis=1)
    
    # 求和并计算面积
    areas = 0.5 * np.abs(np.sum(x_i_y_ip1 - y_i_x_ip1, axis=1))
    
    return areas

def find_cells_for_points(points, para, 
                         return_valid_mask=False, return_local_coords=False):
    """
    高效查询结构化四边形网格中点所在的单元
    
    参数:
    points : (n, 2) NumPy数组，每行包含[x, y]坐标
    x_min, x_max, y_min, y_max : 网格边界
    nx, ny : x/y方向的单元数量
    return_valid_mask : 是否返回有效点掩码（在网格内的点）
    return_local_coords : 是否返回单元内的局部坐标 (xi, eta) ∈ [-1,1]²
    
    返回:
    cell_indices : (n,) 数组，单元索引（-1表示点在网格外）
    valid_mask : (n,) 布尔数组（仅当return_valid_mask=True时返回）
    local_coords : (n, 2) 局部坐标（仅当return_local_coords=True时返回）
    """
    x_min, x_max, y_min, y_max, nx, ny = para
    # 1. 预计算网格参数
    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny
    
    # 2. 初始化结果数组
    n_points = points.shape[0]
    cell_indices = np.full(n_points, -1, dtype=np.int32)
    
    # 3. 创建有效点掩码（在网格内的点）
    x = points[:, 0]
    y = points[:, 1]
    
    # 使用epsilon处理边界浮点精度问题
    eps = 1e-9
    valid_mask = (
        (x >= x_min - eps) & (x <= x_max + eps) &
        (y >= y_min - eps) & (y <= y_max + eps)
    )
    
    # 4. 处理有效点
    if np.any(valid_mask):
        xv = x[valid_mask]
        yv = y[valid_mask]
        
        # 处理边界点：将恰好在右/上边界的点映射到内部
        xv = np.clip(xv, x_min, x_max - eps)
        yv = np.clip(yv, y_min, y_max - eps)
        
        # 5. 核心计算：直接通过坐标计算单元索引
        i_indices = np.floor((xv - x_min) / dx).astype(np.int32)
        j_indices = np.floor((yv - y_min) / dy).astype(np.int32)
        
        # 边界保护（防止因浮点误差导致的越界）
        i_indices = np.clip(i_indices, 0, nx - 1)
        j_indices = np.clip(j_indices, 0, ny - 1)
        
        # 计算全局单元索引（行优先）
        cell_indices[valid_mask] = j_indices * nx + i_indices
        
        # 6. 计算局部坐标（如果需要）
        if return_local_coords:
            local_coords = np.zeros((n_points, 2), dtype=np.float64)
            xi = 2 * ((xv - x_min) / dx - i_indices) - 1  # [-1, 1]
            eta = 2 * ((yv - y_min) / dy - j_indices) - 1  # [-1, 1]
            local_coords[valid_mask, 0] = xi
            local_coords[valid_mask, 1] = eta
    
    # 7. 准备返回结果
    results = [cell_indices]
    
    if return_valid_mask:
        results.append(valid_mask)
    
    if return_local_coords:
        if not return_valid_mask:
            # 确保未请求valid_mask时也创建local_coords
            local_coords = np.zeros((n_points, 2), dtype=np.float64)
            local_coords[valid_mask, 0] = xi
            local_coords[valid_mask, 1] = eta
        results.append(local_coords)
    
    return results[0] if len(results) == 1 else tuple(results)

from dataclasses import dataclass
from typing import Optional

@dataclass
class Element:
    points: np.ndarray # 单元顶点
    gxyz: np.ndarray # 单元形心
    inxy: np.ndarray # 单元内部一点
    Area: float # 面积
    me2ele: int
    material: np.ndarray # 材料矩阵
    stress: np.ndarray # 单元应力
    dof: Optional[np.ndarray] = None # 星点索引
    PointLoad: Optional[np.ndarray] = None # 点荷载
    
# 主函数
def CutMEMesh(nodes, elements, crack, materials, BC, NC, MP, polyHole, para):

    if polyHole is not None:
        isHole = True
    else:
        isHole = False

    # 检测裂隙穿过的单元
    dd = min(para[1] - para[0], para[3] - para[2])/100
    result = find_intersecting_cells(crack, para, dd)

    # 建立E2C:{key(elementIndex):[crackIndex]}
    E2C = defaultdict(list)
    # 遍历每一行
    for row in result:
        key = row[1]   # 第 1 列作为键
        value = row[0] # 第 0 列作为值
        E2C[key].append(value)
    # 转换为普通字典
    E2C = dict(E2C)

    ii = 0
    CME = []
    ME = []
    CarckElementIndex = []
    for ele, ci in E2C.items():

        # 计算每一个单元ele被crack-ci切割后的多边形
        triangle = Polygon(nodes[elements[ele],:])
        lines = []
        for i in ci:
            lines.append(LineString(crack[i,:].reshape(2, 2)))
        # 合并切割线段（处理线段之间的交点）
        cut_lines = unary_union(lines)
        # 使用合并后的线段分割三角形
        result1 = split(triangle, cut_lines)
        # 提取分割后的多边形
        polygons = [p for p in result1.geoms if isinstance(p, Polygon)]
        # 过滤掉面积小于1e-15的多边形
        filtered_polygons = [p for p in polygons if p.area >= 1e-15]

        # 如果过滤后仅剩一个有效多边形，跳过当前循环
        if len(filtered_polygons) < 2:
            continue

        for i, poly in enumerate(polygons):
            
            # 计算多边形内部一点
            innerpoint = poly.representative_point()
            cx, cy = poly.centroid.x, poly.centroid.y

            x, y = poly.exterior.xy
            coords = np.column_stack((x, y))
            df = pd.DataFrame(coords)
            coords = df.round(6).drop_duplicates().values
            coords = np.flipud(coords)

            if isHole:
                pointhole = Point(np.array([innerpoint.x, innerpoint.y]))
                if pointhole.within(polyHole):
                    continue
            
            CarckElementIndex.append(ele)

            ME.append(Element(
                points = coords,
                gxyz = np.array([cx, cy]),
                inxy = np.array([innerpoint.x, innerpoint.y]),
                Area = poly.area,
                material = materials,
                stress = np.zeros((3)),
                me2ele = ele ))
            CME.append(ii)
            ii += 1

    # 被切割单元索引
    CarckElementIndex = np.array(CarckElementIndex)

    # 非切割单元矩阵
    all_indices = np.arange(elements.shape[0])
    not_in_mask = ~np.isin(all_indices, CarckElementIndex)
    NoCarckIndex = all_indices[not_in_mask]
    NoCarckElement = elements[NoCarckIndex]

    # 计算形心
    NoCarckCenterCoords = np.mean(nodes[NoCarckElement], axis=1)
    # 计算面积
    NoCarckEArea = quad_area_vectorized(nodes, NoCarckElement)

    NoCarckIndex_new = []
    for i, index in enumerate(NoCarckIndex):
        if isHole:
            pointhole = Point(NoCarckCenterCoords[i,:])
            if pointhole.within(polyHole):
                continue
        
        NoCarckIndex_new.append(index)

        ME.append(Element(
            points = nodes[elements[index],:],
            gxyz = NoCarckCenterCoords[i,:],
            inxy = NoCarckCenterCoords[i,:],
            Area = NoCarckEArea[i],
            material = materials,
            stress = np.zeros((3)),
            me2ele = index ))
        
    NoCarckIndex = NoCarckIndex_new
    # 构建MP单元矩阵
    indexall = np.hstack((CarckElementIndex, NoCarckIndex)).astype(int)
    MPE = elements[indexall]

    if isHole:
        nodes, MPE, elements = remove_unused_nodes(nodes, MPE, elements)

    # 未被切割直接返回
    if len(CarckElementIndex) == 0:
        # 分配自由度
        for i, me in enumerate(ME):
            me.dof = MPE[i]
  
        # 处理约束
        BCdict = FindBC(BC, nodes)
        # 寻找荷载点所在单元
        if len(NC) != 0:
            cell_indices = find_cells_for_points(NC[:,0:2], para)
            for i, index in enumerate(cell_indices):
                ME[index].PointLoad = NC[i,:]
            
        # 寻找测量点所在单元
        if len(MP) != 0:
            cell_indices = find_cells_for_points(MP, para)
            MPL = np.hstack((MP, cell_indices.reshape(-1,1)))
        else:
            MPL = []

        return nodes, ME, BCdict, MPL, MPE, [], []

    # 被切割单元矩阵
    CarckElement = elements[CarckElementIndex]
    # 找到所有被切割单元对应节点，即星点
    CStarNode = np.unique(CarckElement)

    # 建立数学覆盖{CStarNode}:[elementIndex]
    StarN2E = get_elements_per_node(elements, CStarNode)

    for nodeIndex, elements_list in zip(CStarNode, StarN2E):

        # 找到被所有单元中的切割单元对应的crack索引cindex
        mapped_values = [E2C[x] for x in elements_list if x in E2C]
        mapped_values_flat = list(chain.from_iterable(mapped_values))
        cindex = np.unique(mapped_values_flat)

        # 找到所有的流形单元
        mask = np.where(np.isin(indexall, elements_list))[0]
        mpoint = []
        for i in mask:
            mpoint.append(ME[i].inxy)
        mpoint = np.array(mpoint)

        # 单元边界点，用于辅助分析（避免未完全切割数学覆盖）
        DnodeIndex = np.unique(elements[elements_list])
        DnodeIndex = DnodeIndex[DnodeIndex != nodeIndex]
        Dnodes = nodes[DnodeIndex,:]

        # 合并原始节点和辅助节点
        all_points = np.vstack((mpoint, Dnodes))
        n_original = len(mpoint)  # 原始节点数量

        # 进行连通域分析（BFS算法）
        unvisited = set(range(len(all_points)))
        groups = []

        while unvisited:
            start = next(iter(unvisited))
            unvisited.remove(start)
            group = [start]
            queue = [start]

            while queue:
                current = queue.pop(0)
                for v in list(unvisited):
                    p_current = all_points[current]
                    p_v = all_points[v]
                    if not is_line_intersect_joints(p_current, p_v, crack[cindex,:]):
                        group.append(v)
                        unvisited.remove(v)
                        queue.append(v)
            groups.append(group)

        # 过滤辅助节点，仅保留原始节点的分组
        original_groups = []
        for group in groups:
            # 提取组内原始节点（索引 < n_original）
            orig_group = [idx for idx in group if idx < n_original]
            if orig_group:  # 忽略空组
                original_groups.append(orig_group)

        # 仅当存在多个分组时才需要复制节点
        if len(original_groups) > 1:
            # 第一个分组保留原节点，后续分组创建新节点
            for i, group in enumerate(original_groups[1:], start=1):
                # 复制节点
                new_node = nodes[nodeIndex, :]
                nodes = np.vstack((nodes, new_node))
                new_node_index = len(nodes) - 1  # 新节点索引
                
                # 修改原始单元矩阵中的节点索引
                for idx in group:
                    element_idx = mask[idx]  # 映射回原始单元索引
                    MPE[element_idx, MPE[element_idx] == nodeIndex] = new_node_index

    # 分配自由度
    for i, me in enumerate(ME):
        me.dof = MPE[i]

    # 处理约束
    BCdict = FindBC(BC, nodes)

    # 注意这里直接采用MC寻找可能不准确，需要确认
    # 寻找荷载点所在单元
    if len(NC) != 0:
        cell_indices = find_cells_for_points(NC[:,0:2], para)
        for i, index in enumerate(cell_indices):
            ME[index].PointLoad = NC[i,:]
        
    # 寻找测量点所在单元
    if len(MP) != 0:
        cell_indices = find_cells_for_points(MP, para)
        MPL = np.hstack((MP, cell_indices.reshape(-1,1)))
    else:
        MPL = []    

    # 找到裂尖单元
    import pyvista as pv
    # 建立pyvista格式网格grid
    num_elements = MPE.shape[0]
    cells = np.empty((num_elements, 5), dtype=int)  # 1 (count) + 4 (nodes)
    cells[:, 0] = 4  # 每个单元有 4 个节点
    cells[:, 1:] = MPE  # MPE.shape 应为 (num_elements, 4)
    celltypes = np.full(num_elements, pv.CellType.QUAD, dtype=np.uint8)
    points = np.column_stack((nodes, np.zeros(nodes.shape[0])))  # 升维到 3D
    grid = pv.UnstructuredGrid(cells, celltypes, points)

    zeros = np.zeros((crack.shape[0]*2, 1))
    point_ = np.hstack((crack.reshape(-1,2), zeros))
    # 检查所在的网格
    cell_indices = grid.find_containing_cell(point_)
    CrackE2N = []
    CrackMat = np.zeros((len(ME), 4))
    cii = 0
    for i, cell_idx in enumerate(cell_indices):
        if cell_idx in CME:
            continue
        else:
            crack_seg_idx = i // 2      # 假设每段裂缝有2个端点
            endpt_idx = i % 2           # 0或1，表示线段的首或尾端点    
            line_coords = crack[crack_seg_idx,:]
            line = LineString([(line_coords[0], line_coords[1]), 
                            (line_coords[2], line_coords[3])])
            poly_coords = nodes[MPE[cell_idx],:]
            polygon = Polygon(poly_coords)
            intersection = line.intersection(polygon.boundary)
            if endpt_idx == 0:
                CrackMat[cii] = np.array([intersection.x, intersection.y, line_coords[2], line_coords[3]])
            else:
                CrackMat[cii] = np.array([line_coords[0], line_coords[1],intersection.x, intersection.y])
            CrackE2N.append([cell_idx, cii, endpt_idx, -1, -1])
            cii += 1

    # from DrawAll import PlotCutMesh
    # PlotCutMesh(ME, np.array(CrackE2N)[:,0,])

    return nodes, ME, BCdict, MPL, MPE, np.array(CrackE2N), CrackMat