import numpy as np
import math
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import split, unary_union
import pandas as pd
from CutMesh import is_line_intersect_joints, get_elements_per_node
from ShapeF import Elastic, strain_matrix

def normalize_angle(theta):
    # 将角度转换为 [0, 180) 区间
    theta_norm = theta % 180.0
    if theta_norm < 0:
        theta_norm += 180.0
    return theta_norm

def find_boundary_ray_intersection(vertices, ray_origin, angle_deg, max_ray_length=1e6, epsilon=1e-8):
    """
    从任意多边形边界寻找射线交点（支持凹/凸多边形）
    
    参数:
    vertices : list[tuple] 或 np.ndarray
        多边形顶点列表 [(x0,y0), (x1,y1), ...] (按顺序，无需闭合)
    ray_origin : tuple (x, y)
        射线起点坐标
    angle_deg : float
        射线基准角度（度），0° = +x 轴方向
    max_ray_length : float
        射线最大长度（应大于多边形尺寸）
    epsilon : float
        顶点判定阈值和几何容差
    
    返回:
    np.ndarray (4,)
        [交点x, 交点y, 顶点A索引, 顶点B索引]
        - 交于顶点: [x, y, vertex_idx, -1]
        - 交于边:   [x, y, edge_start_idx, edge_end_idx]
        - 无交点:   [np.nan, np.nan, -1, -1]
    """
    # 1. 验证并创建多边形
    if len(vertices) < 3:
        raise ValueError("多边形至少需要3个顶点")
    
    # 确保顶点是2D坐标
    vertices_2d = [(float(x), float(y)) for x, y in vertices]
    poly = Polygon(vertices_2d)
    
    if not poly.is_valid:
        raise ValueError("无效多边形（可能自相交）")
    
    # 2. 检查射线起点是否在多边形内部
    origin_point = Point(ray_origin)
    # if not poly.contains(origin_point):
    #     # 允许起点在边界上（常见于网格处理）
    #     if not poly.boundary.contains(origin_point):
    #         raise ValueError("射线起点不在多边形内部或边界上")
    
    # 3. 生成双向射线
    theta = math.radians(angle_deg)
    dx, dy = math.cos(theta), math.sin(theta)
    
    # 创建两个方向的射线（足够长的线段）
    forward_ray = LineString([
        ray_origin,
        (ray_origin[0] + max_ray_length * dx, ray_origin[1] + max_ray_length * dy)
    ])
    backward_ray = LineString([
        ray_origin,
        (ray_origin[0] - max_ray_length * dx, ray_origin[1] - max_ray_length * dy)
    ])
    
    # 4. 按优先级检查射线方向（先正向后反向）
    for ray in [forward_ray, backward_ray]:
        # 4.1 计算射线与多边形边界的交点
        intersection = poly.boundary.intersection(ray)
        
        # 4.2 处理不同交集类型
        candidate_points = []
        if intersection.is_empty:
            continue
        
        # 提取所有交点
        if intersection.geom_type == "Point":
            candidate_points = [intersection]
        elif intersection.geom_type == "MultiPoint":
            candidate_points = list(intersection.geoms)
        elif intersection.geom_type in ["LineString", "MultiLineString"]:
            # 当射线沿边时，取最近端点
            for line in ([intersection] if intersection.geom_type == "LineString" else intersection.geoms):
                candidate_points.extend([Point(line.coords[0]), Point(line.coords[-1])])
        elif intersection.geom_type == "GeometryCollection":
            for geom in intersection.geoms:
                if geom.geom_type == "Point":
                    candidate_points.append(geom)
                elif geom.geom_type in ["LineString", "MultiLineString"]:
                    for line in ([geom] if geom.geom_type == "LineString" else geom.geoms):
                        candidate_points.extend([Point(line.coords[0]), Point(line.coords[-1])])
        
        # 4.3 过滤无效点（起点本身）并找最近点
        valid_points = [
            pt for pt in candidate_points 
            if origin_point.distance(pt) > epsilon  # 排除起点
        ]
        
        if not valid_points:
            continue
        
        # 获取距离起点最近的交点
        closest_point = min(valid_points, key=lambda p: origin_point.distance(p))
        closest_dist = origin_point.distance(closest_point)
        
        # 4.4 确定交点位置（顶点或边）
        point_coords = (closest_point.x, closest_point.y)
        
        # 检查是否在顶点上
        for v_idx, vtx in enumerate(vertices_2d):
            if Point(vtx).distance(closest_point) < epsilon:
                return np.array([point_coords[0], point_coords[1], float(v_idx), -1.0])
        
        # 检查在哪条边上
        n = len(vertices_2d)
        for edge_idx in range(n):
            start_vtx = vertices_2d[edge_idx]
            end_vtx = vertices_2d[(edge_idx + 1) % n]
            
            edge = LineString([start_vtx, end_vtx])
            # 检查点是否在边上（含内部点）
            if edge.distance(closest_point) < epsilon:
                # 确保不匹配到起点（已在前面排除）
                return np.array([
                    point_coords[0], point_coords[1],
                    float(edge_idx),
                    float((edge_idx + 1) % n)
                ])
    
    # 5. 无交点情况（理论上不应发生）
    return np.array([np.nan, np.nan, -1.0, -1.0])

def find_cells_for_edge_numpy(elements, p1, p2):
    """
    使用 NumPy 向量化操作高效查找包含指定边的所有单元。

    Args:
        elements (np.ndarray): 单元索引矩阵的 NumPy 数组。
        p1 (int): 边的第一个顶点索引。
        p2 (int): 边的第二个顶点索引。

    Returns:
        np.ndarray: 包含该边的所有单元的索引数组。
    """
    elements_np = np.asarray(elements)
    
    # 创建两个布尔掩码，分别标记包含 p1 和 p2 的行
    mask_p1 = np.any(elements_np == p1, axis=1)
    mask_p2 = np.any(elements_np == p2, axis=1)
    
    # 两个掩码进行逻辑与操作，找到同时包含 p1 和 p2 的行
    combined_mask = mask_p1 & mask_p2
    
    # 使用 np.where 找到满足条件的行的索引
    found_cell_indices = np.where(combined_mask)[0]
    
    return found_cell_indices

def find_cells_for_vertex_numpy(elements, p1):
    """
    使用 NumPy 向量化操作高效查找包含指定顶点的所有单元。

    Args:
        elements (np.ndarray or list of lists): 单元索引矩阵，每行代表一个单元的三个顶点索引。
        p1 (int): 要查找的顶点索引。

    Returns:
        np.ndarray: 包含该顶点的所有单元的索引数组。
    """
    # --- 1. 准备数据 ---
    # 确保输入是 NumPy 数组，以便进行向量化操作
    elements_np = np.asarray(elements)

    # --- 2. 广播比较 ---
    # 将整个 elements_np 数组与 p1 比较，得到一个布尔矩阵
    # 例如: [[0, 1, 2], [1, 3, 2]] == 1  ->  [[F, T, F], [T, F, F]]
    mask = (elements_np == p1)

    # --- 3. 按行聚合 ---
    # 沿着 axis=1 (行) 的方向检查是否有任何 True 值
    # 如果一行中至少有一个 True，则结果为 True
    # 例如: np.any([[F, T, F], [T, F, F]], axis=1) -> [True, True]
    row_mask = np.any(mask, axis=1)

    # --- 4. 提取索引 ---
    # 使用 np.where 找到 row_mask 中为 True 的位置的索引
    # 这些索引就是包含顶点 p1 的单元的索引
    found_cell_indices = np.where(row_mask)[0]

    return found_cell_indices

from dataclasses import dataclass
from typing import Optional

@dataclass
class Element:
    points: np.ndarray # 单元顶点
    gxyz: np.ndarray # 单元形心
    inxy: np.ndarray # 单元内部一点
    Area: float # 面积
    material: np.ndarray # 材料矩阵
    stress: np.ndarray # 单元应力
    dof: Optional[np.ndarray] = None # 星点索引
    PointLoad: Optional[np.ndarray] = None # 点荷载

def CalCrack(crackold, materials, nodes, MPE, ME, U, sigma_t, CrackE2N, CrackMat, cn):
    CrackE2N_new = CrackE2N.copy()
    D = Elastic(materials[0], materials[1])

    # 存储候选开裂单元信息: (索引, 开裂指标, 开裂类型, 角度)
    candidates = []

    # 第一阶段：计算所有候选单元的开裂指标
    for ci in range(CrackE2N.shape[0]):
        cc = CrackE2N[ci]
        MEindex = cc[0]
        isNext = cc[1]

        if isNext == -2:
            continue

        me = ME[MEindex]
        gxy = me.gxyz
        nodecoord = nodes[me.dof,:]
        B = strain_matrix(gxy, nodecoord) 
        DOF = np.dstack((2 * me.dof, 2 * me.dof + 1)).ravel() 
        stress = -(D @ B @ U[DOF])

        sigma_x, sigma_y, tau_xy = stress.ravel()

        # 主应力计算
        avg = (sigma_x + sigma_y) / 2
        R = np.sqrt(((sigma_x - sigma_y) / 2)**2 + tau_xy**2)
        sigma3 = avg - R   # 最小主应力（拉为负）

        # 主方向（σ1方向）
        theta_p = 0.5 * np.arctan2(2 * tau_xy, sigma_x - sigma_y)

        # 拉伸破坏：σ3 <= -sigma_t （因拉为负）
        if sigma3 <= -sigma_t:
            angle = theta_p # + np.pi / 2   # 裂纹面垂直于σ3方向
            indicator = -sigma3 - sigma_t  # 超限值
        else:
            continue
        
        angle = np.degrees(angle)
        candidates.append((ci, indicator, angle))

    # 如果没有候选单元，直接返回
    if not candidates:
        return nodes, MPE, ME, CrackE2N_new, CrackMat, cn, U

    # 第二阶段：选取开裂指标最大的单元
    selected = max(candidates, key=lambda x: x[1])
    ci_selected = selected[0]
    angle_selected = selected[2]

    # 第三阶段：仅对选中单元执行开裂操作
    cc = CrackE2N[ci_selected]
    MEindex = cc[0]
    isNext = cc[1]
    SorE = cc[2]
    ifcc = cc[3]
    me = ME[MEindex]
    nodecoord = nodes[me.dof,:]
    cno = cn
    if ifcc != -1:
        if SorE == 0: 
            point = CrackMat[isNext,0:2]
        elif SorE == 1: 
            point = CrackMat[isNext,2:4]
        soe = cc[4]
        if soe == 0: 
            point2 = CrackMat[ifcc,0:2]
        elif soe == 1: 
            point2 = CrackMat[ifcc,2:4]        
        # 直接连接裂尖
        CrackMat[cn, :] = np.array([point[0], point[1], point2[0], point2[1]])
        cn += 1
        CrackE2N_new[ci_selected, 1] = -2
    else:
        gxy = me.gxyz
        B = strain_matrix(gxy, nodecoord) 
        DOF = np.dstack((2 * me.dof, 2 * me.dof + 1)).ravel() 
        stress = D @ B @ U[DOF]
        angle = normalize_angle(angle_selected)
        # 沿裂尖开裂
        if SorE == 0: 
            point = CrackMat[isNext,0:2]
        elif SorE == 1: 
            point = CrackMat[isNext,2:4]
        results = find_boundary_ray_intersection(me.points, point, angle)
        if len(results) != 0:
            if np.isnan(results[0]):
                return nodes, MPE, ME, CrackE2N_new, CrackMat, cn, U
            CrackMat[cn, :] = np.array([point[0], point[1], results[0], results[1]])
            if int(results[3]) != -1:
                found_cell_indices = find_cells_for_edge_numpy(MPE, int(me.dof[int(results[2])]), int(me.dof[int(results[3])]))
                NextI = found_cell_indices[found_cell_indices != MEindex]
            else:
                found_cell_indices = find_cells_for_vertex_numpy(MPE, int(me.dof[int(results[2])]))
                NextI = found_cell_indices[found_cell_indices != MEindex]
            if len(NextI) != 0:
                for nni in NextI:
                    # 检查 nni 是否已存在于第一列
                    exists_idx = np.where(CrackE2N_new[:, 0] == nni)[0]  # 找到所有匹配的行索引
                    if len(exists_idx) > 0:
                        if CrackE2N_new[exists_idx[0],1] == -1: 
                            CrackE2N_new[exists_idx[0]] = [nni, cn, 1, -1, -1]
                        elif CrackE2N_new[exists_idx[0],1] != -2: # 该单元已有两个裂尖，直接相连
                            CrackE2N_new[exists_idx[0], 3] = cn 
                            CrackE2N_new[exists_idx[0], 4] = 1
                    else:
                        # 若不存在，拼接新行
                        CrackE2N_new = np.vstack((CrackE2N_new, np.array([nni, cn, 1, -1, -1])))
            cn += 1    
            CrackE2N_new[ci_selected, 1] = -2

    # 如果未生成新裂纹，直接返回
    if cno == cn:
        return nodes, MPE, ME, CrackE2N_new, CrackMat, cn, U

    # 延长线段（即使看起来穿过，也延长10%）
    def extend_line(line, extension_ratio=0.1):
        coords = np.array(line.coords)
        vec = coords[1] - coords[0]
        length = np.linalg.norm(vec)
        unit_vec = vec / length if length > 0 else np.zeros(2)
        extension = extension_ratio * length
        return LineString([
            coords[0] - unit_vec * extension,
            coords[1] + unit_vec * extension
        ])

    # 单元开裂与星点更新（仅针对选中单元）
    triangle = Polygon(me.points)
    # 合并切割线段（处理线段之间的交点）
    cut_lines = unary_union(LineString(CrackMat[cno,:].reshape(2, 2)))
    cut_lines = extend_line(cut_lines, extension_ratio=0.1)
    # 使用合并后的线段分割三角形
    result1 = split(triangle, cut_lines)
    # 提取分割后的多边形
    polygons = [p for p in result1.geoms if isinstance(p, Polygon)]
    # 过滤掉面积小于1e-15的多边形
    filtered_polygons = [p for p in polygons if p.area >= 1e-15]

    all_points = np.zeros((len(filtered_polygons),3))
    for i, poly in enumerate(filtered_polygons):
        
        # 计算多边形内部一点
        innerpoint = poly.representative_point()
        cx, cy = poly.centroid.x, poly.centroid.y

        x, y = poly.exterior.xy
        coords = np.column_stack((x, y))
        df = pd.DataFrame(coords)
        coords = df.round(6).drop_duplicates().values
        coords = np.flipud(coords)

        if i == 0:
            me.points = coords
            me.gxyz = np.array([cx, cy])
            me.inxy = np.array([innerpoint.x, innerpoint.y])
            me.Area = poly.area,
            index = MEindex
        elif i == 1:
            ME.append(Element(
                points = coords,
                gxyz = np.array([cx, cy]),
                inxy = np.array([innerpoint.x, innerpoint.y]),
                Area = poly.area,
                material = materials,
                stress = np.zeros((3)) ))
            index = len(ME) -1          
            MPE = np.vstack((MPE, me.dof))

        all_points[i, 0] = index
        all_points[i, 1] = innerpoint.x
        all_points[i, 2] = innerpoint.y
        
    # 记录 {旧节点ID: [新节点ID列表]} 用于进行位移向量的映射
    node_displacement_mapping = {}  

    # 建立数学覆盖
    StarN2E = get_elements_per_node(MPE, me.dof)
    if len(crackold) != 0:
        crackall = np.vstack((crackold, CrackMat[:cn,:]))
    else:
        crackall = CrackMat[:cn,:]

    for i in range(len(StarN2E)):
        nodi = me.dof[i]
        calnodecoord = nodes[nodi,:]
        nodeIndex = StarN2E[i]
        new_nodeIndex = nodeIndex[nodeIndex != MEindex]
        if len(new_nodeIndex) == 0:
            continue
        mpoint = []
        for ei in new_nodeIndex:
            mpoint.append(ME[ei].inxy)
        mpoint = np.array(mpoint)
        mpoint = np.vstack((mpoint, all_points[:,1:3]))
        indexall = np.append(new_nodeIndex, all_points[:,0])

        # 连通域分析（BFS）
        unvisited = set(range(len(mpoint)))
        groups = []
        while unvisited:
            start = next(iter(unvisited))
            unvisited.remove(start)
            group = [start]
            queue = [start]
            while queue:
                current = queue.pop(0)
                for v in list(unvisited):
                    if not is_line_intersect_joints(mpoint[current], mpoint[v], crackall):
                        group.append(v)
                        unvisited.remove(v)
                        queue.append(v)
            groups.append(group)

        for i1 in range(len(groups)):
            if i1 != 0:
                original_size = len(nodes)
                nodes = np.vstack((nodes, calnodecoord))

                # 记录映射关系
                if nodi not in node_displacement_mapping:
                    node_displacement_mapping[nodi] = []
                node_displacement_mapping[nodi].append(original_size)

                index = indexall[groups[i1]].astype(int)
                for ii in index:
                    MPE[ii, MPE[ii] == nodi] = original_size

    # 分配自由度
    for i, me in enumerate(ME):
        me.dof = MPE[i]

    # 根据节点映射更新位移向量
    new_num_dofs = len(nodes) * 2
    # 创建一个新的位移向量，并用旧向量的值填充（先填充旧节点的部分）
    U_new = np.zeros((new_num_dofs))
    U_new[:len(U)] = U
    # 遍历映射，将旧节点的位移复制到新节点上
    for old_node_id, new_node_ids_list in node_displacement_mapping.items():
        # 获取旧节点的位移 (u_x, u_y)
        old_dof_start = old_node_id * 2
        old_node_displacement = U[old_dof_start : old_dof_start + 2]
        # 将这个位移值复制给所有新节点
        for new_node_id in new_node_ids_list:
            new_dof_start = new_node_id * 2
            U_new[new_dof_start : new_dof_start + 2] = old_node_displacement

    return nodes, MPE, ME, CrackE2N_new, CrackMat, cn, U_new