import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.colors as mcolors
from matplotlib.collections import PatchCollection
import numpy as np

def PlotMesh(ME):

    fig, ax = plt.subplots(figsize=(8, 6))

    for me in ME:
        random_color = np.random.rand(3,)
        polygon = Polygon(me.points, closed=True, edgecolor='k', facecolor=random_color, alpha=0.5)
        ax.add_patch(polygon)

    # 设置坐标轴比例一致
    plt.axis('equal')
    plt.axis('off')
    plt.savefig('mesh.png', dpi=600)
    # plt.close()
    plt.show()


def PlotCutMesh(ME, index):

    fig, ax = plt.subplots()

    for i, me in enumerate(ME):
        # random_color = np.random.rand(3,)
        if i in index:
            polygon = Polygon(me.points, closed=True, edgecolor='k', facecolor='blue', alpha=0.5)
        else:
            polygon = Polygon(me.points, closed=True, edgecolor='k', facecolor='red', alpha=0.5)

        ax.add_patch(polygon)

    # 设置坐标轴比例一致
    plt.axis('equal')
    plt.axis('off')
    plt.savefig('mesh.png', dpi=600)
    # plt.close()
    plt.show()
    
def PoltCountor(ME, sol, name):

    vmin = min(sol)
    vmax = max(sol)

    # 创建颜色映射和归一化
    cmap = plt.get_cmap('coolwarm')  # 选择颜色映射
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # 绘制多边形
    fig, ax = plt.subplots()

    # 创建多边形补丁列表
    patches = []
    colors = []

    idx = 0
    for me in ME:
        # 获取多边形顶点（包含结束行）
        poly_points = me.points
        
        # 创建多边形补丁
        polygon = Polygon(poly_points, closed=True, edgecolor='black', linewidth=1.5)
        patches.append(polygon)
        
        # 计算对应的颜色
        color = cmap(norm(sol[idx]))
        colors.append(color)
        idx += 1

    # 创建PatchCollection并设置颜色
    p = PatchCollection(patches, edgecolor='black', linewidth=0.5)
    p.set_facecolor(colors)
    ax.add_collection(p)

    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(sol)
    plt.colorbar(sm, ax=ax)
    ax.autoscale(tight=True)
    ax.set_aspect('equal')
    plt.savefig(f'result_{name}.png', dpi=600)
    plt.axis('off') 
    plt.show()
