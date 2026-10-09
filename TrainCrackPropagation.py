'''
裂纹扩展物理神经网络计算
'''
import torch
import torch.nn as nn
from collections import OrderedDict
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
from GetCrack import CalCrack

from ShapeF import lagrange_basis
def NFunction2D(gxy, nodes):
    """
    在全局坐标下计算四边形单元的重心坐标形函数
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
    return N1D
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
    J_inv = np.linalg.inv(J)
    dNdx = dNdxi @ J_inv
    return dNdx

# 全局设置（保留float32，按你的要求忽略精度问题）
torch.set_default_dtype(torch.float32)

def Preprocess(nodes=None, MPE=None, CE=None, BC=None, mat=None, body_force_np=None):

    # if nodes is None:
    #     with h5py.File(f'FEM_result_0.h5', 'r') as f:
    #         MPE = f['MPE'][:]  # 单元节点索引 [N_elem, N_node_per_elem]
    #         CE = f['CE'][:]    # 单元附加信息（面积）
    #         nodes = f['nodes'][:] # 节点坐标 [N_node, 2]
    #         body_force_np = f['gravity'][:]
    #         BC = f['bbcc'][:]
    #         mat = f['mat'][:]
    E = mat[0]
    v = mat[1]

    # --- Dirichlet边界预处理 ---
    N_node = nodes.shape[0]
    # 1. 布尔掩码：shape [N_node, 2]，True=该分量强制锁定，不参与训练
    dirichlet_mask = np.zeros((N_node, 2), dtype=bool)
    # 2. 预设位移值：shape [N_node, 2]，存储锁定分量的目标位移
    dirichlet_value = np.zeros((N_node, 2), dtype=np.float32)

    for bbcc in BC:
        # bbcc逻辑：[坐标轴, 取最大/最小, 位移分量]
        col_data = nodes[:, int(bbcc[0])]
        MorM = int(bbcc[1])
        if MorM == 0:
            mm = col_data.min()
        elif MorM == 1:
            mm = col_data.max()
        
        # 获取边界节点的全局索引
        node_indices = np.flatnonzero(col_data == mm)
        dof = int(bbcc[2])
        if dof == 2:
            dirichlet_mask[node_indices, :] = True
            dirichlet_value[node_indices, :] = bbcc[3]
        else:
            dirichlet_mask[node_indices, dof] = True
            dirichlet_value[node_indices, dof] = bbcc[3]

    # --- 有限元预计算 ---
    N_elem = MPE.shape[0]
    N_node_per_elem = MPE.shape[1]
    
    # 初始化数组（按1点积分设置，暂时忽略精度）
    N = np.zeros((N_elem, N_node_per_elem), dtype=np.float32)
    dN_dx = np.zeros((N_elem, N_node_per_elem, 2), dtype=np.float32) # [N_elem, N_node, 2]
    detJ_weight = np.zeros(N_elem, dtype=np.float32)

    # 材料参数
    la = (E*v)/((1+v)*(1-2*v))
    mu = E/(2*(1+v))
    elem_lame = np.tile([la, mu], (N_elem, 1))

    for i, mp in enumerate(MPE):
        area = CE[i][0]
        gxy = CE[i,1:3]
        nodecoord = nodes[mp, :]
        Nq = NFunction2D(gxy, nodecoord)
        dNdx = strain_matrix(gxy, nodecoord)
        detJ_weight[i] = area
        N[i] = Nq
        dN_dx[i] = dNdx
    
    return N, detJ_weight, dN_dx, elem_lame, MPE, nodes, dirichlet_mask, dirichlet_value, body_force_np


class Dfem(object):
    def __init__(self, step_id=0, nodes=None, MPE=None, CE=None, BC=None, 
                 body_force=None, mat=None, device=None):
        super(Dfem, self).__init__()
        if device is None:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = device

        # --- 1. 加载预计算数据 ---
        # 动态预处理
        N, detJ_weight, dN_dx, elem_lame, MPE, nodes, dirichlet_mask, dirichlet_value, body_force_np = \
            Preprocess(nodes=nodes, MPE=MPE, CE=CE, BC=BC, body_force_np=body_force, mat=mat)
                
        self.body_force = torch.tensor(body_force_np, device=self.device, dtype=torch.float32)

        self.N_elem = N.shape[0]
        self.N_node = nodes.shape[0]
        self.N_node_per_elem = N.shape[1]
        
        # 转为Tensor
        self.N = torch.from_numpy(N).float().to(self.device)           # [N_elem, N_node_per_elem]
        self.detJ_weight = torch.tensor(detJ_weight).float().to(self.device) # [N_elem]
        self.dN_dx = torch.tensor(dN_dx).float().to(self.device)       # [N_elem, N_node_per_elem, 2]
        self.elem_lame = torch.tensor(elem_lame).float().to(self.device) # [N_elem, 2]
        self.elem_ids = torch.tensor(MPE).long().to(self.device)        # [N_elem, N_node_per_elem] 注意：必须是long
        self.nodes = torch.tensor(nodes).float().to(self.device)
        
        # Dirichlet边界
        self.dirichlet_mask = torch.tensor(dirichlet_mask).bool().to(self.device)
        self.dirichlet_value = torch.tensor(dirichlet_value).float().to(self.device)

        self.step_id = step_id
        self.result_dir = f'results/step_{step_id:03d}'
        os.makedirs(self.result_dir, exist_ok=True)

        # --- 2. PEFEN核心：定义可训练参数（节点位移） ---
        # 移除DNN，唯一可训练参数是节点位移U
        self.U = nn.Parameter(torch.zeros(self.N_node, 2, device=self.device))
        
        # --- 3. 优化器配置 ---
        # 优化对象改为 self.U.parameters()
        self.optimizer = torch.optim.LBFGS(
            [self.U], 
            lr=0.5, 
            max_iter=20,
            line_search_fn="strong_wolfe" # 虽然你没提，但加上对LBFGS收敛好
        )

        # 可视化与日志
        self.loss_history = []
        self.lr_history = []
        self.log_file = 'training_all_steps.log'

    def enforce_dirichlet(self):
        """硬施加Dirichlet边界条件：锁定固定节点位移并清零梯度"""
        with torch.no_grad():
            self.U[self.dirichlet_mask] = self.dirichlet_value[self.dirichlet_mask]
            if self.U.grad is not None:
                self.U.grad[self.dirichlet_mask] = 0.0

    def loss_func(self):
        # 1. 取出每个单元的节点位移
        elem_node_disps = self.U[self.elem_ids]  # [N_elem, 3, 2] (三角形单元3个节点)
        
        # 2. 计算积分点位移 u = sum(N_i * u_i)
        u = torch.einsum('ei,eid->ed', self.N, elem_node_disps)  # [N_elem, 2]
        
        # 3. 计算位移梯度 grad_u = (dN/dx)^T @ u_node
        # self.dN_dx: [N_elem, 3, 2] (3个节点, 对x/y的导数)
        # elem_node_disps: [N_elem, 3, 2] (3个节点, 2个位移分量)
        # [N_elem, 2, 3] @ [N_elem, 3, 2] -> [N_elem, 2, 2]
        grad_u = torch.bmm(self.dN_dx.transpose(1, 2), elem_node_disps)
        
        # 4. 物理编码：线弹性应变能
        # 小应变张量 eps = 0.5 * (grad_u + grad_u^T)
        eps = 0.5 * (grad_u + grad_u.transpose(1, 2)) # [N_elem, 2, 2]
        
        tr_eps = torch.einsum('eii->e', eps) # 应变迹 [N_elem]
        lambda_ = self.elem_lame[:, 0]        # [N_elem]
        mu = self.elem_lame[:, 1]             # [N_elem]
        
        # 应变能密度 psi = 0.5*lambda*(tr(eps))^2 + mu*eps:eps
        psi = 0.5 * lambda_ * tr_eps**2 + mu * torch.sum(eps**2, dim=(1, 2))
        
        # 5. 总应变能积分
        internal_energy = torch.sum(psi * self.detJ_weight)
        
        # 6. 外力功
        # 计算每个积分点的体力功：b·u * 积分权重
        body_work_per_elem = torch.einsum('ed,d->e', u, self.body_force)  # [N_elem] 每个单元的b·u
        body_work = torch.sum(body_work_per_elem * self.detJ_weight)   # 全域积分求和
        neumann_work = 0.0

        # 7. 总势能（损失函数）
        loss = internal_energy - body_work - neumann_work
        return loss

    def _log(self, message):
        print(message)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(message + '\n')

    def _train_one_step(self, num_epoch, step, patience=5, tol=1e-7):
        since0 = time.time()
        best_loss = 1e32
        file_1 = os.path.join(self.result_dir, f'dfem_lowest')
        file_2 = os.path.join(self.result_dir, f'dfem_last')
        
        # 初始化：先施加一次边界条件
        self.enforce_dirichlet()

        # 早停监控变量
        patience_counter = 0
        prev_loss = float('inf')

        log_mode = 'w' if step == 0 else 'a'
        with open(self.log_file, log_mode, encoding='utf-8') as f:
            if step == 0:
                f.write(f"{'='*60}\n")
                f.write(f"Multi-step Crack Training Log\n")
                f.write(f"Started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Device: {self.device}\n")
                f.write(f"{'='*60}\n\n")

        for epoch in range(num_epoch):
            since = time.time()
            
            # 前向验证损失
            with torch.no_grad():
                pd_loss = self.loss_func().item()
            
            # 保存最优
            if pd_loss < best_loss:
                best_loss = pd_loss
                state = {'state_dict': self.U.data.clone(), 'epoch': epoch, 'loss': best_loss}
                torch.save(state, file_1)
                self._log(f'  [Best] Model saved at epoch {epoch}, loss: {best_loss:.6e}')

            # 早停判断（相对变化）
            if epoch > 0:
                rel_change = abs(prev_loss - pd_loss) / (abs(prev_loss) + 1e-12)
                if rel_change < tol:
                    patience_counter += 1
                else:
                    patience_counter = 0  # 有显著下降则重置计数器
            
            if patience_counter >= patience:
                self._log(f"Early stopping triggered at epoch {epoch+1} (loss change < {tol:.1e})")
                break
                
            prev_loss = pd_loss  # 更新历史损失

            # --- LBFGS训练步 ---
            def closure():
                self.optimizer.zero_grad()
                loss = self.loss_func()
                loss.backward()
                # 反向传播后立刻清零Dirichlet节点的梯度
                self.enforce_dirichlet() 
                return loss
            
            self.optimizer.step(closure)
            
            # 记录
            current_lr = self.optimizer.param_groups[0]['lr']
            self.loss_history.append(pd_loss)
            self.lr_history.append(current_lr)
                       
            if epoch == num_epoch - 1 or patience_counter >= patience:
                self._save_plot(epoch)
            
            time_elapsed = time.time() - since
            self._log(f'Epoch [{epoch+1:4d}/{num_epoch}] | '
                     f'Time: {time_elapsed:.2f}s | '
                     f'Loss: {pd_loss:.6e} | '
                     f'LR: {current_lr:.2e}')
            
            state = {'state_dict': self.U.data.clone(), 'epoch': epoch, 'loss': pd_loss}
            torch.save(state, file_2)
        
        time_elapsed = time.time() - since0
        self._log("\n" + "="*60)
        self._log(f'Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.2f}s')
        self._log(f'Best loss: {best_loss:.6e}')

    def save_step(self, U: torch.Tensor, epoch: int, loss: float):
        """保存当前步状态"""
        path = os.path.join(self.result_dir, f'step_{self.step_id:03d}_epoch_{epoch}.pt')
        torch.save({
            'step_id': self.step_id,
            'epoch': epoch,
            'loss': loss,
            'U': U.detach().cpu(),
        }, path)

    def outputs(self):
        file = 'dfem_lowest'
        if os.path.exists(file):
            print(f"Loading model from {file}")
            weight = torch.load(file, map_location=self.device)['state_dict']
            self.U.data = weight
        else:
            print("Warning: Checkpoint not found, using current U")
        
        # 保存结果
        uv = self.U.detach().cpu().numpy()
        np.save('nu.npy', uv)
        print("Results saved to nu.npy")

def plot_loss_curve_all_steps(losses, epochs_per_step, save_path='loss_curve_all_steps.png', 
                              step_marks=True, log_scale=True):
    """
    训练结束后统一绘制多步损失曲线
    
    Parameters:
    - losses: list[float], 全局损失历史（按 step 顺序拼接）
    - epochs_per_step: int, 每步训练的 epoch 数
    - step_marks: bool, 是否用虚线标注步间分隔
    - log_scale: bool, 是否使用对数纵坐标
    """
    
    plt.figure(figsize=(10, 5))
    plt.rcParams.update({'font.size': 10})
    
    x = np.arange(len(losses))
    y = np.array(losses)
    
    # 绘制主曲线
    plt.plot(x, y, 'b-', linewidth=1.2, label='Loss')
    
    # 步间分隔线（可选）
    if step_marks:
        for i in range(1, len(losses) // epochs_per_step):
            plt.axvline(x=i * epochs_per_step, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
    
    # 坐标轴与标签
    if log_scale:
        plt.yscale('log')
        plt.ylabel('Loss (log scale)')
    else:
        plt.ylabel('Loss')
    
    plt.xlabel('Global Epoch')
    plt.title('Training Loss Curve (All Steps)')
    plt.grid(True, alpha=0.2, which='both' if log_scale else 'major')
    
    # 添加步数标注（顶部小刻度）
    ax = plt.gca()
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks([i * epochs_per_step for i in range(len(losses) // epochs_per_step + 1)])
    ax2.set_xticklabels([f'S{i}' for i in range(len(losses) // epochs_per_step + 1)], 
                        fontsize=8, rotation=45, ha='right')
    ax2.set_xlabel('Step', labelpad=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Loss curve saved to {save_path}")
    plt.close()

def main():
    N_steps = 22  # 总步数
    epochs_per_step = 200 # 50
    sigm_t = 0.001

    with h5py.File(f'NMM_result_0.h5', 'r') as f:
        MPE = f['MPE'][:]  # 单元节点索引 [N_elem, N_node_per_elem]
        CE = f['CE'][:]    # 单元附加信息（面积）
        nodes = f['nodes'][:] # 节点坐标 [N_node, 2]
        body_force_np = f['gravity'][:]
        BC = f['bbcc'][:]
        material = f['mat'][:]
        CrackE2N = f['CrackE2N'][:]
        CrackMat = f['CrackMat'][:]
        crack = f['crack'][:]
    ME = np.load('NMMmyarray_0.npy', allow_pickle=True).tolist()

    BCo = BC.copy()
    cn = len(CrackE2N)
    all_losses = [] 
    # 更新输入
    for step in range(N_steps):
        print(f"\n=== Step {step}/{N_steps} ===")
        
        BC[:,3] = (step + 1) * BCo[:,3]
        # 初始化新步模型
        model = Dfem(step_id=step, nodes=nodes, MPE = MPE, CE = CE, BC=BC, body_force=body_force_np, mat=material)
        
        # 训练当前步
        model._train_one_step(epochs_per_step, step)
        
        # 保存并传递状态
        U_prev = model.U.data.clone()
        model.save_step(U_prev, epochs_per_step-1, model.loss_history[-1])

        uv = U_prev.cpu().numpy()
        uv = uv.reshape(-1)

        # 输出当前步结果
        with h5py.File(f'NN_result_{step}.h5', 'w') as f:
            f.create_dataset('nodes', data=nodes)
            f.create_dataset('U', data=uv)
        np.save(f'NNmyarray_{step}.npy', ME)

        nodes, MPE, ME, CrackE2N, CrackMat, cn, uv = CalCrack(crack, material, nodes, MPE, ME, uv, sigm_t, CrackE2N, CrackMat, cn)
        CE = np.zeros((len(ME),3))
        for i, me in enumerate(ME):
            area_val = me.Area[0] if hasattr(me.Area, '__len__') and len(me.Area) == 1 else me.Area
            CE[i] = np.array([area_val,me.gxyz[0],me.gxyz[1]])

        all_losses.extend(model.loss_history)

    plot_loss_curve_all_steps(all_losses, epochs_per_step, save_path='loss_curve_all_steps.png')

if __name__ == '__main__':
    main()