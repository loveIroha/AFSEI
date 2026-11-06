# 旧参数下分离空间离散与CG实现的收益

入口scripts/benchmark_ep_controlled.py，原始配置取服务器results/ep_slab_cell_models_20261013/reference_h0.25_gpb/config_resolved.json。每轮原样保存original_config.json，并另存用于二维/三维的共同参数config.json。

区域二维2×0.7 cm、三维2×0.7×0.3 cm，间距0.25 mm，二维2349节点/4480三角形、三维30537节点/161280四面体。刺激角落各轴0–0.15 cm，35.714285714285715 µA/cm²，持续0–2 ms；全部外边界绝缘。纤维沿x，sheet沿y，常量单位正交方向。

Cm=1 µF/cm²、Sv=0.1/cm、电导(0.1171,0.03903333333333334,0.03903333333333334) µS/cm，恢复旧benchmark最终参数。CN扩散步长0.05 ms，GPB ODE子步0.002 ms，组织传播模拟120 ms。先进行200 ms无刺激孤立细胞预平衡，再将同一套标量GPB状态和电位复制到所有节点、三种方法中。预平衡及装配不计入传播计时。

三种方法：

- A consistent_tc：TorchCor原生P1一致质量/扩散装配，TorchCor Jacobi-CG。
- B nodal_tc：AFSEI顶点积分集中质量，使用同一TorchCor Jacobi-CG。
- C nodal_afsei：与B完全相同的矩阵，AFSEI独立轻量Jacobi-PCG。

所有方法使用相同PDEMethod右端构造、检查和CN公式，A/B比较质量积分，B/C比较求解器实现。常量方向下K矩阵须一致。三侧都使用AFSEI GPB CUDA内核，不将该实验称为完整原生TorchCor GPB实现对照。

三侧相同当前电位初猜，无外推；共同真残差验收max(1e-13,1e-10||b||)，每100步独立复核。TorchCor根据预条件残差换算共同标准（沿用旧基准保守0.25因子），AFSEI按真残差终止。因此B/C开销包含迭代数、停止判据和内核实现差异，并非单个GPU核的纯执行时间差。

完整轨迹各自推进GPB，会产生稍有不同的右端项，故另外从A的0、2、10、20、40 ms状态保存共同V/I/S进行控制重放：actual_ionic保留实际GPB电流，zero_ionic将I置零以与旧图的测试范围对应。每组预热2轮、每次重复5轮×5快照，共3次重复=75个计时求解，三种方法轮换测量顺序。新CN快照来自本轮轨迹，并非旧benchmark使用的历史快照文件，不能要求精确复现旧图数字。

计时均为同步墙钟RHS＋输入检查＋线性求解，不包括GPB、输出、组装或容限校准。校准对TorchCor和AFSEI一律在计时外。所有方法发布独立拥有的状态副本，避免TorchCor复用CG缓冲区修改旧时刻电位；副本复制在PDE计时外，对三侧执行相同策略。

保存每10 ms VTU和ep.pvd；fields.npz记录全场激活、快照、逐步PDE/ODE时间及两端探针。绘图scripts/plot_ep_controlled.py生成三种方法统一色标的激活图、远端电位波形、沿边激活和性能分解图。

服务器有效结果目录results/ep_controlled_rectangles_20261029_v2。无v2的目录保存首轮缓冲区共享问题触发的失败诊断，不能用作有效轨迹结果。
