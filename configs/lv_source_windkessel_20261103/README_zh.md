# 原左心室几何＋纯固体源 Windkessel

2026-11-03：按用户最新决定，真实左心室采用纯固体项目的压力驱动循环方程。
本入口替代此前未完成的固定连接区/双流道路线。旧实验和模块保留作历史记录，当前入口不调用它们。

## 方程与载荷

原始参考：`/mnt/large2/gjh/test_ele_soild_heart-test/windkessel.h`、`lv_cycle.h`，commit `5bdde73c6be10cefadb005a0fcec5fdb6521f336`。

```
dp = pLV - pAO
Q = 0                                                 dp <= 0
R(dp) = Rmin + (Rmax-Rmin)*exp(-2dp/eps)/(1+exp(-2dp/eps))
Q = dp/R(dp)                                          dp > 0
C*(pc_next-pc_old)/dt = Q - (pc_next-Pvenous)/Rp
pAO = pc_next + Rc*Q
V(X_next) - Vblood_old + dt*Q(pLV) = 0
```

参数保持原源码默认值：Rc=0.05、Rp=1.05 mmHg·s/mL，C=1.3 mL/mmHg，Pvenous=8 mmHg，Rmin=0.001、Rmax=10000、eps=0.1 mmHg。原源码的 Raortic 是未使用的旧参数，不额外加入。

给定 pLV 的标量试算与 C++ 原函数一致，pc 只在整个物理步接受后提交。闭阀仍按 Q=0 推进动脉电容压力；初次收缩开始时 pc=85 mmHg，后续周期不重置。

**pLV 是与位移联立求出的心内膜载荷；pAO 是下游主动脉压力，二者不能直接设为相等。**
载荷仅施加到原心内膜，通过既有 IB 传递；不再向盒面或额外端口重复施加 pAO。
保守血容量账本 Vblood 在射血期间按 Q 更新，不在每一步重置为带数值误差的几何体积。
闭阀时 Q=0，同一残差退化为全局等容条件。闭合体积定义中的基底扇面只用于体积计算和导数，不新增封盖力或局部乘子。

这是用户指定的**纯固体循环机制与三维流固力学组合的降阶模型**。
原基底保持自然敞开，不宣称实体瓣膜关闭或真实截面流量等于 Q_0D。
MAC 流体压力是空间场，仍通过 IB 影响心肌；日志中的 endocardial_load_mmhg 不是流体核心平均压力。
因此方程对齐并不保证 EDV/ESV 与无流体的参考解完全相同，需要完整周期结果检验。

## 阶段

1. 0–200 ms：心内膜压力线性 0→8 mmHg；200–500 ms 保持 8 mmHg。
2. 500 ms 首次刺激。先等容收缩，压力驱动瓣膜开启后射血。
3. 出流关闭时冻结接受态主动张力，在 80 ms 内线性卸载到零，并按源方案复位/暂停 Land。
4. 卸载完成且耦合压力下降到充盈阈值，进入下一次充盈：200 ms 升至8、300 ms保持，再触发下一次刺激。
5. 两次收缩及最后一次充盈完成后停止；4 s 是防止阶段卡住的上限，到上限会标记 time_limit，而不是 completed。

200+300 ms 加载时序和后续0→8加载由用户明确指定；不把它误称为原源码默认500 ms线性预加载波形。
充盈压力是已知量，直接代入，不增加压力未知量；收缩/等容阶段使用一个压力未知量与原隐式位移系统联立，无反复执行完整FSI的外层迭代。

## 保留的空间与时间离散

- 固体仍为原 LV：26889 节点、135430 四面体，不新增连接壁或细化。
- MAC 为原128³、13 cm流体盒；保留已有对流、隐式黏性、压力投影与 Griffith–Luo IB 传递、BE–BE推进。
- 本配置复用既有 spectral Dirichlet 压力线性求解后端；这是已有与multigrid对照过的执行选项，MAC方程不改变。
- 盒外边界沿用既有开放盒 p=0 / 速度预测Neumann设置；本轮不改成 npuheart 的封闭外盒。
- 原心肌源本构、主动应力和基底 beta=1e6 惩罚保留，径向自由，纵向/环向罚约束，心尖自由。不启用实验性的额外基底反力未知量。
- EP原网格、nodal P1、CN、轻量Jacobi-PCG；fiber/sheet一致质量L2投影到节点，再正交归一化，normal由叉乘确定。不使用AmgX。GPB/Land及其既有推进不在本轮修改。

## 运行与输出

服务器目录 `/mnt/large2/gjh/AFSEI`：

```bash
export CUDA_VISIBLE_DEVICES=1
export CUDA_HOME=/mnt/large2/gjh/conda-envs/afsei
export TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export MPLCONFIGDIR=/mnt/large2/gjh/AFSEI/.cache/matplotlib
/mnt/large2/gjh/conda-envs/afsei/bin/python scripts/run_afsei_source_lv.py configs/lv_source_windkessel_20261103/two_cycles.json
```

可加 `--output results/新的目录`，或者 `--max-steps 2` 做短程检查。
后台运行可在上述 Python 命令前加 `nohup`，末尾加 `> results/运行日志.log 2>&1 &`。
已有结果目录拒绝从头覆写；使用 `--resume 该目录/checkpoint.npz` 才继续。重启使用检查点内保存的物理配置。
旧双流道/封盖/全局压力模态检查点不直接转接新模型。

输出包括 `history.csv`、`events.jsonl`、完整校验和 `checkpoint.npz`、`wall.pvd` 与VTU序列。
VTU保存真实变形、位移、电位和Land伸长。CSV明确区分 pLV载荷、pAO、pc、0D射血流量及体积推得的充盈流量。

## 验证

- `scripts/check_source_wk_cpp.py` 编译原始头文件，对比144组压力/步长。pc最大差异1.42e-14 mmHg，pAO5.68e-14 mmHg，Q4.55e-13 mL/s。
- `tests/test_source_windkessel_lv.py`：RCR余额、阀门导数、耦合切线与Schur预条件、卸载/刺激事件和拒绝步回滚。
- `tests/test_source_lv_gpu.py`：实际CUDA/IB充盈、EP/GPB/Land等容推进、新检查点恢复、开阀切线。使用小型验证网格，部分时钟与动脉压力为明确标注的制造测试输入，不作为生理结果。
- `tests/test_afsei_mac_isovolumic.py`：原有等容求解器回归。
- 原真实LV短程结果与验证状态见 `results/source_lv_validation_20261103/`，不等于完整两周期验证。

最终验证记录：新增32项测试全部通过；原等容5项回归通过（首次其中2项因环境指向系统CUDA失败，改用AFSEI的CUDA13后通过）。
`preload_direct/` 在原真实网格上接受两步，dt均为1e-4 s、无回退，压力分别0.004和0.008 mmHg。
两步均2次Newton、6次GMRES；计步耗时分别5.007 s和0.540 s，第一步含首次执行开销。这不能用于推算收缩期或完整周期速度。
`restart/` 从上述检查点恢复，再接受一步到0.3 ms；PVD、VTU、CSV和检查点均已生成。
先前 `preload/` 是充盈压力未精确消元时的失败调试记录，不是最终版本结果；已修正，保留供追溯。
未启动完整两周期，尚无新版本的EF、EDV或ESV结论。


## 最新 2.5 s 左心室算例

`two_beats_2p5s_passive08.json` 采用 0–0.5 s 线性升压至 8 mmHg，0.5–0.8 s 保持 8 mmHg。预加载期间主动张力和出口流量为零。0.8 s 和 1.65 s 开始两次主动搏动，运行至 2.5 s。该配置指向服务器上的原始左心室网格，运行前按实际环境设置网格路径和输出目录。

```bash
python scripts/run_afsei_source_lv.py configs/lv_source_windkessel_20261103/two_beats_2p5s_passive08.json --output results/lv_passive08_active2
python scripts/plot_lv_passive08_20261106.py results/lv_passive08_active2
```

RTX 4090 上完整运行已接受 25,000 步。两次主动搏动射血量为 67.61 mL 和 72.24 mL，按各自刺激时刻容积计算的射血分数为 48.25% 和 49.71%。0.8001 s 的等容切换仍有 −97.80 mmHg 的心内膜耦合压力瞬态，原始结果和全时程 PV 图均保留该数据。
