# EP 顶点积分与 AmgX 主线

用户授权 EP 线性求解可使用 AmgX。新计算默认采用 `nodal_p1` 与
`amgx_pcg_amg`；PyTorch 装配和时间推进，AmgX 在 GPU 内部执行 PCG，
以聚合 AMG 的固定 V-cycle 为预条件。原 `consistent_p1` 与
`torch_pcg_jacobi` 仍可显式选择。

## 数值离散

四面体 P1 形函数和节点电位保留。EP 的质量、扩散与节点源项使用四个
顶点、每点权重 `volume/4`，集中质量为 `m_i=sum_T(volume/4)`。
扩散矩阵使用四个顶点处的 CG1 方向场计算电导张量，再取加权和；
并非只把一致质量矩阵的对角提取出来。

每步求解 `A V_new=b`，其中 `A=Sv*Cm/dt*M_lumped+K_vertex`。
保留绝缘边界、固定参考网格、原电导数值与单位、ODE 先行的一阶分裂、
EP 步长 0.05 ms 和 GPB/Land 子步长 0.002 ms。
方向没有额外归一化，电导没有新增缩放。

Land 单元伸长至节点的反馈继续使用独立的一致 L2 投影；该投影不采用
EP 的集中质量矩阵。MAC 流固、RCR、阶段切换和主动载荷卸载不作修改。

## AmgX 后端

使用固定 AmgX 提交 `91a8413ef267b1c32aff4014c02820e1c5897ac2`，
服务器 CUDA 13.0、RTX 4090 架构 8.9 已实际构建验证。
通过 ctypes 接入稳定 C API，无需 PyAMGX；小型 C 辅助库从实际构建头文件
读取 `AMGX_mode_dDDI`，矩阵与向量均为 FP64，CSR 索引为 int32。

AmgX 配置使用 PCG、聚合 AMG、SIZE_2、D2、V-cycle、前后各一次
BLOCK_JACOBI 平滑及小规模粗层 DENSE_LU_SOLVER。
单进程共享一个 AmgX Resources：其析构释放全局内存池，不能让多个
并存的检查点驱动各自销毁 Resources。本后端支持每进程一个 CUDA 设备。

系统矩阵和 AMG 层次按 EP dt 缓存。当前固定 EP dt 下跨周期复用，
FSI 回退不重建。右端与初值通过设备指针复制到 AmgX 内部向量，结果复制
回 GPU 工作区；时间推进不经过 CPU 大向量。启动时的 CSR 索引转换允许
一次 CPU 传输。原始指针交换不是零拷贝，跨库所有权边界有必要的同步。

AmgX 使用规范化后的 ABSOLUTE 判据；规范化尺度使项目的接受标准仍为
`||b-Ax|| <= max(atol,rtol*||b||)`。内部目标取该标准的四分之一，
最终用 PyTorch 计算原方程的真残差。默认 rtol=1e-10、atol=1e-13。
没有放宽精度；失败不静默切换后端。

## 配置与安装

服务器已构建，无需重新安装。其他机器可以执行：

```bash
source scripts/activate_afsei.sh
python scripts/install_afsei_amgx.py
```

本机动态库默认位于 `.deps/amgx/build/libamgxsh.so`，可用
`AFSEI_AMGX_LIBRARY` 指向其他同样构建了辅助库的安装位置。
构建目录、缓存和二进制不提交到 Git。

EP 配置文件为 `configs/ep_nodal_amgx.json`，独立于 MAC 配置。

```bash
python scripts/check_afsei_ep.py --steps 40
python scripts/run_afsei.py --duration-ms 2 --output results/my_nodal_ep_fsi
```

正式周期入口新计算默认使用 nodal/AmgX；也可显式传入
`--ep-config configs/ep_nodal_amgx.json`。原有预加载和参考检查点可用于新计算。
新计算从参考检查点取配置时明确选择新的 EP 数值设置。

`--resume` 恢复保存的 EP 离散与后端，不能同时覆盖 EP 配置。
旧检查点没有 `ep_numerics` 时恢复为 consistent/Jacobi，防止续算自动
改变已接受轨迹。新检查点保存离散方式、后端与求解容差。

对照运行可使用：

```bash
python scripts/check_afsei_ep.py --steps 40 \
  --ep-discretization consistent_p1 --ep-backend torch_pcg_jacobi
```

## 实际验证与性能范围

独立四面体参考检查顶点质量、变系数扩散、常数零扩散、守恒、单位和
非均匀伸长的一致 L2 投影。GPU 检查 AmgX 与同一方程的独立稠密解一致，
复用层次、输出不覆盖旧状态以及多个驱动关闭时互不影响。
耦合检查包含接受、失败回退、检查点恢复、负载更新与原 MAC 阶段计时。
相关检查共 32 项通过（原单域 8 项、ODE 8 项、顶点/AmgX 7 项、
驱动与入口 9 项）。真实 MAC 128³ 正式入口推进至 500.2 ms，
随后从新检查点续算至 500.3 ms，均 completed；最大质量误差
2.846e-10 ml/s，累计误差 2.83e-14 ml。仅检查启动与恢复，未运行完整周期。

真实 LV：26889 节点、135430 四面体、source-serial-cg1。
40 个 EP 子步产生真实 GPB 电流；PDE 对照重放相同冻结输入，每种后端
预热 5 次并测量 120 次。耗时包括后端接口、边界同步和真残差检查，
不包括 ODE 和一次装配/setup。

| 离散与后端 | 平均每次 PDE 求解 | 平均迭代数 |
|---|---:|---:|
| consistent P1 / PyTorch Jacobi-PCG | 10.610 ms | 20.000 |
| consistent P1 / AmgX PCG-AMG | 5.411 ms | 11.275 |
| nodal P1 / PyTorch Jacobi-PCG | 4.151 ms | 5.000 |
| nodal P1 / AmgX PCG-AMG | 2.280 ms | 6.375 |

该短窗口的新组合约为原一致质量/Jacobi 组合的 4.65 倍速度。
nodal/AmgX 对比 nodal/Jacobi 约为 1.82 倍。
同一 nodal 方程下，两后端最大电位差 2.0711e-7 mV；
AmgX 真残差/容差最大比值 0.236。测量不代表整个周期或全部 EP/ODE 的加速比。

40 步 nodal/AmgX＋GPB/Land 轨迹有限，最大残差/容差 0.226。
但是顶点积分与一致质量的空间离散不同：2 ms 末最大节点电位差达到
97.36 mV。该差异不是 AmgX 求解误差，不能据此声称两种离散的传播
精度等价；后续需要激动时间、传导速度和网格收敛验证。
此轮不运行完整周期，不调整电导来拟合旧轨迹。

性能结果保存在服务器 `results/ep_nodal_amgx/benchmark.json`，
日志在 `results/logs/ep_nodal_amgx_benchmark.log`。
正式入口短程输出在 `results/ep_nodal_amgx/mac128_smoke` 和
`results/ep_nodal_amgx/mac128_resume`。首次推进包含冷启动，不用于外推周期性能。

参考：[AmgX C API](https://github.com/NVIDIA/AMGX/blob/main/include/amgx_c.h)、
[官方 PCG-AMG 配置](https://github.com/NVIDIA/AMGX/blob/main/src/configs/PCG_AGGREGATION_JACOBI.json)。
