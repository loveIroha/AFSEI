# 固定组织 EP 传播 demo

本例用 AFSEI 的 EP 求解器完成角部刺激的三维组织传播实验。计算区域取自
Santiago et al. (2018), Figure 6 / Section 3.1；组织参数取自 Table 1。
论文 DOI：[10.1002/cnm.3140](https://doi.org/10.1002/cnm.3140)。

论文使用 O'Hara–Rudy，当前例子按用户选择使用项目已有 GPB。
因此本例复现几何、组织参数与传播实验组织，不宣称复现论文的具体激动时间。
GPB/Land CUDA 接口被复用，但 `advance_land=false`，不推进 Land，也不计算主动载荷。
固体位移恒为零，伸长恒为 1；本阶段不建立或求解流体域。

实际 GPU 结果、网格对照和适用范围见 [验证记录](VALIDATION.md)。

## 配置、实验与求解器分离

| 文件 | 职责 |
|---|---|
| `configs/santiago_gpb.json` | 完整实验定义及单位，可独立复制修改 |
| `run.py` | 同一实验的代码默认值和命令行入口 |
| `run_convergence.py` | 同参数多网格实验，调用公共驱动 |
| `compare.py` | 比较已经完成的结果，不重复求解 |
| `plot.py` | 仅利用已保存的场和轨迹重新绘图 |
| `src/afsi_torch/ep/experiment.py` | 配置类型与输入验证 |
| `src/afsi_torch/geometry/tissue_slab.py` | 确定性的结构化四面体生成 |
| `src/afsi_torch/simulation/ep_tissue.py` | 时间循环、刺激和观测组织 |
| `src/afsi_torch/simulation/ep_tissue_analysis.py` | 相同实验的网格结果比较 |

主要计算直接调用已有 `Monodomain`、`MonodomainParameters`、
`MonodomainNumerics`、`GPBLand` 和 `initial_state`，没有复制 PDE 或 GPB 实现。
修改 demo 不会改变真实左心室的默认组织参数、刺激或周期驱动。

## 完整实验定义

所有坐标采用 cm，时间采用 s，内部电位采用 V，电流采用 microA/cm²，
电导采用 microS/cm。图与 VTK 中电位输出 mV、激动时间输出 ms、钙输出 microM。

| 配置项 | 本例取值 | 来源 |
|---|---|---|
| 区域 | `[0,2] × [0,0.7] × [0,0.3] cm`，体积 0.42 cm³ | Figure 6 |
| 纤维、片层、法向 | x、y、z 方向，恒定正交单位向量 | 长轴纤维的组织实验定义 |
| 电导 `g_f,g_s,g_n` | `1.17,0.39,0.39 microS/cm` | Table 1 的 `0.00117,0.00039,0.00039 mS/cm` 乘 1000 |
| 膜电容 `C_m` | `1 microF/cm²` | Table 1 |
| 表面积体积比 `S_v` | `0.1 /cm` | Table 1，按原文保留 |
| 各向有效扩散系数 `g/(S_v C_m)` | `11.7,3.9,3.9 cm²/s` | 上述参数直接推导，无额外缩放 |
| 电边界 | 六面 `n·D∇V=0`，自然 Neumann 绝缘边界 | 论文 Section 2.4.4 |
| 初值 | 项目固定的 GPB 静息状态，`V=-81.5455936324844 mV` | 现有 AFSEI 模型 |
| 刺激区域 | `[0,0.15]³ cm`，角部有限体积区域 | Niederer 2011 的组织基准 |
| 刺激时间 | `[0,0.002) s`，单次 2 ms 脉冲 | Niederer 2011 的组织基准 |
| 刺激幅值 | 正向去极化 `35.714285714285715 microA/cm²` | 该基准的 `50000/1400` 膜面等效电流 |
| 空间离散 | 结构化六四面体分割，P1 顶点积分、集中质量 | 当前 AFSEI nodal FEM |
| 网格间距 | 默认 `0.025 cm = 0.25 mm` | demo 标准配置 |
| PDE 步长 | `0.00005 s = 0.05 ms` | 当前 AFSEI |
| ODE 步长 | `0.002 ms`，每 EP 步 25 子步 | 当前 AFSEI |
| 总时间 | `0.08 s = 80 ms` | 本 demo 的传播观察窗口 |
| 求解 | FP64；AmgX PCG-AMG；rtol `1e-10`、atol `1e-13` | 当前 AFSEI |
| 激动判据 | 第一次从下方跨越 `0 mV`，步内线性插值 | 本 demo 显式定义的观测标准 |
| 观测线 | `y=z=0`，沿 x 轴 | 对应 Figure 6 的长度方向曲线 |
| 观测切片 | `z=0` 的 x-y 面 | 本 demo 等时线输出定义 |
| 探针 | x=`0,0.5,1,1.5,2 cm` 的观测线节点及对角远端 `(2,0.7,0.3)` | 本 demo |

重要的刺激单位区别：Niederer 基准的 `50000 microA/cm³` 与
`S_v=1400 /cm` 对应膜面电流 `50000/1400 microA/cm²`。本例保留这个膜面幅值，
同时使用 Santiago 的 `S_v=0.1 /cm`，因此本例实际体积源为
`3.5714285714285716 microA/cm³`。不会把 `50000 microA/cm³`
直接当作膜面电流输入。Santiago 图 6 的具体刺激幅值、时间步长和激动阈值
在所提供正文中没有完整给出；这些项目在本例中明确指定并记录来源。
这也不是使用 Ten Tusscher–Panfilov 的完整 Niederer N-version 基准复现。

刺激的空间指示函数在四面体节点取值，边界节点包含在刺激区内。
因此集中质量计算的刺激等效体积可能大于解析立方体体积，并随网格加密变化。
报告同时保存 `stimulus_geometric_volume_cm3` 和
`stimulus_nodal_quadrature_volume_cm3`，不隐含重标定刺激强度。

## 数值方法

公共单域求解器推进

\[
 S_v C_m M_L (V^{n+1}-V^n)/\Delta t+K_N V^{n+1}
 = S_v M_L(I_{stim}^n-I_{ion}^n).
\]

`M_L` 为顶点积分的集中质量矩阵，`K_N` 为同一顶点积分的扩散矩阵。
节点 GPB ODE 使用已有 CUDA 内核和源模型的两阶段 SDIRK 子步，电位冻结于
EP 步左端；离子电流返回至已有隐式 PDE 步。保持项目当前一阶分裂流程。
矩阵与 AMG 层次在时间循环前建立，固定 dt 下复用一次 setup。
每步仍使用公共求解器的真实残差验收标准，失败会抛出异常并写 `failure.json`。

网格为每个结构化长方体切成六个相容的 Freudenthal 四面体。
`spacing_cm` 表示结构化轴向间距，不是四面体最长边；最长边为 `sqrt(3)*spacing_cm`。
域长必须为间距的整数倍，不会默默改变几何尺寸。

| 间距 | 节点数 | 四面体数 |
|---|---:|---:|
| 0.5 mm | 4,305 | 20,160 |
| 0.25 mm | 30,537 | 161,280 |
| 0.125 mm | 229,425 | 1,290,240 |
| 0.0625 mm | 1,777,377 | 10,321,920 |

最后一级采用精确嵌套的 0.0625 mm。所给论文正文写 0.06125 mm，图 6 图例写
0.0615 mm；这里不沿用该不一致标注，也不自动取整。默认对照只运行前三档，
最细一档的 GPU 内存和时间需要另行评估。

## 运行

在服务器项目根目录执行：

```bash
source scripts/activate_afsei.sh
python demo/run.py demo/ep_slab/configs/santiago_gpb.json
```

统一入口会生成唯一的结果目录，也可明确指定新目录：

```bash
python demo/run.py demo/ep_slab/configs/santiago_gpb.json \
  --output results/my_ep_slab
```

JSON 是参数文件，不是 Python 程序；标准调用形式是 `python demo/run.py case.json`。
直接入口也支持参数覆盖，最终生效设置始终保存：

```bash
python demo/ep_slab/run.py --config demo/ep_slab/configs/santiago_gpb.json \
  --mesh-spacing-mm 0.125 --output results/my_ep_slab_0125mm
```

后台启动：

```bash
mkdir -p results/logs
nohup python -u demo/run.py demo/ep_slab/configs/santiago_gpb.json \
  --output results/my_ep_slab > results/logs/my_ep_slab.log 2>&1 < /dev/null &
```

现有输出目录有文件时拒绝覆盖。可以写一份完整模板后修改：

```bash
python demo/ep_slab/run.py --write-config my_ep_slab.json
```

统一配置解码拒绝拼错的键、域外刺激、不一致的网格间距和时间步。
AmgX 缺失或求解失败不会自动更换后端；若明确要对照，可传
`--ep-backend torch_pcg_jacobi`，保持同一 nodal FEM 方程。

### 网格对照

```bash
python demo/ep_slab/run_convergence.py \
  --config demo/ep_slab/configs/santiago_gpb.json \
  --spacing-mm 0.5 0.25 0.125 --output results/my_ep_slab_meshes
```

如果三档已经完成，直接利用旧结果，避免重复求解：

```bash
python demo/ep_slab/compare.py \
  results/ep_slab_santiago_gpb_05mm \
  results/ep_slab_santiago_gpb_025mm \
  results/ep_slab_santiago_gpb_0125mm \
  --output results/my_ep_slab_comparison
```

比较前核对方程参数、刺激、边界、时间设置和观测一致；允许网格间距和输出频率不同。
给出相对于最细数值结果的观测线差异，默认不判定“已经收敛”，也不把最细网格当精确解。

## 输出

- `config_resolved.json`：所有生效参数和单位来源。
- `report.json`：求解状态、激动比例、探针位置与时刻、体积、残差、GPU 和实测耗时。
- `mesh.npz`：固定参考坐标、四面体和刺激节点标记。
- `final_state.npz`：最终电位、57 维 GPB 状态、钙和激动时间；无激动节点为 NaN。
- `probes.csv`：各探针电位和钙轨迹，时间列使用 ms。
- `activation_line.csv`：长度方向的节点坐标与激动时间。
- `voltage_snapshots.npz`：指定时刻的全场电位。
- `activation.png`：切片等时线与长度方向激动时间。
- `voltage_traces.png`、`voltage_snapshots.png`：探针轨迹及切片传播过程。
- `series.pvd` 与 `fields_*.vtu`：ParaView 时间序列；时间采用 s，坐标采用 cm。
- `activation.vtu`：最终全组织激动时间、电位与钙。

探针和切片解析到最近的结构化节点/网格面；报告保留请求和实际坐标。
默认位置都精确落在三档网格上，未用最近节点坐标伪装请求坐标。
本例保存最终状态用于分析，不提供途中断点续算；80 ms demo 可完整运行。
默认探针轨迹每 10 步（0.5 ms）采样一次；需要观察快速上升支和短暂峰值时，
可将配置 `output.trace_every` 改为 1。激动检测始终在每个 0.05 ms EP 步执行。

绘图与求解可分别执行，例如计算时传 `--no-plots`，完成后执行：

```bash
python demo/ep_slab/plot.py results/my_ep_slab
```

参考刺激定义：[Niederer et al. 2011](https://pmc.ncbi.nlm.nih.gov/articles/PMC3263775/)。
