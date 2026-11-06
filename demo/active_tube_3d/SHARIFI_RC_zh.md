> 更新：主入口 `configs/tube_sharifi_rc_gpu.json` 已采用隔离储液区与密封接头。请看 [当前边界说明](SEALED_PORTS_zh.md)。下文保留盒面RC阶段历史，仅适用于 `tube_box_rc_legacy_gpu.json`，不能当作当前主入口的物理说明。

# Sharifi RC 边界接入：第一阶段

本配置实现 Sharifi et al. (2021), DOI: 10.1007/s10237-021-01489-7，第3节、式(2)的两元 RC 方程。没有多孔介质、阀门或单向流量裁剪。

## 可运行配置

- 正式边界演示：`demo/active_tube_3d/configs/tube_box_rc_legacy_gpu.json`。
- 短程三场验证：`demo/active_tube_3d/configs/tube_box_rc_legacy_smoke_gpu.json`。
- 原 `tube_gpu.json` 保留原有边界，供历史结果追溯。

入口零参考压力，不指定入口速度和流量。初始流速为零。电激活驱动管壁收缩，出入口允许物理回流。

## 压力和流量的离散

使用 C*(p[n+1]-p[n])/dt = Q[n+1] - p[n+1]/R。
原 MAC 压力投影和黏性、对流求解不变。预先解一次出口单位压力的线性响应，然后以一个标量 Schur 补同时求出当前步出口压力和流量，无需额外的流固外迭代。三维 Newton 的流体响应包含 RC 的导数；不能仅更新残差而漏掉这个导数。

RC 历史只在流固状态通过接受检查后提交一次。失败的流体、Newton 或几何尝试不更新 RC。最终 NPZ 保存 RC 压力、步数和累计边界流量；目前圆管驱动本身不支持断点续跑，保存字段不代表已经实现重启。

## 单位

论文 Table 1: R=60 mmHg*s/mm^3，C=0.26 mm^3/mmHg，时间常数 R*C=15.6 s。
换成 CGS：R=60*1333.22387415*1000 dyn*s/cm^5；C=0.26/1000/1333.22387415 cm^5/dyn。
R、C 保持独立的带单位 JSON 字段。不能把 mm^3 当成 mL，也不能直接套用左心室 RCR 参数。

二维的 MAC 流量单位是 cm^2/s，必须乘 `extrusion_depth_cm` 才得到 mL/s。本演示显式取1 cm；这是平面通道的厚度假设，不是论文的轴对称换算。改变厚度会改变同一 RC 的负载效应。

## 边界位置和当前验证范围

二维：RC 加载整个 x-high 盒边界，x-low 压力为零，外部上下边界无滑移。
三维：RC 加载整个 z-high 盒边界，其余五个盒边界压力为零，速度使用原法向零梯度处理。
这些是原 MAC 向量拉普拉斯/压力投影的开放边界实现，不能宣称完整对称 Cauchy 牵引离散。

现有管壁和管口仍在盒子内部，保留管外流体。RC 使用实际受载盒边界的总流量，不使用管腔中截面或出口截面流量。三维允许流体通过侧面开放边界，因此此阶段的 RC 流量不能解释为论文的心管输出量。不能把一个内部截面流量拿来更新外部压力，却忽略两者之间的外部流体。

本阶段只是 RC 边界模块、状态提交、Newton 导数、单位与守恒验证。原有厘米级单层几何、材料、EP-GPB-Land、固体有限元、MAC+IB、时间格式均保留；本配置使用论文 R/C 也不构成其几何尺度下的物理复现。

下一阶段才对齐微米尺度双层结构、实际管口位置、端部约束与流体参数；二维平面与论文轴对称的差异也需明确。对管外流体和开边界的处理必须同步解决，才能比较管内净泵血量。保留 GPB-Land 时，主动加载仍不同于论文指定函数，需要另行给出同载荷对照。

## 输出

沿用 solid/fluid/ep 的 VTK/PVD 文件。history.json 与 report.json 增加 RC 出口压力、受载边界流量、RC 体积平衡误差、累计输出体积及离散能量误差。三维另有管腔 inlet/mid/outlet 截面流量，均与 RC 总边界流量分开命名。

RC 的功率为 p*Q，等于电容储能变化率、阻性耗散 p^2/R 和向后欧拉数值耗散 C*(p[n+1]-p[n])^2/(2*dt) 之和。这是循环模块的能量核对，不是整体流固时间格式的零耗散断言。

## 运行

```bash
cd /mnt/large2/gjh/AFSEI
export AFSEI_GPU=0
source scripts/activate_afsei.sh
python demo/run.py demo/active_tube_3d/configs/tube_box_rc_legacy_gpu.json --check-config
python demo/run.py demo/active_tube_3d/configs/tube_box_rc_legacy_gpu.json --output results/tube_3d_sharifi_rc
```
