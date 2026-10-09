# 固定 LV 血液腔体、理想化端口与流量驱动 RCR

现有输入只有左心室及内膜、外膜、基底标记，没有 AV/MV 几何。因此本阶段在基底平面
定义两个可调整的圆形端口。名称 MV/AV 表示进出口的功能，不能解释为患者真实瓣口。
这是独立的固定腔体流体子问题，尚未接入运动心肌、电生理、预充盈或完整心动周期。
`run_afsei.py` 的现有 EP/GPB/Land/FSI 路径和源力学参数保持不变。

## 几何与离散边界

- 反转内膜三角形，使法向朝血液腔体外；基底边缘补三角扇，检查每条边两次且方向相反。
- 基底法向由现有边缘拟合，正向朝心室外；拒绝不平面或非星形开口，避免静默改变网格。
- 圆形端口必须严格位于基底边缘内且互不相交，至少有四个流体 MAC 面；不允许静默裁剪。
- 全单元阶梯边界只保留血液单元。边界面上的实际标记定义封闭壁面和端口，
  参考三角扇只用于几何分类，不能用于给原开放外箱制造“封闭”状态。
- 默认局部中心 MV=(0.65,0)、AV=(-0.65,0) cm，半径 0.55、0.40 cm，网格间距 0.18 cm。
  局部原点为基底边缘节点平均位置，x 轴为世界 x 在基底平面的投影，y 轴完成右手系。
  输出 VTK 已转回原世界坐标，端口图使用局部坐标。

当前壁面固定、零法向速度；切向使用阶梯边界的无滑移鬼点。
求解非定常不可压 Stokes 方程，没有对流项。黏性采用耗散的标量图离散；端口法向
黏性导数为零，端口边缘/切向鬼点仍按无滑移处理。这是验证压力、流量与电路耦合的
近似边界，不能用于声称已经实现完整应力出口或得到收敛的生理血流。

## 流量与压力关系

采用 cm–g–s 单位；压力转换系数为源项目的 1333.22368421 dyn/cm²/mmHg。
流量由实际 MAC 面速度积分：

\[
q_j=\sum_{f\in\Gamma_j}u_f n_f A_f,
\qquad Q_{\rm in}=-q_{\rm MV},\quad Q_{\rm out}=q_{\rm AV}.
\]

流量正号朝血液腔体外，不采用腔体体积差替代实际流体端口流量。固定不可压腔体满足
\(q_{\rm MV}+q_{\rm AV}=0\)。壁面零速度与端口流量均在最终速度场上检查，没有对
输出流量做截断、减均值或事后修正。

三元件 RCR 后向 Euler：

\[
C\frac{P_c^{n+1}-P_c^n}{\Delta t}
=Q_{\rm out}^{n+1}-\frac{P_c^{n+1}-P_v}{R_p},
\qquad P_{\rm ao}^{n+1}=P_c^{n+1}+R_c Q_{\rm out}^{n+1}.
\]

端口压力以 mmHg 表示为
\(p_{\rm MV}=P_{\rm at}+R_{\rm MV}q_{\rm MV}\)，
\(p_{\rm AV}=P_c+(R_c+R_{\rm AV})q_{\rm AV}\)。R_c 只出现一次。
消去 RCR 状态得到 \(b=b_0+Zq\)，流体端口响应为 \(q=q_0-Yb\)，
因此在每步直接解 2×2 系统 \((I+Y Z)q=q_0-Yb_0\)。均匀压力模式使用解析
常量解，避免两个大绝对压力响应相减放大求解残差。全局 PDE 仍使用设备上的稀疏 PCG，
没有 CPU 稠密全局求解。

默认 R_c=0.05、R_p=1.05 mmHg·s/ml，C=1.3 ml/mmHg、P_v=8 mmHg；
MV/AV 集总阻力为 0.01、0.001 mmHg·s/ml。心房 8.5 mmHg、初始 P_c=8 mmHg
仅用于验证有流量的端口子问题，不代表患者参数或完成 8 mmHg 预充盈。

单向阀是集总模型，没有瓣叶。固定不可压腔体的两端口流量必须相等，所以这里只允许
两个端口一起开/关：若候选流量逆向，就重新求解封闭速度边界，端口面速度确实归零。
这不能模拟运动心室中的独立瓣膜开闭及等容期。两个端口关闭时压力选参考零点，
封闭腔体的绝对压力未定，不能把该零点解释为左心室实际压力。

## 守恒与保存

MAC 压力梯度与散度按面双体积配对；端口法向面质量使用半单元双体积。
检验离散分部积分、压力算子的对称正定性和端口功率 \(-\sum_j p_j q_j\)。
报告分别给出流体能量误差与流体/RCR 总能量误差，包括 BE 预测与投影耗散、黏性耗散、
电容储能/BE 耗散、各电阻耗散以及心房和静脉储库的外部功率。
这些是当前离散模型的守恒检查，不是空间收敛或生理正确性的证明。

检查点保存网格、封闭域、端口标记、坐标系、流体/RCR 全状态和配置，采用原子 NPZ、
纯数值数组及 SHA-256；禁止 pickle。恢复时核对流量、壁面、不可压条件和压力关系。
格式 producer=`afsei-rigid-cavity-rcr`，与 EP/FSI 检查点严格区分。
恢复参数来自检查点，CLI 的几何和物理参数只作用于新建运行；`--steps` 是追加步数。

## 运行及后续连接

```bash
cd /mnt/large2/gjh/AFSEI
source scripts/activate_afsei.sh
python scripts/run_afsei_cavity_ports.py --steps 20 --write-vtk --output results/my_cavity_ports_20ms
python scripts/run_afsei_cavity_ports.py --resume results/my_cavity_ports_20ms/checkpoint.npz --steps 5 --output results/my_cavity_ports_25ms
python -m pytest -q tests/test_cavity_ports.py
```

输出目录须新建或为空。`cavity_final.vtu` 是实际流体体积网格，`cavity_walls_ports.vtu`
是实际阶梯边界，标签 0/1/2=壁面/MV/AV；`idealized_ports.png` 显示圆与离散端口。
验证记录见 [CAVITY_PORTS_VALIDATION.json](CAVITY_PORTS_VALIDATION.json)。

已完成 87 项相关回归检查，以及真实 LV 20 ms 流体/RCR 运行。粗网格腔体体积
68.432688 ml，相对原三角网格 68.330348401 ml 偏差约 0.150%；实际端口面积
MV=0.9072、AV=0.5184 cm²，应与名义圆面积区别使用。20 ms 内最大流量平衡误差
1.74e-10 ml/s，最大散度 5.69e-11 s⁻¹，壁面漏流为零；总能量平衡相对误差
2.76e-10，流体能量平衡相对误差 5.10e-9。续算到 25 ms 与连续计算逐项完全一致。
真实 LV 的 8/80 mmHg 逆压差测试确认端口面速度归零。0.12 cm 网格另做了几何审计，
体积偏差约 0.0581%，没有在细网格上验证流动或声称空间收敛。

状态明确区分 `fluid_ports_connected=true` 和
`windkessel_ports_connected_to_afsei_driver=false`；`three_field_integration_implemented=false`。

下一阶段需要把壁面速度与运动心肌一致地传到流体边界，并处理运动切单元/网格的
几何守恒律。移动边界时不可继续强制 Qin=Qout，而应满足
\(dV/dt+q_{\rm MV}+q_{\rm AV}=0\)（端口移动时使用相对流量）。
RCR 仍由流体端口流量推进；流体压力应通过界面牵引反馈到固体，不能同时给内膜
重复施加 Windkessel 压力。封闭期的压力、初始预充盈以及独立阀门事件必须与运动壁面
一起处理后，才能触发 EP/Land 的接受步开闭事件并开始完整周期验证。
