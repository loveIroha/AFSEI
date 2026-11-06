# 二维 CG FEM＋Euler 电激活收缩波实验

几何与力学沿用原二维管壁：长24mm，初始管腔高度6mm，上下壁厚1mm；P2平面应变Neo-Hookean（μ=1e5、λ=2e5 dyn/cm²），固定端部封接。固体970个P2节点、384个三角形，EP294个P1节点。流体64×32 MAC网格，计算盒32×16mm；使用原有MAC多重网格、IB传递和显式滞后2D流固推进。入口零参考压，出口原RC；没有新增多孔介质或瓣膜。

EP采用优化后的AFSEI一致质量CG P1-Galerkin、CN20µs、Jacobi-PCG，无AMG。纤维沿轴向X，绝缘壁面EP边界，左端1mm宽区域的上下壁同时施加0–2ms、35.714µA/cm²刺激。电导率、χ和Cm取最近的openCARP对齐benchmark参数，有效D=(0.952984,0.125758,0.125758)cm²/s。

细胞模型保留原圆管的GPB57与Land参数、初态（Vm=-81.5455936mV），并非benchmark中特定的原生Grandi54适配器。新增GPB显式Euler CUDA内核，1µs子步，冻结EP左端电位、平均子步末离子电流；Land沿用源代码前向Euler。用exprel等价极限处理原GPB的Vm=0、-5、-47.13mV可去奇点，没有改变生理方程或参数。旧配置默认仍是SDIRK。

主动力沿用上下壁配对内向牵引，逐位置由当地GPB钙与Land状态产生。二维省略环向取等长λ=1，不将管腔间隙直接当肌肉伸长；因此没有真实环向机电双向反馈。固体与流体为原双向IB耦合。未规定收缩速度、传播速度或位移波形。弹性和流体的非局部传递可能使某处位移早于当地电激活，需结合三个场实际结果判断机械传播。

100ms验证载荷比例0.02达到最小间隙1.6003mm（73.3%局部缩小），故正式600ms选择0.002。该数值验证的主动幅值已明确缩放，不声称复现真实血管或文献绝对收缩幅度。原20%间隙保护保留，不进行接触闭合。FSI步长10µs；正式计划600ms，共60000个FSI步。GPU1共享运行，不能给出独占性能结论。

运行配置：`demo/active_tube_2d/configs/tube_cg_euler_wave_gpu.json`。

```bash
CUDA_VISIBLE_DEVICES=1 python demo/active_tube_2d/run.py \
  --config demo/active_tube_2d/configs/tube_cg_euler_wave_gpu.json \
  --output results/my_tube_cg_euler_wave
```

输出每2ms保存EP、固体和流体PVD，时间s，几何cm。`wave_sections.npz`为沿轴向的实时电位、钙、实际牵引及间隙，`history.json`为进度与守恒/稳定性诊断。`comparison/electromechanical_wave.png`绘制三个场的x-t图；`wave_probes.png`绘制同位置探针；`propagation_times.csv`分别统计Vm=0和间隙缩小5%的到达时间，未达到阈值保留NaN。

GPB Euler检查包括：独立单步RHS/Land核对、1µs/0.5µs电压钳对照、可去奇点的对称极限。100ms耦合验证为实测完成；600ms长时段是否完成必须以最终report.json为准。
