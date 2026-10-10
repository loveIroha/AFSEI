# 原 MAC 框架中的源等容体积验证

本阶段提供独立的全局等容验证模式。保留规则 MAC 网格、后向欧拉流体投影、旧几何 IB 模板、
源被动材料、GPB/Land 主动力和解析主动切线。它尚不是物理密封腔体或完整 Windkessel 心动周期。

## 联立方程与源载荷

每个机械步只准备一次旧几何转移模板和流体输运速度，联立

\[
 R_y=y-\Delta t\,J_n U(F(X_n+y,p))=0,\qquad
 R_V=V(X_n+y)-V_*=0.
\]

`V_*` 在显式进入验证模式时从当前封闭几何确定，此后保持固定并写入检查点。
体积包括随基底边缘运动的三角扇封面，单位 cm³=ml。
压力力仍只作用于源项目内膜；不向用于计算体积的封面添加压力载荷。
因此压力力列与体积梯度不假定互为负转置，体积导数包含移动封面的贡献。

现有源预加载模型含 8 mmHg 内膜压力。求解时增加 `(p-8)` 的同一源压力算子，
实际合力等价于用未知 `p` 替换原规定压力，避免重复加压。
压力切线同时包含位移方向的 follower 项和压力方向的力列。
`p` 允许为负数；不以取绝对值或截断改变源等容舒张的有符号压力关系。

联合 Newton–GMRES 调用原 MAC 线性流体响应。每次 Newton 线性化缓存压力响应列，
使用带符号的近似 Schur 预条件；没有额外的外层完整 FSI 压力试探循环。
压力与体积残差按当前压力移动度和封闭体积梯度进行单位缩放，
机械收敛阈值仍由原配置和初始机械残差确定。最终检查真实联合残差以及源体积阈值 1e-3 ml。

普通规定压力模式仍调用原 `BEIBStepper`；等容模式使用单独的联合 Newton 适配器，
不修改原六个 MAC/流固框架文件。此模式不会自动调用此前局部基底投影实验。

## 循环状态与事件范围

本阶段只推进闭合出口假设下的 RCR 电容状态：

\[
 p_c^{n+1}=\frac{C p_c^n+\Delta t\,p_v/R_p}{C+\Delta t/R_p},\quad Q=0.
\]

采用源初始 `pc=85 mmHg`、`Rc=0.05`、`Rp=1.05 mmHg·s/ml`、`C=1.3 ml/mmHg`、`pv=8 mmHg`。
仅在完整 EP、流固和应变反馈步骤接受后更新。回退试探不发布压力或 RCR 状态。
续算保留原 `pc`，不会重新设为 85。这里的 Q=0 是验证模式假设，尚非实际瓣口密封产生的流量。

主动 PK1、Land 应变历史以及接受 AV 关闭事件后的空间张力快照和 80 ms 线性卸载均沿用已有代码。
验证模式尚未自动产生该物理 AV 关闭事件，不能把此次早期收缩短算称为已验证等容舒张。
超过动脉压力时只记录诊断，不在没有物理流量接口时擅自切换射血。
后续需接入正式等效边界与实际相对端口流量，再完成固定 ED/ES 目标、射血、关闭卸载和源平滑储库充盈的状态转换。
用户 notebook 仅作为等容压力求解参考；不复制其预设主动波形、两元素 RC 和逐步更新体积目标。

腔压未知量是源内膜载荷的全局乘子。MAC 外箱仍是零压力边界，不能将它视为已初始化的局部血液绝对压力。
全局体积约束也不证明局部基底不可穿透。
报告保持 `windkessel_ports_connected=false`、`three_field_integration_implemented=false`、
`source_full_cycle_reproduced=false`，并标明 `isovolumic_validation_only=true`。

## 运行与续算

```bash
cd /mnt/large2/gjh/AFSEI
source scripts/activate_afsei.sh
python scripts/run_afsei_isovolumic.py \
  --resume results/my_mac_preload_8mmhg/checkpoint.npz \
  --duration-ms 2 --profile --output results/my_mac_isovolumic_2ms
python scripts/run_afsei.py \
  --resume results/my_mac_isovolumic_2ms/checkpoint.npz \
  --duration-ms 1 --output results/my_mac_isovolumic_3ms
```

进入新模式要求选择全新输出目录，避免覆盖输入或旧规定压力运行。
检查点使用独立 producer `afsei-mac-isovolumic-validation`，保存原完整物理张量及固定目标、
腔压、动脉电容压力、RCR 参数和已接受闭合步数。普通 CLI 根据检查点恢复模式，不会静默取消约束。
原 `afsei-ep-fsi` 检查点仍可正常读取。

## 实测与验证

RTX 4090、真实 LV 26889 节点/135430 单元、128³ MAC，EP 步长 0.05 ms、机械步长 0.1 ms。
从源 8 mmHg 预加载 `results/phase10_mac128_source_preload_v1/checkpoint.npz` 开始：

- 2 ms：20 个机械步、40 个 EP 步、1000 个 ODE 子步，无回退。
- 固定体积 136.78932412738652 ml，最大体积偏差 2.842170943040401e-14 ml。
- 末腔压 8.024495619927599 mmHg；动脉电容压力 84.88726622558443 mmHg。
- 首步 7.8226 s；20 步均值 1.6071 s；去首步后 1.2800 s/步。
- 末步 1.3465 s，其中原隐式流固 1.2333 s、GPB/Land 0.0701 s、单域 PDE 0.0225 s。
- 末步 2 次 Newton、10 次 GMRES、19 次流体求解、220 个压力多重网格循环，真实残差 7.045e-10 < 1e-9。

这只是早期、低张力的等容验证，不能据此预测完整两周期耗时，也不能与较晚且规定压力的
第十一阶段 10–20 ms 窗口直接解释为性能提升。
2 ms 输出为 `results/phase14_mac128_isovolumic_2ms_v1`。
GPU 回归覆盖联合 Jacobian 有限差分、移动封面体积导数、受激固定体积压力响应、
RCR 接受后发布、完整张量与控制状态续算、强制失败回退、负腔压检查点及原规定压力模式。

```bash
pytest -q tests/test_afsei_mac_isovolumic.py tests/test_afsei_driver.py \
  tests/test_afsei_mac_preload.py tests/test_afsei_mac_mainline.py \
  tests/test_ep_windkessel.py tests/test_source_mechanics.py
```

本次上述 23 项检查通过（35.67 s）。未覆盖实际开/闭瓣及完整等容舒张。
源预加载与旧 20 ms 检查点的 SHA256 均未改变。

继续通过普通 `run_afsei.py` 续算 1 ms，再通过等容入口续算 7 ms：
累计推进到 510 ms，即刺激后 10 ms；100 个机械步、200 个 EP 步、5000 个 ODE 子步，无回退。
固定目标保持 136.78932412738652 ml，100 步中最大体积偏差 9.6634e-13 ml。
末腔压 10.131102125533065 mmHg，电容压力 84.43797922022382 mmHg，
末最大主动张力 10326.46 dyn/cm²，最小 detF=0.89286。
后一次 70 步运行去首步后均值为 1.6955 s/步；末步 1.7565 s，
其中流固 1.6433 s、GPB/Land 0.0702 s、单域 0.0224 s。
末步 3 次 Newton、14 次 GMRES、26 次流体求解、302 个压力循环。
最后输出为 `results/phase14_mac128_isovolumic_10ms_v1`。
每个输出目录记录本次运行的历史；100 步轨迹分布在 `..._2ms_v1`、`..._resume_v1` 和 `..._10ms_v1` 三个目录。
联合核对三个历史与最终检查点，验证模式和 RCR 状态跨两次续算均保持，没有重置体积目标或电容压力。
