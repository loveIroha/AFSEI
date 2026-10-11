# 第二十五阶段：新预加载后的主动收缩与实际流量 RCR

本阶段对应压力交接后剩余工作的前两项：从第二十四阶段静态验收通过的新状态开启 EP/GPB/Land；
在同一原 MAC 隐式流固步中连接实际相对出口流量与 RCR。
四阶段自动组织和两个完整周期仍是后续工作。本入口保存独立验证检查点，普通周期入口会拒绝它。

规则 MAC 网格、原 BE–BE 流固响应、源材料与基底参数、源方向映射、主动 PK1 和 80 ms 空间快照卸载均沿用。
EP 单域 PDE 继续由 PyTorch GPU 推进，GPB/Land 继续使用现有 CUDA 内核。
未增加外层完整 FSI 迭代、解剖瓣膜或贴体流体网格。

## 动态压力和有限步体积的一致性

直接沿用静态的旧几何压力模态，在早期 2 ms 可以通过，但张力增大后出现有限步体积误差。
例如旧版本 506 ms 的瞬时净端口流量接近零，有限步体积变化率却约为 2.64e-4 ml/s，
超过固定的 1e-5 ml/s 验收目标。这个诊断版本不作为通过结果。

修复仅针对数值传递模态。`S/J`、流体压力势 q、基底流量和其配对反力仍由已接受的旧几何建立；
P1 腔体体积梯度沿当前候选位移精确积分：

```
gbar = [gV(x_old) + 4*gV((x_old+x_new)/2) + gV(x_new)] / 6
gbar : (x_new-x_old) = V_new - V_old
tbar = -C_mmhg*(gbar - cap_covector_old)
a = tbar/(tbar:tbar)
T = S + (target-S*tbar)*a^T
I = J + a*(target-S*tbar)^T*W
```

体积是 P1 三角面及移动平均边缘盖面定义的三次多项式，梯度为二次多项式，因此 Simpson 积分精确。
零位移时 `tbar` 退化为第二十四阶段的源内膜压力模态。
`tbar` 是传递算子的数值方向，不是新增物理压力载荷；材料主动/被动力不作替换。
闭合均匀基底反力仍显式组装，与源内膜载荷的定义分开。

当前同一 Newton 步的未知量只有心肌位移和基底压力：

```
F = F_source_passive+basal+active(x_new) + C_cap_old^T*lambda
(u,p) = original_BE_response(u_depart,T*F-cap_lift(lambda))
dy = dt*I(u)
Q_i = B_i*u - C_i*I(u)
```

完整切线包含 `tbar(x)` 与 `a(x)` 对候选位移的导数；不是把变化的传递算子当常量近似。
这些导数使用解析 P1 面积导数，不增加额外压力求解。
压力模态的已知响应仍按第二十四阶段的精确 BE 叠加计算。
在 MAC 无散度条件下，传递关系给出 `gbar:I(u) = -sum(Q_i)`，
因此收敛的运动及闭合端口方程同时满足有限步体积质量关系。
接受后仍独立用实际位移速度 `(x_new-x_old)/dt` 重测 Q 和质量误差。
这里是有限步总量一致性；动态角动量、空间收敛与点位无穿透不能据此宣布完成。

本分支移除了旧路径的规定内膜压力及其额外腔压控制未知量，避免重复施加同一压力。
实际 MAC 腔内压力、基底边界压力与动脉电容压力分别记录。
保留既有基底 Schur 缓存，只按残差单位和当前出口分支作代数转换。

## 实际出口流量与 RCR

只有左心室几何，因此闭合模式使用完整盖面；出口模式沿用已有理想化 AV 区域，MV 和非端口盖面保持弱闭合。
这些区域只是可配置的边界验证模型，不代表几何中已经存在解剖瓣膜。
实际出口流量为 `Q_AV = B_AV*u - C_AV*dy/dt`，不能用 `-DeltaV/dt` 代替测量。
后者只用于独立质量核对。

```
Cw*(Pc_new-Pc_old)/dt = Q_AV - (Pc_new-Pv)/Rp
Pout = Pc_new + Rc*Q_AV
Q_AV >= 0; Pout + Rvalve*Q_AV - lambda_AV/C_mmhg >= 0
Q_AV*(Pout + Rvalve*Q_AV - lambda_AV/C_mmhg) = 0
```

RCR 与机械/流体残差联立，旧电容压力在所有候选计算中保持不变，只发布接受状态。
出口关闭时，RCR 仍按零流量的真实状态继续衰减，不能把 Pc 冻结或把它作为心室规定压力。
边界功检查使用同一 `B/C` 的伴随配对；RCR 能量检查包含 BE 电容存储、数值耗散和两电阻耗散。
质量、闭合通量、压力律及功误差超标会拒绝该步，EP、Land、应变、反力和电容状态一并回退。

## 运行与续算

```bash
python scripts/run_afsei_mac_balanced.py \
  --preload results/phase24_mac128_balanced_preload_final_v1/preload_result.npz \
  --reference-checkpoint results/phase10_mac128_source_preload_v1/checkpoint.npz \
  --mode rcr --duration-ms 40 \
  --output results/my_balanced_rcr_40ms

python scripts/run_afsei_mac_balanced.py \
  --resume results/my_balanced_rcr_40ms/checkpoint.npz \
  --duration-ms 20 --output results/my_balanced_rcr_60ms

python -m pytest -q --show-capture=no \
  tests/test_afsei_*.py tests/test_ep_*.py tests/test_mac_spectral_dirichlet.py
```

新建的参考检查点只提供原生产网格、配置和容差，不导入旧预加载位移或 Zs/Zw。
预加载结果必须通过自身校验和、参考材料字段、加载端点历史和当前被动力平衡检查。
`--mode closed` 可单独验证关闭的完整盖面；关闭状态可显式切换到 RCR 状态。
`--probe-pc-mmhg` 只用于标记清楚的循环条件探针，不能将探针结果当成生理周期通过。

保存生产者为 `afsei-mac-balanced-interval-validation`，含全部 EP/GPB/Land、应变、当前主动载荷历史、
旧基底几何、基底压力和 RCR 状态，数组与元数据均参与校验和，原子发布。
执行元数据固定 `volume_transfer=integrated-p1-gradient-v1`，拒绝旧时间几何离散的检查点。
普通 `run_afsei.py --resume` 拒绝本格式，避免误用旧执行路径。

## 验收与性能

固定工程验收目标保持为：质量误差 1e-5 ml/s，闭合盖面通量范数 1e-5 ml/s，
边界功相对误差 1e-7，开放压力律误差 1e-7 mmHg，RCR 平衡 1e-8 ml/s，RCR 能量相对误差 1e-8。
材料、张力、ODE 步长、EP 步长和原生产 Newton 容差未为通过测试而调整。

GPU 检查涵盖关闭/开放分支、非零主动张力及非零候选位移的完整解析切线、
有限步体积输运恒等式、实际正向出口流量、重启重放以及失败状态回退。
真实 LV / RTX 4090 / 原 128³ MAC 网格，原生 Pc=85 mmHg，无循环条件探针：

| 窗口 | 接受步数 | 回退 | 暖步平均 | 最大质量误差 |
|---|---:|---:|---:|---:|
| 500–540 ms | 400 | 0 | 2.264 s | 5.70e-10 ml/s |
| 540–560 ms | 200 | 0 | 2.876 s | 5.70e-10 ml/s |

553.9 ms 首次开放，开放 62 步；560 ms 实际 Qout=57.0888 ml/s，
腔内核心平均压力 103.205 mmHg，Pc=81.7910 mmHg，体积 138.189692 ml。
关闭期间体积 138.322715802 ml，最终最大张力 869350.151 dyn/cm²。
RCR 平衡最大 2.27e-10 ml/s，RCR 能量相对误差最大 1.90e-12。
第二窗口最终传递功相对误差 2.15e-14。
耗时包含 EP、GPB/Land、载荷、隐式 FSI、反馈与核心压力诊断，排除文件写入；包含缓存重建。
第一窗口 EP/GPB/Land 合计约 0.092 s，FSI 约 2.151 s；第二窗口分别约 0.083 s 与 2.772 s。
这只是早期收缩/开放窗口性能，不是两个完整周期的实测运行时间。

本阶段通过仅指主动收缩和实际流量/RCR 区间的工程验证。
未自动启动四阶段周期、闭瓣后的 80 ms 卸载或下一次充盈；卸载实现仍是已有源实现，事件组织待第三项接入。
两个完整周期、PV 环、射血/充盈体积以及周期恢复仍需第四项实际运行。
