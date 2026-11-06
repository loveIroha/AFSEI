# 单层圆管完整收缩—舒张观察（2026-10-24）

用户已允许修改主动应力反馈、材料和支撑，并要求完整运行。本轮引入一个明确标注的简化电驱动验证分支，不将其解释为真实心肌标定。

## 模型选择

- 新选项 `active.stretch_feedback="isometric-driver"`：GPB 和单域方程保持原样；Land ODE 输入 λ=1，畸变历史为 0。计算主动张力幅值时也使用相同等长条件。因而去掉力学短缩对 Land 的幅值反馈；它是单向电激活幅值驱动，不是完整机电双向反馈。
- 主动应力仍调用原 `active_pk1(F,fiber,Ta)`，机械变形梯度、环向纤维旋转以及 PK1 的几何导数保留。环向真实伸长仍用于诊断；VTK 分别输出 `circumferential_stretch` 与 `land_input_stretch`。
- 原 `circumferential` 模式仍为默认，旧单层、双层及真实左心室配置不切换。
- 新配置端部使用有限全方向参考表面弹簧，k=10⁶ dyn/cm³，限制轴向漂移与端部径向扩张。它是有限柔性支撑，不宣称严格固支。
- 材料仍为原单层 E=30 kPa、ν=0.3 的 Neo-Hookean；没有降低刚度或增大张力比例来拟合位移。张力比例为 1。
- NS、MAC、多重网格、IB 积分核、固体 FE 和 BE–BE 不变；EP nodal FEM+CN、GPB/Land 时间步不变。流体与力学网格不加密。

这不是单变量对照，因为主动反馈和端部支撑同时改变。也不是 Sharifi 定量复现。仍有管外虚拟流体，外箱体 p=0，原始管口位于域内，没有 RC、多孔区或瓣膜。

## 完整窗口与卸载

刺激仍在 5–7 ms，只有一次刺激。已完成的固定几何 800 ms EP/Land 检查显示，采样最大张力约 205 ms 达峰 82.34 kPa；500 ms 尚有 7.01 kPa，800 ms 为 0.273 kPa，约峰值的 0.33%。因此选择 800 ms 观察一次激活、收缩和自然松弛，没有人为截断 Land 卸载。

达到 800 ms 仅表示观察窗口完成。是否几何恢复必须由末态半径、体积、位移和张力共同判断，不能由结束状态自动宣称。

## 运行和续算

服务器结果：`/mnt/large2/gjh/AFSEI/results/single_tube_full_20261024/`。

1. `active_120ms`：从静息到 120 ms 的新模型三场前段。
2. `checkpoint_roundtrip`：零新增时间步恢复，逐项验证原保存状态没有改变。
3. `active_continued_800ms`：使用原 dt=0.05 ms 从 120 ms 继续到 800 ms；不重新刺激，不重复前段。

完整任务入口 `complete_single_cycle.py` 等待指定前段成功；若前段失败，记录失败而不擅自改参数继续。当前后台任务已启动，状态文件 `full_run_status.json`，完整日志 `full_run.log`，前段日志 `active_120ms.log`。

手动新运行可直接执行：

```bash
python demo/active_tube_3d/validate_single.py \
  --config demo/active_tube_3d/configs/tube_single_isometric_full_gpu.json \
  --mode coupled --output results/my_single_full
```

继续同模型、同时间步的已接受状态：

```bash
python demo/active_tube_3d/validate_single.py \
  --config demo/active_tube_3d/configs/tube_single_isometric_full_gpu.json \
  --mode coupled --resume results/my_prefix/final_state.npz \
  --output results/my_continuation
```

续算只用于开放单层管，物理参数和时间步必须与源配置相同。允许改变终点与输出频率。首次运行和续算均输出固体、流体、电位 PVD；后台完整入口完成后自动合并到 `analysis/solid.pvd`、`fluid.pvd`、`ep.pvd`，生成收缩恢复图和 `summary.json`。

## 已完成检查与限制

16 项 CPU/CUDA 与旧圆管检查通过，包含等长幅值对几何不变、PK1 几何切线有限差分一致、全方向端部反力及旧双层回归。完整运行结果以服务器状态文件为准，本说明不预先宣称完成。

既有 EP 轴向分辨率检查仍适用：0.5/0.25/0.125 mm 给出约 55.7/64.8/68.8 cm/s。此次沿用粗网格作为功能验证，不能宣称传播或流固已网格收敛。近闭合阶段也不能用离散散度代替移动管腔守恒误差。接触模型未加入，原最小半径保护保留。
