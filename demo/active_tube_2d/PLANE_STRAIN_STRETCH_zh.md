# 二维平面应变管的电激活牵引

本配置是厘米级单层平面通道的电激活扩展验证。它没有变成 Sharifi 的微米级双层轴对称模型；不能把收缩幅度变化当作文献复现误差的改善。

## 主动肌肉伸长的两种明确选项

`active.stretch_model = "plane-strain-isometric"`：采用当前被动力学平面应变假设中未解析方向的伸长 `F33 = 1`，由 EP–GPB–Land 产生等长肌肉张力，再施加二维配对内向牵引。这是一个规定等长肌肉伸长的二维驱动器；管壁的运动不会通过管腔间隙改变 Land 的肌肉长度或缩短速度。它不声称具有真实圆管的环向机电双向耦合。

`active.stretch_model = "lumen-gap"`：保留原来的间隙代理。旧配置省略此字段时仍采用该选项，可复现历史结果。新旧方式不会同时叠加。

Land 公式、GPB、刺激幅值、张力比例和静息张力基准均保留原值。新方式不再把管腔间隙的大幅缩小直接等同于肌肉纤维缩短，因此消除了由该代理引入的约 28.24% 间隙缩小处的张力消失点。真实肌肉的长度反馈仍保留在共享 Land 求解器中；这里的输入是等长值。三维心肌的 `|F f0|` 计算不受这个二维选项影响。

## 固定接头区域的载荷

`active.end_inactive_cm` 指定两端无主动牵引区域，随后用原来的 `end_taper_cm` 平滑增加载荷。新封接配置令其等于 `(collar_cells + 2) × axial_MAC_spacing`，本例为 0.2 cm，其中 0.1 cm 为固定接头长度，另外 0.1 cm 为四点 IB 核的两格支撑余量，避免加载的平滑支撑直接进入固定接头。

该字段只改变主动牵引的空间支撑。没有此字段的旧配置仍取 0。

新配置同时选择 `solid.end_condition = "fixed-sealing-collars"`，将既有固定接头范围内的壁面作为固定支撑。长度为每端 `collar_cells × axial_MAC_spacing = 0.1 cm`，不延伸到活动管段。前序只固定端面时，接头内壁仍发生了弯曲和脱离，不能继续维持原有的水力隔离。

实现上在有限元质量投影中限制固定支撑自由度，力传播和速度插值使用同一个受限逆质量矩阵，保留配对的离散功率恒等式；不是只在更新后把节点坐标强行置回去。固定支撑反力由支撑承担。原 MAC 方程、IB 核、体积分和显式流固推进保留。

这里增加的是既有固定接头的固体支撑边界，没有增加内部体积约束或瓣膜，封接脱离保护仍然启用。`fixed-reference-ends` 仍可单独选择，只固定端面；旧配置省略 `solid.end_condition` 时继续采用弹簧近似固定。有限长度接头与其固定壁段均是本示例的附加几何，不能宣称这部分已与论文完全一致。

## 新的输出

历史和报告记录：

- `active_traction_max_dyn_cm2`、`active_traction_mean_dyn_cm2`：实际配对边积分点上的牵引峰值，以及按参考管长加权的平均值，已经包含静息基准扣除、几何系数、比例和端部衰减。
- `active_edge_tension_increment_*`：上述边积分点上的肌肉张力增量。
- `active_muscle_stretch_min/max`：实际传入 Land 的肌肉伸长。
- `active_zero_increment_length_fraction`：积分点上主动增量为零的参考长度比例。

`snapshots.npz` 新增配对加载位置、实际牵引和肌肉伸长的逐帧数组；原 solid/fluid/ep PVD 输出保留。EP VTK 增加肌肉伸长字段。失败时额外保存末个接受状态的几何、速度和压力，以及稳定性指标。失败发生时，EP/加载可能已进入正在尝试的显式步，因此失败文件并未宣称是可直接重启的完整检查点。

## 运行

在服务器 `/mnt/large2/gjh/AFSEI` 和现有 afsei 环境中：

```bash
python demo/active_tube_2d/run.py \
  --config demo/active_tube_2d/configs/tube_plane_strain_check_gpu.json \
  --output results/my_planar_stage1

python demo/active_tube_2d/run.py \
  --config demo/active_tube_2d/configs/tube_plane_strain_contraction_gpu.json \
  --resume results/my_planar_stage1/final_state.npz \
  --output results/my_planar_contraction
```

新配置使用 EP 50 μs、ODE 2 μs。前段显式流固 10 μs 推进到 45 ms，后段 2.5 μs 从检查点继续，计划终止时间为 60 ms。后段配置 `time.stop_on_no_contact_limit = true`，在下一尝试步将越过原 20% 参考间隙保护时正常停止，不接受该越界步。此停止状态为 `stopped-no-contact-limit`，不能当成已计算到 60 ms 或完成周期。

减小流固步长用于满足新载荷下原显式稳定性限制；没有放宽稳定性和几何保护。正式耗时应按实际步长统计，不能沿用旧 50 μs 的单步耗时直接推算。继续运行保留 EP 阶段、GPB/Land、RC 压力、流量积分与体积预算；可以缩小流固步长，不能改变物理配置。`accepted_steps` 为累计真正接受的步数，`time_grid_step` 为新时间网格上的步号，两者在缩步继续后不同。

原 `tube_sharifi_rc_gpu.json` 与 `tube_sealed_rc_gpu.json` 配置保持历史模型，用于复现；新建模方式由上述新配置明确选择。完整 60 ms 是否通过及其有效验证区间，以本次结果报告为准，不能根据配置的计划终止时间推断。
