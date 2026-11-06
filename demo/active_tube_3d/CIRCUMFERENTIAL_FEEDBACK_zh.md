# 三维单层管实际环向伸长反馈

2026-11-01：`configs/tube_cg_euler_rc_wide_gpu.json` 从 `isometric-driver` 改为
`circumferential`。旧800 ms结果的 `config.resolved.json` 保留原设置，不代表新模型。
本轮只修正反馈及增加日志；未调主动幅值、材料、支撑、流体边界或终止时间，未启动完整模拟。

参考机械纤维是单元上的单位环向方向 f0；EP导电方向仍沿管轴。
真实单元伸长 lambda=|F f0|，不是当前半径除以初始半径；刚体平移/转动不改变它。
通过一致质量 L2 投影将单元标量伸长送至节点，供节点 Land ODE 使用。
轴向拉伸不应直接当成环向伸长。

原分裂算法保持不变：每步 ODE 使用上一已接受的伸长及 Zs/Zw；隐式力学中的
Land张力使用当前试探F、上一接受F及旧Zs/Zw，并保留几何导数。
只有EP与FSI及边界检查都成功后才发布新伸长和畸变历史。
这是已有分裂的力学→Land反馈，不包含形变依赖的电导率或伸展激活电流。

日志增加 `land_input_stretch_min/max`、`land_zs_min/max`、`land_zw_min/max`。
VTK已有 `circumferential_stretch` 与 `land_input_stretch`；实际反馈时二者应一致。
Land固有的长度上限、长度相关Ca50和非负张力截断继续保留，没有新增经验限幅。

检查命令（服务器项目根目录、afsei环境）：

```sh
python scripts/check_tube_circumferential_feedback.py \
  --config demo/active_tube_3d/configs/tube_cg_euler_rc_wide_gpu.json \
  --output results/tube_feedback_20261101/feedback_validation.json
```

该检查使用实际网格、GPU一致L2投影、Euler GPB/Land及CG/CN求解器，验证规定变形、
刚体运动不变性、节点CUDA更新与可读Land方程一致性、反馈敏感性与历史不被试探修改。
它不是完整收缩试验；不能由此声称202 ms侧弯已经消除。
