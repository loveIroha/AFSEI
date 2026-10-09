# 第三阶段：源力学对齐与直接组装验证

本阶段完成源项目的被动材料、基底参数和基底积分规则对齐，并在真实 LV 网格
比较原 UFL 的完整残差、Jacobian 作用和伸长投影右端。
不改变原 AFSI_GPU 仓库或环境，也不改写用户已完成的第二阶段运行。
预充盈、物理瓣膜通道、RCR 流体端口和完整周期仍是后续阶段。

## 修正纤维空间的判断

参考固定提交：`5bdde73c6be10cefadb005a0fcec5fdb6521f336`。
[原 main.cpp](https://github.com/loveIroha/test_ele_soild_heart/blob/5bdde73c6be10cefadb005a0fcec5fdb6521f336/main.cpp)
中有两套方向系数：

- 力学 H/J、Land 张力投影与伸长投影直接绑定 `FiberDirections` 单元表达式 `f0/s0`。
- EP 绑定由它们全局 CG1 插值得到的 `fiber_func/sheet_func`。

虽然 UFL 为方向声明了 CG1 系数空间，DOLFIN 将表达式限制到一个单元时，
在该单元所有系数节点得到相同值，因此力学实际是单元常方向，与现有 DG0 相符。
只有全局 CG1 插值才会发生第二阶段验证过的共享节点覆盖。
第二阶段提出的“力学也应改为全局 CG1”推断是错误的，本阶段已纠正文档。

验证中特意把机械方向改绑到全局 CG1 Function，真实网格完整力的相对差异为
**10.178%**。这证明两种绑定不能互换；生产力学保持单元表达式等价的 DG0。

## 源参数与积分规则

参数取自 [固定 Active_land.ufl](https://github.com/loveIroha/test_ele_soild_heart/blob/5bdde73c6be10cefadb005a0fcec5fdb6521f336/Active_land.ufl)。
长度 cm，力学压力/张力 dyn/cm²，基底弹簧系数 dyn/cm³。

| 参数 | 第二阶段 AFSI 配置 | 源配置 |
|---|---:|---:|
| a / b | 2362 / 10.81 | 2400 / 5.08 |
| a_f / b_f | 200370 / 14.154 | 14600 / 4.15 |
| a_s / b_s | 37245 / 5.1645 | 8700 / 1.6 |
| a_fs / b_fs | 4108 / 11.3 | 3000 / 1.3 |
| 不可压罚系数 | 5×10⁶ | 5×10⁶ |
| 基底中心 x / y | 7.5 / 7.5 | 7.30042735 / 7.6448494 |
| 基底弹簧系数 | 5×10⁶ | 10⁶ |
| 基底三角形积分 | 16 点 Duffy | FFC 的 7 点对称五次规则 |

Land 参数、主动 PK1 的额外长度系数 4.9、积分点几何历史与关闭锚点均延续前两阶段。
被动力使用与源 UFL 相同的非等容矩阵项、逆转置修正、各向异性项和 `2κ ln(J)F⁻ᵀ`。
不能改用另一个 H–O 变体的等容矩阵项。

基底径向投影包含参考坐标的有理式。不同的五次积分规则对该项不必给出同一结果。
直接核对定位了这项差异，源配置已使用原 7 点规则，旧配置保留 16 点规则。
力与稀疏切线使用同一套规则；体积分仍为双方一致的 15 点五次四面体规则。

## 参考形式的可信来源

固定提交中已提交的 `Active_land.h` 尚含 `lmbda_prev`，而同提交 UFL/main.cpp
已采用 `X_previous` 与冻结锚点系数，不能将该旧头文件用于最新力学验证。
也不能把原服务器目录里未提交的头文件当作固定来源。

新增脚本从固定 Git 提交提取 UFL，在 AFSEI 私有缓存用服务器现有 FFC/DOLFIN
重新生成头文件，再编译 `validation/source_mechanics/oracle.cpp`。
它对相同 P1 当前位置、旧几何、Land 状态和方向向量组装自由残差 H、Jacobian J、
伸长投影右端，导出 `-H` 和 `-Jv` 与 AFSI 力的负弱式约定一致。
关闭松弛例使用固定锚点乘以 0.37，锚点在求切线时不依赖当前几何。
全程端膜压力设为零，与流体通过界面传力、不另加内膜压力的 AFSEI 接口一致。

此检查不解静态平衡，也不添加瓣膜事件或循环球压力；它只验证相同状态下的
本构载荷、空间积分及变形导数。包含非均匀 P1 状态和非仿射节点扰动。
Land 例与松弛例还从完整源残差减去相同几何的被动残差，单独核对主动力和导数。

UFL SHA-256：`132771eb69155111c9d930dd35b2a418171bfd016cb699fa2a4fc3da4baa7bc7`。
重新生成头文件的哈希及完整数值见 `SOURCE_MECHANICS_VALIDATION.json`。
生成头文件和大 CSV 位于 `.cache/`，不提交到 Git。

## 配置与旧检查点

新 CLI 运行默认选择源配置：

```bash
cd /mnt/large2/gjh/AFSEI
source scripts/activate_afsei.sh
python scripts/check_afsei_mechanics.py
python scripts/run_afsei.py --mechanics source --duration-ms 2 --output results/my_source_mechanics_2ms
```

首次检出或需要重新生成参考时：

```bash
bash scripts/prepare_afsei_mechanics_reference.sh
python scripts/check_afsei_mechanics.py
```

脚本在独立进程配置现有 Spack FEniCS 环境；不会将其 Python 路径混入 AFSEI 环境。
可通过 `AFSEI_EP_SOURCE_REPO`、`AFSEI_REFERENCE_MESH`、`AFSEI_FENICS_SETUP`
和 `DOLFIN_DIR` 指定机器上的对应路径。

Python API 使用 `AFSEIConfig.source(fsi=...)`。
`AFSEIConfig(fsi=...)` 与 `--mechanics inherited` 保留传入 AFSI 参数。
若将源配置标为 `source` 却更改其材料/基底参数，配置验证会拒绝不实标签。

旧检查点未保存 `mechanics_profile`，恢复时默认 `inherited`，保留材料、基底中心、
弹簧和 16 点规则。新源检查点保存 `source`，恢复后保持源规则。
因此当前用户已完成的 `my_ep_fsi_2ms` 可继续用于原模型续算；要对照源配置必须新建运行。
没有将新参数强行用于旧状态，也没有覆盖旧历史数据。

## 验证结果与边界

2026-10-09，RTX 4090，真实网格 26,889 节点、135,430 个四面体。
重新生成原 UFL 后，三个状态的完整力相对 L2 误差不超过约 `3.1×10⁻¹³`，
JVP 和稀疏组装切线的误差约 `10⁻¹⁴`，伸长投影右端约 `9×10⁻¹⁶`。
检查同时限制相对 L2 和最大绝对误差；不是只核对向量范数。

相关测试 **74 项通过**，包括源模型完整试算后的失败回退、减步重算、检查点重启，
以及删除新配置字段后模拟旧检查点、确认其物理参数与续算结果保持不变。
原 EP、载荷和 Paper LV 回归也全部通过。

源配置真实网格、32³ MAC、首次刺激后 2 ms：20 个接受流固步、40 个 EP 步、
1,000 个 ODE 子步，无重试。张力仍约 0.212208 dyn/cm²，初期运动很小。
这项结果只验证新参数下的统一推进，不代表峰值收缩、预充盈或分辨率研究。
源配置检查点成功续算至 502.2 ms，累计 22 个流固步；第二阶段无配置标签的真实
检查点也成功续算至 502.1 ms，自动保留 inherited 模型，两者均无重试。

下一阶段处理真实腔室封闭与物理 AV/MV 端口，然后由流体端口通量驱动 RCR，
建立 8 mmHg 预充盈和瓣膜关闭事件。
当前 `windkessel_ports_connected`、`three_field_integration_implemented` 仍为 false。
