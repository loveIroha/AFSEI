# 连续三场轨迹参考验证

本例已完成 40 步/2 ms 的配对比较。数值独立范围：参考路径采用 DOLFINx/UFL 与 SciPy 计算 EP 和流固；两条路径共用 GPB/Land CUDA 核，但使用各自电位和各自细胞状态。它不是独立细胞模型验证或完整周期算例。

已有结果在 `/mnt/large2/gjh/AFSEI/results/independent_coupled_20261025/`，无需为了查看结果再次计算。详细解释见该目录 `assessment_zh.md`。

## 查看结果

- `comparison.png/pdf/svg`：内径、动能、电位及误差曲线。
- `candidate/vtk/solid.pvd`、`fluid.pvd`、`ep.pvd`：AFSEI 路径。
- `reference/vtk/solid.pvd`、`fluid.pvd`、`ep.pvd`：独立参考路径。

每个场各 9 帧；PVD 时钟为验证经过时间 0–2 ms。几何已经变形，无需额外 Warp。细胞绝对时钟以已有 60 ms 电状态为起点；机械初态单独规定为参考几何与零流速，不能把它当成原大圆管机械历史的续算。

## 独立复现

在空闲 GPU 上激活现有 AFSEI 环境后，使用新的输出目录：

```bash
python demo/verification/run_independent_coupled.py \
  --config demo/verification/configs/coupled_reference_gpu.json \
  --initial-state results/independent_coupled_20261025/initial_state_complete.npz \
  --output results/my_independent_coupled \
  --steps 40
```

需要本机已有 `mapengfei/dolfinx:local` 镜像和 Docker 使用权限；参考容器使用两核 CPU、无网络。模型参数在 JSON 中，完整初态在 NPZ 中，不需要重新运行大圆管来生成种子。

仅检查输入可添加 `--check-inputs`，该模式只在 CPU 检查网格和字段，不推进方程。输出目录仍须为空。

`verified_source/` 保存实际通过版本。主入口后来增加了配置与完整初态参数，所调用的数值求解器没有因此改变；新输入接口已用实际初态作逐位一致核对。
