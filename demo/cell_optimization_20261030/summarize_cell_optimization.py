from pathlib import Path
import json,tarfile,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('/mnt/large2/gjh/AFSEI');here=ROOT/'demo/cell_optimization_20261030'
out=ROOT/'results/cell_optimization_20261030';out.mkdir(exist_ok=True)
v=json.loads((here/'validation_fused.json').read_text());d=json.loads((here/'sdirk_diagnosis.json').read_text())
e=json.loads((here/'explicit_feasibility.json').read_text());installed=json.loads((here/'installed_validation.json').read_text())
old=ROOT/'results/ep_afsei_nodal_opencarp_matched_20261030_smoke'
new=ROOT/'results/ep_afsei_nodal_ode_optimized_2ms_20261030'
r0=json.loads((old/'report.json').read_text());r1=json.loads((new/'report.json').read_text())
assert r0['duration_ms']==r1['duration_ms']==2 and r1['status']=='completed'
N=r0['nodes'];a=np.memmap(old/'vm.igb',mode='r',dtype='<f4',offset=1024,shape=(3,N))
b=np.memmap(new/'vm.igb',mode='r',dtype='<f4',offset=1024,shape=(3,N))
maxvm=float(abs(a-b).max());la=np.load(old/'activation_ms.npy');lb=np.load(new/'activation_ms.npy')
assert np.array_equal(np.isfinite(la),np.isfinite(lb))
finite=np.isfinite(la);maxlat=float(abs(la[finite]-lb[finite]).max())
assert maxvm<1e-5 and maxlat<1e-7,(maxvm,maxlat)
integration=dict(passed=True,compared_duration_ms=2,common_full_frames=3,
    maximum_voltage_difference_mv=maxvm,maximum_activation_difference_ms=maxlat,activation_masks_equal=True,
    old_ODE_s=r0['ode_s'],new_ODE_s=r1['ode_s'],ODE_speedup=r0['ode_s']/r1['ode_s'],
    note='Same PDE/GPB parameters/SDIRK; preexisting baseline reused. Does not repeat the full 50ms benchmark.')
report=dict(conclusion='SDIRK repeated RHS evaluations are a major cost; node parallelism already existed. Explicit integration is promising but not promoted to default.',
    production_default='Optimized fused SDIRK; no numerical method switch',fused_validation=v,algorithm_diagnosis=d,
    explicit_diagnostic=e,installed_validation=installed,integration_validation=integration,
    rejected_strategy='Global SoA staged CUDA Graph execution: correct but slower on this benchmark; not installed as default',
    numerical_models_changed=False,GPB_per_node=True)
(out/'report.json').write_text(json.dumps(report,indent=2))
(out/'integration_validation.json').write_text(json.dumps(integration,indent=2))
fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained')
native=v['performance'];vals=[native['native_baseline']['median_ms'],native['native_cached']['median_ms'],e['Euler_full_node_timing']['median_ms']]
bars=axes[0].bar(['Original\nSDIRK','Optimized\nSDIRK','Diagnostic\nEuler 2 us'],vals,color=['#586b7a','#248bb5','#ef9d43'])
axes[0].bar_label(bars,fmt='%.2f',padding=4);axes[0].set(title='GPB, 1,777,377 nodes | EP dt 20 us',ylabel='ODE time per EP step [ms]',ylim=(0,max(vals)*1.2))
vals=[100*e['metrics'][name]['current_relative_L2_error'] for name in ['SDIRK_2us','Euler_2us','Euler_1us']]
bars=axes[1].bar(['SDIRK\n2 us','Euler\n2 us','Euler\n1 us'],vals,color=['#586b7a','#ef9d43','#d1b44c'])
axes[1].bar_label(bars,fmt='%.3f',padding=4);axes[1].set(title='50 ms voltage-clamp replay, 8 nodes',ylabel='Ionic-current relative L2 difference [%]',ylim=(0,max(vals)*1.2))
item=v['small_production']['26889'];vals=[item['baseline']['median_ms'],item['optimized']['median_ms']]
bars=axes[2].bar(['Original\nSDIRK','Optimized\nSDIRK'],vals,color=['#586b7a','#248bb5'])
axes[2].bar_label(bars,fmt='%.2f',padding=4);axes[2].set(title='Production GPB/Land, 26,889 nodes',ylabel='ODE time per EP step [ms]',ylim=(0,max(vals)*1.2))
fig.suptitle('GPU cell-model diagnosis | FP64 | Euler remains experimental',fontsize=15)
fig.supxlabel('Left: identical frozen-voltage input, 5 timing samples. Middle: refined SDIRK 0.5 us reference; no PDE accuracy claim. Right: EP dt 50 us.',fontsize=9)
for ax in axes:ax.grid(axis='y',alpha=.15)
fig.savefig(out/'cell_performance_diagnosis.png',dpi=165);fig.savefig(out/'cell_performance_diagnosis.pdf',dpi=165)
doc=f'''# 细胞模型并行性能优化与 SDIRK 算法判断

结论：细胞模型已经逐节点部署并在 GPU 并行；SDIRK 的多次 RHS 工作量是主要成本，当前实现也有可压缩的开销。正式默认仍是 SDIRK，未自动切换为显式。

## 实现优化已接入

生产入口 `src/afsi_torch/ep/cuda_tissue.py` 调用新融合执行内核，方程、模型参数、FP64、步长、迭代上限和 Land 更新保持不变。原始 `gpb_device.h`、`tissue.cu` 保留；生成器及源文件 SHA256 记录了来源。

优化包括：每节点冻结电压相关系数只计算一次；避免重复构造 58 维完整 GPB 状态及导数；在输出新导数时融合误差最大值；复用自治 GPB 的子步末导数作为下个子步的初始导数。输入状态不被覆盖，结果拥有独立内存。每节点仍独立求解，未用空间平均、跨节点复用状态或粗网格替换。

逐节点 2 ms 轨迹比较中，GPB、Land 状态和离子电流最大差异均为零。已安装公共 GPBLand API 的 25 子步验证也通过。实际 nodal PDE/CN 的 2 ms 集成测试中，全场电位最大差异 {maxvm:.6g} mV，激活时间最大差异 {maxlat:.6g} ms，激活掩码一致；ODE 耗时 {r0['ode_s']:.3f}→{r1['ode_s']:.3f} s。未重跑已有 50 ms 基准或 openCARP。

| GPU 微基准 | 原 SDIRK ms/EP步 | 同算法优化 ms/EP步 |
|---|---:|---:|
| 对齐 openCARP 的 GPB，1,777,377 节点、10 子步 | {native['native_baseline']['median_ms']:.3f} | {native['native_cached']['median_ms']:.3f} |
| 生产 GPB/Land，26,889 节点、25 子步 | {item['baseline']['median_ms']:.3f} | {item['optimized']['median_ms']:.3f} |
| 生产 GPB/Land，1,777,377 节点、10 子步 | {native['production_baseline']['median_ms']:.3f} | {native['production_cached']['median_ms']:.3f} |

这些是同输入 GPU 微基准，不是全周期流固性能。原始大规模基线复用本轮首个候选的记录，采样有波动。不同 GPB 参数版本、节点规模和 EP 步长的行之间不用于推导算法加速比。

## SDIRK 是否为主要限制

同一 GPB 和 1,777,377 节点下，一次 RHS 为 {d['native_single_RHS']['median_ms']:.3f} ms，原 SDIRK 一个 20 us EP步为 {d['native_baseline_SDIRK']['median_ms']:.3f} ms。每个 EP 步有 10 个 2 us ODE 子步；每子步含初始 RHS、两个阶段各至少两次迭代、末态电流 RHS，合计至少 60 次、最多 80 次 RHS。这与实测耗时量级一致，但不能把单 RHS 计时乘法视为严格成本分解，融合内核的访存和占用不同。

全局 SoA 阶段工作区与 CUDA Graph 候选在数值上通过，benchmark 耗时却达到 {d['native_staged_SDIRK']['median_ms']:.3f} ms，比原融合内核慢。该候选没有接入生产默认。降低寄存器压力伴随的额外显存读写抵消了收益。

## 显式诊断与决策

Euler 2 us 在同输入微基准耗时 {e['Euler_full_node_timing']['median_ms']:.3f} ms，约为原 SDIRK 的 {e['raw_Euler_speedup_over_baseline_SDIRK']:.2f} 倍速度。此入口仅存在于实验模块，没有改动生产算法。

用已有 openCARP 的 8 个节点、50 ms 电压输出插值得到冻结电压轨迹，比较原 SDIRK 2 us、Euler 2 us/1 us 与时间加密的 SDIRK 0.5 us。电流相对 L2 差异分别为 {100*e['metrics']['SDIRK_2us']['current_relative_L2_error']:.4f}%、{100*e['metrics']['Euler_2us']['current_relative_L2_error']:.4f}%、{100*e['metrics']['Euler_1us']['current_relative_L2_error']:.4f}%。全部有限，但 Euler 的某些状态分量瞬态误差明显高于 SDIRK，不能只依据电流均方指标批准替换。

64 个采样状态的冻结电压 Jacobian 给出最小 Euler 局部线性稳定步长约 {e['min_local_forward_Euler_step_limit_us']:.3f} us，2 us 在采样局部范围内满足条件。这不能保证所有未采样状态、PDE–ODE 反馈和长周期都稳定，也不能直接推广到参数不同的生产 GPB/Land。

判断：值得进行显式方法的完整 EP 精度验证；尚不应直接替换生产默认。下一步优先比较显式二阶/Rush–Larsen 或 Euler 1 us 与当前参考的激活场、波形、钙和张力，再决定格式与默认步长。此前 nodal 与 openCARP 约 7.51% 的远端激活差异仍是独立的验收问题，局部电流误差较小不能证明传播误差已达 5%。
'''
(out/'ANALYSIS_zh.md').write_text(doc)
for name in ['validation.json','validation_compact.json','validation_fused.json','sdirk_diagnosis.json','explicit_feasibility.json','installed_validation.json']:
    shutil.copy2(here/name,out/name)
with tarfile.open(out/'diagnosis_bundle.tar.gz','w:gz') as tar:
    for p in sorted(out.iterdir()):
        if p.suffix in ['.json','.md','.png','.pdf']:tar.add(p,arcname=p.name)
print(json.dumps(integration,indent=2));print(out,flush=True)
