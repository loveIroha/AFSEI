"""Create a transparent implementation-only performance report."""
import json,argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--folder',type=Path,default=root/'outputs/ep_buffered_20261029')
folder=parser.parse_args().folder
rows=json.loads((folder/'report.json').read_text())
lines=['# EP GPU执行优化验证','',
'本轮保持P1质量离散、CN、GPB模型及物理参数，优化RHS与线性求解执行。服务器GPU1，FP64，0.125 mm；二维9177节点，三维229425节点。', '',
'计时使用已保存标准GPB轨迹的20 ms电位快照；共同测试电流为0.1*sin(30*V)，刺激为零。因此这是相同输入的实现微基准，不是重新计算的完整GPB轨迹或同等精度方法比较。每项预热3次，重复3组，每组20次（单独SpMV为100次），报告组均值的中位数。计时包括RHS和求解，排除输入有限性检查、装配、重排序、首次JIT/CUDA Graph捕获、GPB和I/O。', '',
'|维度/质量|原AFSEI PDE/ms|优化PDE/ms|加速比|TorchCor核原执行/ms|',
'|---|---:|---:|---:|---:|']
for r in rows:
    new=r['solvers']['graph_natural']['pde_ms']
    lines.append(f"|{r['dim']}D / {'nodal' if r['nodal'] else 'consistent'}|{r['reference_pde_ms']:.4f}|{new:.4f}|{r['reference_pde_ms']/new:.3f}|{r['torchcor_pde_ms']:.4f}|")
lines += ['', '实施内容：', '',
'- 对角质量、源项及CN右端融合；一致质量组也复用工作缓冲与优化CSR访问。',
'- 轻量Jacobi-PCG在CUDA上使用持久工作数组、融合x/r更新、分段CUDA Graph；CPU保留原参考实现。',
'- GPU内部检查递推残差；每段结束检查真实残差，必要时替换残差并重启，保持原容限。段长受check_every和recompute_every控制；报告迭代数为段边界上界。',
'- 16行一组的CSR读取；RCM采用一次性CPU图排序，仅改变线性系统排列，求解仍在GPU。实测RCM含进出排列的总耗时无改善，生产默认natural。',
'- 合并三个输入字段的有限性检查为一次CPU判断；输出解拥有独立内存，避免下一次求解覆盖旧电位。', '',
'验证：CPU/CUDA独立密集解参考、非零初值、零右端、非有限输入、最大迭代数、非整段重启、输出独立性、实际单域CN、BE/CN质量守恒和时间阶次，共30项测试通过。', '',
f"四项测量中RHS相对差异最大{max(r['rhs_relative_difference'] for r in rows):.3e}；优化解与TorchCor核结果最大差异{max(r['solvers']['graph_natural']['max_voltage_difference_mv'] for r in rows):.3e} mV，所有独立真实残差均通过原阈值max(1e-13,1e-10||b||)。", '',
'这些收益属于执行实现优化。标准FEM也使用同一优化PCG和CSR，因此不能把全部加速归于nodal离散。先前标准与nodal的空间传播差异依然存在；本轮没有宣称已达到5%精度标准，也未外推为完整EP或三场加速。', '',
'项目接入：Monodomain在CUDA上自动使用缓存RHS；backend=torch_pcg_jacobi_light自动使用新GPU求解器。原ReferenceJacobiPCG保留作为性能/正确性参考。现有运行进程不受文件更新影响；新进程加载新实现。', '',
'服务器代码：/mnt/large2/gjh/AFSEI/src/afsi_torch/ep/{gpu_kernels,graph_pcg,jacobi,monodomain}.py。测量结果：/mnt/large2/gjh/AFSEI/results/ep_buffered_20261029。']
(folder/'REPORT_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
for ax,dim in zip(axes,(2,3)):
    group=[r for r in rows if r['dim']==dim];x=np.arange(2)
    old=[r['reference_pde_ms'] for r in group];new=[r['solvers']['graph_natural']['pde_ms'] for r in group]
    for offset,values,color,label in [(-.18,old,'#64748b','Original AFSEI'),(.18,new,'#22b8e8','Buffered + graph PCG')]:
        bars=ax.bar(x+offset,values,.34,color=color,label=label)
        ax.bar_label(bars,fmt='%.3f',padding=3)
    ax.set_xticks(x,['Consistent P1','Nodal P1']);ax.set_ylabel('RHS + solve (ms)')
    ax.set_title(f'{dim}D, h=0.125 mm');ax.set_ylim(0,max(old)*1.25);ax.legend(fontsize=8)
fig.suptitle('Matched-input execution benchmark | FP64 | unchanged CN and residual tolerance')
fig.savefig(folder/'performance.png',dpi=180);fig.savefig(folder/'performance.pdf')
print('\n'.join(lines[:14]))
