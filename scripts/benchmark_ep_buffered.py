"""Matched-matrix implementation benchmark, including CSR ordering ablation."""
import os,json,time,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.compare_ep_rectangles import mesh
from afsi_torch.ep.monodomain import Monodomain,MonodomainParameters
from afsi_torch.ep.numerics import MonodomainNumerics
from afsi_torch.ep.jacobi import ReferenceJacobiPCG,JacobiPCG
from afsi_torch.ep.graph_pcg import GraphJacobiPCG
from afsi_torch.ep.gpu_kernels import BufferedRHS,CSRAction
from afsi_torch.fluid.solvers import SolverOptions
from validation.torchcor_pde import load_core,TorchCorCG,PDEMethod,COMMIT

def timing(fn,n=20):
    for _ in range(3):fn()
    samples=[]
    for repeat in range(3):
        torch.cuda.synchronize();start=time.perf_counter()
        for _ in range(n):fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter()-start)*1000/n)
    return float(np.median(samples))

@torch.no_grad()
def main():
    root=Path('results/ep_buffered_20261029');root.mkdir(exist_ok=True)
    config=json.loads(Path('results/ep_controlled_h0p125_20261029/original_config.json').read_text())
    print(config.keys(),flush=True)
    # This is the exact saved benchmark config, not a new physical parameter set.
    torch.set_default_dtype(torch.float64)
    c=dict(config['domain'],parameters=config['parameters'],dt_ep_s=config['time']['dt_ep_s'])
    classes,_=load_core(Path('.deps')/('torchcor-'+COMMIT))
    results=[]
    for dim in (2,3):
        X,cells=mesh(dim,c);f=torch.zeros_like(X);s=f.clone();f[:,0]=1;s[:,1]=1
        parameters=MonodomainParameters(**{k:v for k,v in c['parameters'].items()})
        alpha=parameters.surface_to_volume_per_cm*parameters.capacitance_uf_per_cm2/c['dt_ep_s']
        states=np.load(f'results/ep_controlled_h0p125_20261029/{dim}d/consistent_tc/fields.npz')
        print('statekeys',states.files,flush=True)
        volts=states['voltage_snapshots_v']
        v=torch.tensor(volts[min(2,len(volts)-1)],device=X.device,dtype=X.dtype)
        i=torch.sin(v*30)*.1;stim=torch.zeros_like(v)
        o=SolverOptions(rtol=1e-10,atol=1e-13,max_iterations=2000,check_every=5)
        for nodal in (False,True):
            ep=Monodomain(X,cells,f,s,parameters=parameters,numerics=MonodomainNumerics(
                discretization='nodal_p1' if nodal else 'consistent_p1',backend='torch_pcg_jacobi_light'))
            _,A,_=ep._system(c['dt_ep_s']);K=ep.stiffness;M=ep.mass;D=alpha*ep.mass_diagonal+.5*ep.stiffness_diagonal
            tc=TorchCorCG(A,o,classes)
            old=PDEMethod(X,M,ep.mass_diagonal,A,parameters,c['dt_ep_s'],tc,nodal=nodal,stiffness=K,theta=.5)
            fused=BufferedRHS(M,ep.mass_diagonal,K,alpha,parameters.surface_to_volume_per_cm,.5,nodal)
            b=old.rhs(v,i,stim).clone();bnew=fused(v,i,stim)
            rhs_gap=float(torch.linalg.vector_norm(b-bnew)/torch.linalg.vector_norm(b))
            assert rhs_gap<1e-13,rhs_gap
            cal=tc.calibrate(b);reference=tc.solve(b,v,cal)[0].clone()
            solvers={'reference':ReferenceJacobiPCG(A,D,o),'graph_natural':JacobiPCG(A,D,o),
                     'graph_rcm':GraphJacobiPCG(A,D,o,ordering='rcm')}
            row={'dim':dim,'nodal':nodal,'nodes':len(X),'rhs_relative_difference':rhs_gap,
                 'rhs_old_ms':timing(lambda:old.rhs(v,i,stim)),
                 'rhs_buffered_ms':timing(lambda:fused(v,i,stim)),
                 'torchcor_solve_ms':timing(lambda:tc.solve(b,v,cal)),'solvers':{}}
            for name,solver in solvers.items():
                x,info=solver.solve(b,initial=v);saved=x.clone()
                residual=float(torch.linalg.vector_norm(b-A@x));tol=max(o.atol,o.rtol*float(torch.linalg.vector_norm(b)))
                error=float((x-reference).abs().max())
                assert residual<=tol and error<1e-8,(name,residual,tol,error)
                solver.solve(b,initial=v);assert torch.equal(x,saved),'published output alias'
                row['solvers'][name]={'solve_ms':timing(lambda:solver.solve(b,initial=v)),
                    'residual_ratio':residual/tol,'max_voltage_difference_mv':error*1000,
                    'iterations_bound':info.iterations}
            for name in ('graph_natural','graph_rcm'):
                solver=solvers[name]
                row['solvers'][name]['pde_ms']=timing(lambda:solver.solve(fused(v,i,stim),initial=v))
            row['reference_pde_ms']=timing(lambda:solvers['reference'].solve(old.rhs(v,i,stim),initial=v))
            row['torchcor_pde_ms']=timing(lambda:tc.solve(old.rhs(v,i,stim),v,cal))
            # Alternative cuSPARSE vs row-gather CSR; exact same K and V.
            out=torch.empty_like(v);action=CSRAction(K)
            row['csr_ms']=timing(lambda:action(v,out),100)
            row['cusparse_ms']=timing(lambda:K@v,100)
            results.append(row);print(json.dumps(row),flush=True)
            (root/'report.json').write_text(json.dumps(results,indent=2))
    print('COMPLETED',flush=True)

if __name__=='__main__':main()
