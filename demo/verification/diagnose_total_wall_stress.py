"""Saved 120 ms states: extra elastic stress plus Eulerian pressure/viscosity.

No PDE/ODE advancement. Centered VTI gradients approximate the viscous term.
"""
from pathlib import Path
import json,base64,zlib,xml.etree.ElementTree as ET
import numpy as np,torch,torch.nn.functional as nnf,meshio
from afsi_torch.solids.circular_tube import generate_circular_tube
from afsi_torch.solids.refined_circular_tube import refine_circular_tube
from afsi_torch.simulation.circular_tube_config import resolve_config
from afsi_torch.ep.circular_tube import CircularTubeSolid
from afsi_torch.ep.tissue import TissueState

ROOT=Path('/mnt/large2/gjh/AFSEI/results');OUT=ROOT/'solid_space_campaign_20261025'

def vti(path,shape):
    result={}
    for a in ET.parse(path).findall('.//CellData/DataArray'):
        s=a.text.strip();n=int(np.frombuffer(base64.b64decode(s[:16])[:8],dtype='<u8')[0])
        k=4*((8*(3+n)+2)//3);header=np.frombuffer(base64.b64decode(s[:k]),dtype='<u8')
        raw=base64.b64decode(s[k:]);i=0;chunks=[]
        for size in header[3:]:
            size=int(size);chunks.append(zlib.decompress(raw[i:i+size]));i+=size
        v=np.frombuffer(b''.join(chunks),dtype='<f8').copy()
        components=int(a.attrib.get('NumberOfComponents',1))
        if components==1:v=v.reshape(tuple(shape)[::-1]).transpose(2,1,0)
        else:v=v.reshape((*tuple(shape)[::-1],components)).transpose(2,1,0,3)
        result[a.attrib['Name']]=v
    return result

@torch.no_grad()
def main():
    device='cuda:0';torch.set_num_threads(2)
    cases={
      'Solid L0 / MAC32':ROOT/'single_tube_full_20261024/active_120ms',
      'Solid L0 / MAC64':ROOT/'reliability_campaign_20261025/mac64_budget16m',
      'Solid L1 / MAC32':OUT/'solid2_mac32/continued_300ms',
      'Solid L1 / MAC64':OUT/'solid2_mac64/continued_300ms'}
    source=np.load(ROOT/'single_tube_full_20261024/active_120ms/final_state.npz')
    result={}
    for label,folder in cases.items():
        c=json.loads((folder/'config.resolved.json').read_text());g,_,_,_=resolve_config(c)
        mesh=generate_circular_tube(g,device=device);transfer=None
        if c.get('solid_refinement_factor',1)==2:mesh,transfer=refine_circular_tube(mesh,g)
        solid=CircularTubeSolid(mesh,g,**{k:v for k,v in c['solid'].items() if k!='end_condition'},
            **{k:v for k,v in c['active'].items() if k not in ('mechanism','rest_reference')},
            dt_s=c['time']['fsi_dt_s'],end_support=c['solid']['end_condition'])
        solid.field_transfer=transfer
        get=lambda k:torch.as_tensor(source[k].copy(),device=device,dtype=mesh.X.dtype)
        tissue=TissueState(get('voltage_v'),get('gpb'),get('land'))
        solid.set_reference(get('resting_tension_dyn_cm2'))
        x=torch.as_tensor(meshio.read(folder/'vtk/solid_002400.vtu').points.copy(),device=device,dtype=mesh.X.dtype)
        solid.stage(x,tissue,get('zs'),get('zw'))
        F=solid.element_gradient(x);J=torch.linalg.det(F)
        P=torch.vmap(lambda f,*v:solid.stress(f,v,x.new_tensor(.12)))(F,*solid.cell_fields)
        tangent=F[:,:,2];tangent=tangent/tangent.norm(dim=1,keepdim=True)
        axial=(torch.einsum('ei,eij->ej',tangent,P)*torch.einsum('ei,eij->ej',tangent,F)).sum(1)/J
        extra_z=(P[:,2,:]*F[:,2,:]).sum(1)/J
        points=torch.einsum('qa,eai->eqi',solid.geometry.values,x[mesh.cells])
        origin=x.new_tensor(c['fluid']['origin_cm']);length=x.new_tensor(c['fluid']['lengths_cm'])
        query=(2*(points-origin)/length-1).flip(-1)[None,:,:,None,:]
        weights=solid.geometry.weights/solid.volumes[:,None]
        def sample(values):
            field=torch.as_tensor(np.ascontiguousarray(values),device=device,dtype=x.dtype)
            pq=nnf.grid_sample(field[None,None],query,mode='bilinear',padding_mode='border',align_corners=False)[0,0,:,:,0]
            return (pq*weights).sum(1)
        fields=vti(folder/'vtk/fluid_002400.vti',c['fluid']['shape'])
        pressure=sample(fields['pressure_dyn_per_cm2'])
        u=torch.as_tensor(fields['velocity_cm_per_s'].copy(),device=device,dtype=x.dtype)
        spacing=np.array(c['fluid']['lengths_cm'])/c['fluid']['shape']
        grad=[[torch.gradient(u[...,i],spacing=(float(spacing[j]),),dim=(j,),edge_order=1)[0] for j in range(3)] for i in range(3)]
        viscous=torch.zeros_like(axial)
        for i in range(3):
            for j in range(i,3):
                dij=(grad[i][j]+grad[j][i])*.5
                dmean=sample(dij.cpu().numpy())
                viscous+=2*c['fluid']['viscosity_dyn_s_per_cm2']*dmean*tangent[:,i]*tangent[:,j]*(1 if i==j else 2)
        total=axial-pressure+viscous
        zref=mesh.X[mesh.cells,2].mean(1)-g.axial_start_cm
        central=(zref>=.2*g.length_cm)&(zref<=.8*g.length_cm)
        w=(solid.volumes*J)[central]
        mean=lambda value:float((value[central]*w).sum()/w.sum()/1e4)
        result[label]=dict(time_ms=120,mean_extra_global_z_kpa=mean(extra_z),
            mean_extra_material_axial_kpa=mean(axial),mean_pressure_kpa=mean(pressure),
            mean_viscous_material_axial_kpa=mean(viscous),mean_estimated_total_axial_kpa=mean(total),
            compressed_current_volume_fraction=float(((total[central]<0)*w).sum()/w.sum()),
            diagnostic_scope='middle 60% in reference z, current-volume weighted; material longitudinal direction; pressure integrated at mechanical quadrature; viscous term from centered VTI gradients')
        print(label,json.dumps(result[label]),flush=True)
        del solid,mesh,F,P,x,points,query,fields,grad,u
    report=dict(results=result,scope='matched saved 120 ms states; same independent isometric EP history; no stepping and no dynamic stability proof',
                previous_diagnostic_limitation='Prior material-only stress and tangent omitted Eulerian pressure and fluid response; it cannot rule out axial compression of the total stress.')
    (OUT/'total_wall_stress_120ms.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
