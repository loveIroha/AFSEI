"""Export unchanged physical FE meshes and the exact coarse EP field map.

These are input data for the native IBAMR driver, not AFSEI discrete operators.
The short constant-active pilot is a separate, explicitly prescribed FSI IVP.
"""
from pathlib import Path
import argparse,json,struct,hashlib
import numpy as np,torch
from afsi_torch.solids.circular_tube import CircularTubeGeometry,generate_circular_tube
from afsi_torch.solids.refined_circular_tube import refine_circular_tube

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    args=p.parse_args();torch.set_num_threads(2);c=json.loads(args.config.read_text());g=CircularTubeGeometry(**c['geometry'])
    base=generate_circular_tube(g);out=args.output;out.mkdir(parents=True,exist_ok=True)
    for factor in (1,2,3):
        mesh=base if factor==1 else refine_circular_tube(base,g,factor)[0]
        with (out/f'mesh_solid{factor}.bin').open('wb') as f:
            f.write(struct.pack('<4Q',len(mesh.X),len(mesh.cells),len(base.X),len(base.cells)))
            f.write(struct.pack('<Q',factor))
            for a,dt in ((mesh.X.numpy(),'<f8'),(mesh.cells.numpy(),'<u8'),(mesh.fiber.numpy(),'<f8'),
                         (base.X.numpy(),'<f8'),(base.cells.numpy(),'<u8')):
                f.write(np.asarray(a,dtype=dt).tobytes())
    seed=Path('/mnt/large2/gjh/AFSEI/results/volumetric_stabilization_20261025/from_rest_300ms/solid1/prefix_60ms/final_state.npz')
    a=np.load(seed);rest=a['resting_tension_dyn_cm2'];xs=a['land'][:,0]
    assert len(xs)==len(base.X)
    with (out/'constant_60ms_active.bin').open('wb') as f:
        f.write(struct.pack('<QQd',len(xs),1,c['time']['ep_dt_s']))
        f.write(np.asarray(rest,dtype='<f8').tobytes());f.write(np.asarray(xs,dtype='<f8').tobytes())
    np.savez(out/'constant_60ms_active.npz',resting_tension_dyn_cm2=rest,land_xs=xs,
        land=a['land'],zs=a['zs'],zw=a['zw'],reference=base.X.numpy(),cells=base.cells.numpy())
    manifest=dict(physical_geometry_unchanged=True,EP_mesh_refined=False,
        stress='same passive Neo-Hookean plus circumferential nominal Land PK1, projected deviatoric plus kappa log(J) F^-T',
        end_spring='same finite full-vector reference end-surface spring',
        constant_pilot='new FSI IVP at reference geometry and zero fluid, constant active amplitude from 60ms electrical checkpoint; not a mechanical restart or full EP simulation',
        constant_seed=str(seed),files={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(p.open('rb'),'sha256').hexdigest()) for p in out.glob('*.bin')})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
