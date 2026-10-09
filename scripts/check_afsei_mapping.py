"""Compare PyTorch source mapping with the actual serial DOLFIN oracle."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from afsi_torch.paper_lv import PaperLVConfig,imported_model
from afsi_torch.ep.mapping import source_p1_fields


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--oracle-csv',type=Path,required=True)
    parser.add_argument('--source-dir',default='/mnt/large2/gjh/realistic_left_ventricle')
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    oracle = np.loadtxt(args.oracle_csv,delimiter=',')
    model = imported_model(PaperLVConfig(source_dir=args.source_dir))
    mesh = model.mesh
    f,s,mask,owner = source_p1_fields(mesh)
    # Match by coordinates; source and imported local vertex numbering need
    # not agree. Sorted coordinate keys are unambiguous on this reference mesh.
    raw = mesh.X.numpy()
    a = np.lexsort(oracle[:,1:4].T[::-1]); b = np.lexsort(raw.T[::-1])
    if (len(oracle) != len(raw) or not np.allclose(oracle[a,1:4],raw[b],rtol=0,atol=1e-12)):
        raise ValueError('reference and AFSEI meshes do not have identical vertices')
    compared = oracle[a]
    f_error = np.max(np.abs(f.numpy()[b]-compared[:,4:7]))
    s_error = np.max(np.abs(s.numpy()[b]-compared[:,7:10]))
    owner_mismatch = np.count_nonzero(owner.numpy()[b] != compared[:,10].astype(np.int64))
    mask_mismatch = np.count_nonzero(mask.numpy()[b] != compared[:,11].astype(bool))
    passed = f_error < 1e-12 and s_error < 1e-12 and owner_mismatch == 0 and mask_mismatch == 0
    report = dict(passed=bool(passed),scope='actual serial DOLFIN 2019.1 CG1 interpolation comparison',
        node_count=len(raw),cell_count=len(mesh.cells),fiber_max_error=float(f_error),sheet_max_error=float(s_error),
        owner_mismatch_count=int(owner_mismatch),stimulus_mask_mismatch_count=int(mask_mismatch),
        source_stimulated_nodes=int(mask.sum()),endo_surface_nodes=len(torch.unique(model.endo_faces)),
        mapping='last incident source cell; not averaging',parallel_source_mapping_verified=False)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if not passed: raise SystemExit('DOLFIN mapping comparison failed')


if __name__ == '__main__': main()
