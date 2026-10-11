"""Compact cap operators equal the quadrature oracle, including work adjoints."""
import pytest
import torch
from test_real_lv import real_case
from test_afsei_mac_preload import mac_case
from test_afsei_mac_balanced_dynamic import dynamic_driver
from afsi_torch.ep.mac_cap_coupled import closed_rows
from afsi_torch.ep.mac_rcr_coupled import outlet_rows
from afsi_torch.ep.mac_compact_cap import CompactCapRows
from afsi_torch.mac.open_boundary import face_weights


@pytest.mark.parametrize('label',[0,1,2])
def test_gpu_compact_cap_oracle_and_adjoint(mac_case,label):
    d=dynamic_driver(mac_case); d.compact_cap_operators=False
    e=d._fsi_stepper(d.maximum_ticks); x=d.mesh.X*1.002
    ref=closed_rows(e,x) if label==0 else outlet_rows(e,x,label=label)
    rows=CompactCapRows(ref)
    pressure=3+torch.sin(torch.arange(rows.count,device=x.device,dtype=x.dtype))
    velocity=tuple(torch.sin(e.grid.coordinates(c,device=x.device)[...,c]) for c in range(3))
    material=torch.cos(x)
    for a,b in zip(rows.lift(pressure),ref.lift(pressure)):
        torch.testing.assert_close(a,b,rtol=2e-12,atol=1e-10)
    torch.testing.assert_close(rows.flow(velocity),ref.flow(velocity),rtol=2e-12,atol=1e-10)
    torch.testing.assert_close(rows.reaction(pressure),ref.reaction(pressure),rtol=2e-12,atol=1e-10)
    torch.testing.assert_close(rows.material_flow(material),ref.material_flow(material),rtol=2e-12,atol=1e-10)
    fluid=sum((u*f*face_weights(e.grid,c,f)).sum() for c,(u,f) in enumerate(zip(velocity,rows.lift(pressure))))
    solid=(rows.reaction(pressure)*material).sum()
    torch.testing.assert_close(fluid-solid,pressure@(rows.flow(velocity)-rows.material_flow(material)),rtol=2e-12,atol=1e-10)
