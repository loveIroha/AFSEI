import numpy as np
import pytest
from afsi_torch.simulation.tube_validation import propagation_metrics


def test_activation_velocity_ignores_stimulated_region_and_missing_sections():
    z=np.repeat(np.linspace(0,2,21),4)
    X=np.column_stack((z*0,z*0,z+.4));t=.005+z/40.
    t[z<.2]=0.;t[z>1.8]=np.inf
    r=propagation_metrics(X,t,.4,2.,.1)
    assert r['axial_conduction_speed_cm_s']==pytest.approx(40.)
    assert r['activation_fit_r_squared']==pytest.approx(1.)
    assert r['sections'][-1]['activation_mean_ms'] is None


def test_unexcited_tissue_does_not_report_zero_activation_or_speed():
    X=np.zeros((8,3));X[:,2]=np.arange(8)
    r=propagation_metrics(X,np.full(8,np.inf),0.,7.,1.)
    assert r['activated_fraction']==0
    assert r['first_activation_ms'] is None
    assert r['axial_conduction_speed_cm_s'] is None
