"""Generate allocation-free device arithmetic from the pinned C++ source.

No parameter substitution or algebraic simplification is performed. The
original source files remain separate, enabling independent CPU comparisons.
Run from any directory; output belongs to src/afsi_torch/ep/csrc.
"""
from pathlib import Path
import hashlib
import json
import re

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1]/'src/afsi_torch/ep/csrc'
COMMIT = '5bdde73c6be10cefadb005a0fcec5fdb6521f336'


def function_body(text, marker):
    start = text.index('{', text.index(marker))
    depth = 1
    end = start+1
    while depth:
        depth += (text[end] == '{')-(text[end] == '}')
        end += 1
    return text[start+1:end-1]


def main():
    cpp = (HERE/'GPB_cell_model.cpp').read_text()
    body = function_body(cpp, 'GPBCellModel::f_heartfailure_1(')
    assert body.count('Result result;') == 1
    assert body.count('result.ydot.resize(FULL_ODE_DIM, 0.0);') == 1
    body = body.replace('Result result;', 'double I_ion;')
    body = body.replace('result.ydot.resize(FULL_ODE_DIM, 0.0);',
                        'for (int i=0; i<58; ++i) ydot[i]=0.0;')
    body = body.replace('result.ydot', 'ydot').replace('result.I_ion', 'I_ion')
    body = body.replace('return result;', 'return I_ion;')
    assert 'std::vector' not in body and 'result.ydot' not in body and 'result.I_ion' not in body
    text = '// Generated from GPB_cell_model.cpp at '+COMMIT+'; do not hand edit.\n'
    text += '#pragma once\n#include <cmath>\n'
    text += '__device__ inline double gpb_full_rhs(double t, double* y, double Vm, double I_app, double* ydot) {\n'
    text += body+'\n}\n'
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'gpb_device.h').write_text(text)

    tissue = (HERE/'GPBTissueManager_Land.h').read_text()
    params = function_body(tissue, 'namespace LandParams')
    state = function_body(tissue, 'struct LandState')
    land = function_body(tissue, 'static void update_land_ode(')
    text = '// Generated from GPBTissueManager_Land.h at '+COMMIT+'; do not hand edit.\n'
    text += '#pragma once\n#include <cmath>\n#include <algorithm>\n'
    text += '#ifdef __CUDACC__\n#define AFSEI_LAND_DEVICE __host__ __device__\n#else\n#define AFSEI_LAND_DEVICE\n#endif\n'
    text += 'namespace LandParams {\n'+params+'\n}\nstruct LandState {\n'+state+'\n};\n'
    text += 'AFSEI_LAND_DEVICE inline void update_land_ode(LandState& s, double Ca_i_uM, double lmbda, double Zetas, double Zetaw, double dt_ms) {\n'
    text += land+'\n}\n#undef AFSEI_LAND_DEVICE\n'
    (OUT/'land_device.h').write_text(text)
    initial = function_body(cpp, 'GPBCellModel::get_initial_state()')
    match = re.search(r'y0_full\s*=\s*\{([^}]+)\}', initial)
    full = [float(v.strip()) for v in match[1].split(',') if v.strip()]
    # The source declares 58 entries but supplies only 57, then reads [57].
    # Full state 57 is unused (its derivative is zero); make it deterministic.
    source_initial_entries = len(full)
    assert source_initial_entries == 57 and 'ydot[i] = 0.0' in body
    full.append(0.0)
    assert len(full) == 58
    (OUT.parent/'initial.py').write_text(
        '# Generated from the pinned GPB initial state; voltage is separate.\n'
        'RESTING_VOLTAGE_V = '+repr(full[38]/1000)+'\n'
        'GPB_INITIAL = '+repr(tuple(full[:38]+full[39:]))+'\n')
    (HERE/'provenance.json').write_text(json.dumps({
        'repository': 'https://github.com/loveIroha/test_ele_soild_heart',
        'commit': COMMIT,
        'sha256': {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                   for name in ('GPB_cell_model.cpp', 'GPB_cell_model.h',
                                'SDIRK3_Solver.h', 'GPBTissueManager_Land.h', 'Active_land.ufl')},
        'mechanical_source_file': 'Active_land.ufl',
        'source_initial_entries': source_initial_entries,
        'initial_state_correction': 'Missing unused full state 57 explicitly initialized to zero; source reads out of bounds',
        'scope': 'Numerical components; full EP/FSI cycle and physical valve ports pending'
    }, indent=2)+'\n')
    print('Generated pinned GPB/Land device arithmetic and initial state')


if __name__ == '__main__':
    main()
