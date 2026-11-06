"""Compare the migrated RCR/valve with the ORIGINAL solid-project C++ header."""
import argparse,json,hashlib,subprocess,tempfile
from pathlib import Path
import numpy as np
from afsi_torch.ep.source_windkessel import pressure_driven_trial

parser=argparse.ArgumentParser()
parser.add_argument('--header',default='/mnt/large2/gjh/test_ele_soild_heart-test/windkessel.h')
parser.add_argument('--output',default='results/source_lv_validation_20261103/cpp_comparison.json')
args=parser.parse_args();header=Path(args.header).resolve()
cpp='''#include <iostream>
#include <iomanip>
#include "windkessel.h"
int main() {
  dolfin::Windkessel3Element wk;
  double oldpc,p,dt;
  std::cout << std::setprecision(17);
  while(std::cin >> oldpc >> p >> dt) {
    auto r=wk.evaluate_pressure_driven(p,oldpc,dt);
    std::cout << r.p_c << " " << r.p_ao << " " << r.Q_av << "\\n";
  }
}
'''
cases=[(pc,p,dt) for pc in (8.,40.,85.,120.) for dt in (5e-5,1e-4,.001,.01)
       for p in (0.,8.,pc-.001,pc,pc+.001,pc+.1,pc+1,pc+20,pc+100)]
with tempfile.TemporaryDirectory(prefix='afsei_source_wk_') as work:
    src=Path(work)/'check.cpp';exe=Path(work)/'check';src.write_text(cpp)
    subprocess.run(['g++','-std=c++17','-O2','-I',str(header.parent),str(src),'-o',str(exe)],check=True)
    output=subprocess.check_output([str(exe)],input=''.join(f'{pc:.17g} {p:.17g} {dt:.17g}\n'
        for pc,p,dt in cases),text=True)
reference=np.array([[float(v) for v in line.split()] for line in output.splitlines()])
python=[]
for pc,p,dt in cases:
    trial=pressure_driven_trial(pc,p,dt);python.append([trial.pc_mmhg,trial.arterial_mmhg,trial.flow_ml_s])
error=np.abs(reference-np.array(python));passed=bool(np.allclose(reference,python,rtol=1e-9,atol=1e-8))
report=dict(passed=passed,cases=len(cases),original_header=str(header),
    original_header_sha256=hashlib.sha256(header.read_bytes()).hexdigest(),
    max_abs_error=dict(zip(('pc_mmhg','pao_mmhg','flow_ml_s'),error.max(0).tolist())))
path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
path.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if not passed:raise SystemExit(1)
