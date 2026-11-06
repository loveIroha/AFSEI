"""Algebraically identical GPB removable limits, Euler path only."""
from pathlib import Path
import hashlib,json
p=Path('/mnt/large2/gjh/AFSEI/src/afsi_torch/ep/csrc');source=p/'gpb_node_device.h';s=source.read_text()
helper='''
__device__ inline double gpb_euler_exprel(double x) {
 if(fabs(x)<1e-6){double x2=x*x;return 1.-x*.5+x2/12.-x2*x2/720.;}
 return x/expm1(x);
}
'''
pos=s.index('__device__');s=s[:pos]+helper+s[pos:]
replacements={
'0.32 * (Vm + 47.13) / (1.0 - exp(-0.1 * (Vm + 47.13)))':'3.2 * gpb_euler_exprel(-0.1 * (Vm + 47.13))',
'dss * (1.0 - exp(-(Vm + 5.0) / 6.0)) / (0.035 * (Vm + 5.0))':'dss / (0.21 * gpb_euler_exprel(-(Vm + 5.0) / 6.0))'}
for old,new in replacements.items():
    assert s.count(old)==1,old;s=s.replace(old,new)
start=s.index('    double ibarca_j =');end=s.index(';',s.index('    double ibarna_sl =',start))+1
s=s[:start]+'''    double ca_factor=2.0*Frdy*gpb_euler_exprel(2.0*Vm*FoRT);
    double mono_factor=Frdy*gpb_euler_exprel(Vm*FoRT);
    double ibarca_j=pCa*ca_factor*(0.341*y[35]*cache.math_0-0.341*Cao);
    double ibarca_sl=pCa*ca_factor*(0.341*y[36]*cache.math_0-0.341*Cao);
    double ibark=pK*mono_factor*(0.75*y[34]*cache.math_1-0.75*Ko);
    double ibarna_j=pNa*mono_factor*(0.75*y[31]*cache.math_1-0.75*Nao);
    double ibarna_sl=pNa*mono_factor*(0.75*y[32]*cache.math_1-0.75*Nao);'''+s[end:]
dest=p/'gpb_euler_device.h';dest.write_text(s)
(p/'gpb_euler_provenance.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),generated_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),change='Only exprel limits of late-Na gate, d gate and GHK currents; equations and parameters unchanged'),indent=2))
print('Generated Euler GPB stable removable limits')
