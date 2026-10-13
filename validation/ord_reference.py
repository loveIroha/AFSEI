"""Independent original generated-C reference for GPU port verification.

Downloaded reference and compiled binary remain machine-local under .deps.
CPU double arithmetic and a literal reference SDIRK validate the CUDA adapter.
"""
import ctypes
import hashlib
from pathlib import Path
import subprocess
import numpy as np

HASH='f6f2e979416149117029043af9718c7fefb590f0b1b05720c2d13847ae9cff92'


class Reference:
    def __init__(self,celltype=1):
        root=Path(__file__).resolve().parents[1]/'.deps/ord_reference'
        source=(root/'ord_generated.c').read_text()
        if hashlib.sha256(source.encode()).hexdigest()!=HASH:
            raise ValueError('wrong independent ORd reference source')
        # Native cell type enters BEFORE dependent conductances are evaluated.
        source=source.replace('CONSTANTS[0] = 0;',f'CONSTANTS[0] = {celltype};')
        path=root/f'reference_{celltype}.c'; library=root/f'reference_{celltype}.so'
        path.write_text('#include <math.h>\n'+source)
        subprocess.run(['cc','-O2','-fPIC','-shared',str(path),'-o',str(library),'-lm'],check=True,capture_output=True)
        self.library=ctypes.CDLL(str(library)); ptr=ctypes.POINTER(ctypes.c_double)
        self.library.initConsts.argtypes=[ptr]*3
        self.library.computeRates.argtypes=[ctypes.c_double,ptr,ptr,ptr,ptr]
        self.constants=np.zeros(139); self.initial=np.zeros(41); rates=np.zeros(41)
        self.library.initConsts(self.ptr(self.constants),self.ptr(rates),self.ptr(self.initial))
        self.constants[12]=0. # common external stimulus, not the source single-cell pulse

    @staticmethod
    def ptr(a): return a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    def rhs(self,y,voltage):
        states=np.r_[voltage,y].astype(np.float64); rates=np.zeros(41); algebraic=np.zeros(198)
        self.library.computeRates(1.,self.ptr(self.constants),self.ptr(rates),self.ptr(states),self.ptr(algebraic))
        return rates[1:],-rates[0]

    def advance(self,y,voltage,dt=.002,steps=25):
        state=np.array(y,copy=True); current=0.
        gamma=.788675134594812882254; a21=1-2*gamma
        for _ in range(steps):
            k1,_=self.rhs(state,voltage)
            for i in range(3):
                previous=k1.copy(); k1,_=self.rhs(state+dt*gamma*k1,voltage)
                if i>0 and np.max(np.abs(k1-previous)/np.maximum(np.abs(k1),1.))<1e-6: break
            k2=k1.copy()
            for i in range(3):
                previous=k2.copy(); k2,_=self.rhs(state+dt*(a21*k1+gamma*k2),voltage)
                if i>0 and np.max(np.abs(k2-previous)/np.maximum(np.abs(k2),1.))<1e-6: break
            state+=dt*.5*(k1+k2)
            _,value=self.rhs(state,voltage); current+=value
        return state,current/steps
