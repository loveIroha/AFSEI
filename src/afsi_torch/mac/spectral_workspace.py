"""Reusable buffers for the unchanged three-dimensional DST-II inverse.

FFT input/output shapes and transforms match SpectralDirichletPressure.
Only internal scratch is reused; inverse() returns borrowed scratch and its
caller must clone a published pressure. Solves are serial per workspace.
"""
from math import prod
import torch


class SpectralWorkspace:
    def __init__(self,solver):
        self.solver=solver
        shape=solver.grid.shape; count=prod(shape)
        like=solver.rhs
        self.real=like.new_empty(2*count)
        maximum=max((n+1)*(count//n) for n in shape)
        # The packing kernels address this real storage; FFT sees its complex view.
        self.frequency_raw=like.new_empty((maximum,2))
        self.frequency=torch.view_as_complex(self.frequency_raw)
        self.values=(torch.empty_like(like),torch.empty_like(like))
        self.real_views=[]; self.frequency_views=[]
        self.pack_forward=[]; self.unpack_forward=[]; self.pack_inverse=[]; self.unpack_inverse=[]
        for axis,n in enumerate(shape):
            sizes=list(shape); sizes[axis]=2*n
            real=self.real.view(sizes)
            sizes[axis]=n+1
            frequency=self.frequency[:prod(sizes)].view(sizes)
            self.real_views.append(real); self.frequency_views.append(frequency)
            self.pack_forward.append(self._forward_packer(axis))
            self.unpack_forward.append(self._forward_unpacker(axis))
            self.pack_inverse.append(self._inverse_packer(axis))
            self.unpack_inverse.append(self._inverse_unpacker(axis))

    def _forward_packer(self,axis):
        def kernel(value,packed):
            n=value.shape[axis]
            packed.narrow(axis,0,n).copy_(value)
            packed.narrow(axis,n,n).copy_(-value.flip((axis,)))
        return kernel

    def _forward_unpacker(self,axis):
        sine,cosine=self.solver.phases[axis]
        eigenvalue=self.solver.eigenvalue
        def kernel(real,imag,out):
            n=out.shape[axis]
            value=real.narrow(axis,1,n)*sine-imag.narrow(axis,1,n)*cosine
            # Last forward stage: fuse the existing eigenvalue division.
            out.copy_(value/eigenvalue if axis==0 else value)
        return kernel

    def _inverse_packer(self,axis):
        sine,cosine=self.solver.phases[axis]
        def kernel(value,real,imag):
            n=value.shape[axis]
            real.narrow(axis,0,1).zero_(); imag.narrow(axis,0,1).zero_()
            real.narrow(axis,1,n).copy_(value*sine)
            imag.narrow(axis,1,n).copy_(-value*cosine)
        return kernel

    def _inverse_unpacker(self,axis):
        def kernel(packed,out): out.copy_(packed.narrow(axis,0,out.shape[axis]))
        return kernel

    @property
    def storage_bytes(self):
        return sum(v.numel()*v.element_size() for v in (self.real,self.frequency,*self.values))

    @torch.no_grad()
    def inverse(self,rhs):
        if (rhs.shape!=self.solver.rhs.shape or rhs.dtype!=self.solver.rhs.dtype
                or rhs.device!=self.solver.rhs.device):
            raise ValueError('spectral RHS shape, dtype and device must match the workspace')
        cuda=rhs.device.type=='cuda'
        if cuda:
            from . import _triton_spectral as kernels
        value=rhs; stage=0
        for axis in (2,1,0):
            packed,frequency=self.real_views[axis],self.frequency_views[axis]
            out=self.values[stage%2]
            if cuda: kernels.forward_pack(value,packed,axis)
            else: self.pack_forward[axis](value,packed)
            torch.fft.rfft(packed,dim=axis,out=frequency)
            if cuda:
                kernels.forward_unpack(frequency,*self.solver.phases[axis],self.solver.eigenvalue,out,axis)
            else: self.unpack_forward[axis](frequency.real,frequency.imag,out)
            value=out; stage+=1
        for axis in (0,1,2):
            packed,frequency=self.real_views[axis],self.frequency_views[axis]
            out=self.values[stage%2]
            if cuda: kernels.inverse_pack(value,*self.solver.phases[axis],frequency,axis)
            else: self.pack_inverse[axis](value,frequency.real,frequency.imag)
            torch.fft.irfft(frequency,n=2*self.solver.grid.shape[axis],dim=axis,out=packed)
            if cuda: kernels.inverse_unpack(packed,out)
            else: self.unpack_inverse[axis](packed,out)
            value=out; stage+=1
        return value
