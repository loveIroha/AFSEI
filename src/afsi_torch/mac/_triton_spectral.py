"""Pack/unpack the existing Torch DST FFTs into caller-owned GPU buffers."""
import triton
import triton.language as tl


@triton.jit
def _coordinates(i, NY: tl.constexpr, NZ: tl.constexpr):
    return i // (NY * NZ), i // NZ % NY, i % NZ


@triton.jit
def _forward_pack(X, P, NX: tl.constexpr, NY: tl.constexpr, NZ: tl.constexpr,
                  S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr,
                  P0: tl.constexpr, P1: tl.constexpr, P2: tl.constexpr,
                  AXIS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < NX * NY * NZ
    x, y, z = _coordinates(i, NY, NZ)
    value = tl.load(X + x*S0 + y*S1 + z*S2, valid, 0)
    tl.store(P + x*P0 + y*P1 + z*P2, value, valid)
    if AXIS == 0: x = 2*NX - 1 - x
    elif AXIS == 1: y = 2*NY - 1 - y
    else: z = 2*NZ - 1 - z
    tl.store(P + x*P0 + y*P1 + z*P2, -value, valid)


@triton.jit
def _forward_unpack(F, S, C, E, O, NX: tl.constexpr, NY: tl.constexpr, NZ: tl.constexpr,
                    F0: tl.constexpr, F1: tl.constexpr, F2: tl.constexpr,
                    AXIS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < NX * NY * NZ
    x, y, z = _coordinates(i, NY, NZ)
    if AXIS == 0: k = x; x += 1
    elif AXIS == 1: k = y; y += 1
    else: k = z; z += 1
    position = 2*(x*F0 + y*F1 + z*F2)
    real = tl.load(F + position, valid, 0)
    imag = tl.load(F + position + 1, valid, 0)
    value = real*tl.load(S+k, valid, 0) - imag*tl.load(C+k, valid, 0)
    if AXIS == 0: value = value/tl.load(E+i, valid, 1)
    tl.store(O+i, value, valid)


@triton.jit
def _inverse_pack(X, S, C, F, NX: tl.constexpr, NY: tl.constexpr, NZ: tl.constexpr,
                  S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr,
                  F0: tl.constexpr, F1: tl.constexpr, F2: tl.constexpr,
                  AXIS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < NX * NY * NZ
    x, y, z = _coordinates(i, NY, NZ)
    value = tl.load(X+x*S0+y*S1+z*S2, valid, 0)
    if AXIS == 0: k = x; x += 1; stride = F0
    elif AXIS == 1: k = y; y += 1; stride = F1
    else: k = z; z += 1; stride = F2
    position = 2*(x*F0+y*F1+z*F2)
    tl.store(F+position, value*tl.load(S+k, valid, 0), valid)
    tl.store(F+position+1, -value*tl.load(C+k, valid, 0), valid)
    tl.store(F+position-2*stride, 0., valid & (k == 0))
    tl.store(F+position-2*stride+1, 0., valid & (k == 0))


@triton.jit
def _inverse_unpack(P, O, NX: tl.constexpr, NY: tl.constexpr, NZ: tl.constexpr,
                    P0: tl.constexpr, P1: tl.constexpr, P2: tl.constexpr,
                    BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < NX*NY*NZ
    x, y, z = _coordinates(i, NY, NZ)
    tl.store(O+i, tl.load(P+x*P0+y*P1+z*P2, valid, 0), valid)


def _launch(kernel, count, *args):
    kernel[(triton.cdiv(count, 256),)](*args, BLOCK=256, num_warps=4, enable_fp_fusion=False)


def forward_pack(value, packed, axis):
    _launch(_forward_pack, value.numel(), value, packed, *value.shape,
            *value.stride(), *packed.stride(), axis)


def forward_unpack(frequency, sine, cosine, eigenvalue, out, axis):
    _launch(_forward_unpack, out.numel(), frequency.real, sine, cosine, eigenvalue, out,
            *out.shape, *frequency.stride(), axis)


def inverse_pack(value, sine, cosine, frequency, axis):
    _launch(_inverse_pack, value.numel(), value, sine, cosine, frequency.real,
            *value.shape, *value.stride(), *frequency.stride(), axis)


def inverse_unpack(packed, out):
    _launch(_inverse_unpack, out.numel(), packed, out, *out.shape, *packed.stride())
