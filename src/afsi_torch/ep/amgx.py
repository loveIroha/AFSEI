"""Optional FP64 AmgX PCG-AMG through the stable C API.

Only scalar diagnostics reach the host during stepping. Matrix upload/setup
is cached by Monodomain. Device pointers are copied to AmgX-owned storage:
this is GPU-to-GPU exchange, not a claim of shared/zero-copy storage.
"""
import atexit
import ctypes as C
import json
import os
from pathlib import Path
import sys
import weakref
from math import isfinite
import torch
from ..fluid.solvers import SolveInfo


_api = None


def _library_path():
    explicit = os.environ.get('AFSEI_AMGX_LIBRARY')
    if explicit:
        return Path(explicit)
    project = Path(__file__).resolve().parents[3]
    for candidate in (project/'.deps/amgx/build/libamgxsh.so',
                      Path(sys.prefix)/'lib/libamgxsh.so'):
        if candidate.is_file():
            return candidate
    raise RuntimeError('AmgX is selected but libamgxsh.so is unavailable. Run '
        'python scripts/install_afsei_amgx.py, or set AFSEI_AMGX_LIBRARY. '
        'Use ep_numerics.backend= torch_pcg_jacobi for an explicit comparison.')


class _API:
    def __init__(self):
        path = _library_path()
        self.lib = C.CDLL(str(path), mode=C.RTLD_GLOBAL)
        # A tiny companion uses the named enum from the build's actual header;
        # no assumption about numeric AMGX_mode values is made in Python.
        mode = C.CDLL(str(path.with_name('libafsei_amgx_mode.so')))
        mode.afsei_amgx_mode_dDDI.restype = C.c_int
        self.mode = mode.afsei_amgx_mode_dDDI()
        H, I, P = C.c_void_p, C.c_int, C.c_void_p
        signatures = {
            'initialize': [], 'finalize': [],
            'get_error_string': [I, P, I],
            'config_create': [C.POINTER(H), C.c_char_p],
            'config_destroy': [H],
            'resources_create': [C.POINTER(H), H, P, I, C.POINTER(I)],
            'resources_destroy': [H],
            'matrix_create': [C.POINTER(H), H, I], 'matrix_destroy': [H],
            'matrix_upload_all': [H, I, I, I, I, P, P, P, P],
            'vector_create': [C.POINTER(H), H, I], 'vector_destroy': [H],
            'vector_upload': [H, I, I, P], 'vector_download': [H, P],
            'vector_bind': [H, H],
            'solver_create': [C.POINTER(H), H, I, H], 'solver_destroy': [H],
            'solver_setup': [H, H], 'solver_solve': [H, H, H],
            'solver_get_status': [H, C.POINTER(I)],
            'solver_get_iterations_number': [H, C.POINTER(I)],
        }
        for name, args in signatures.items():
            fn = getattr(self.lib, 'AMGX_'+name)
            fn.argtypes, fn.restype = args, I
        self.call('initialize')
        self.instances = weakref.WeakSet()
        self.resource = self.resource_config = None
        self.device_index = None
        atexit.register(self.finish)

    def call(self, name, *args):
        code = getattr(self.lib, 'AMGX_'+name)(*args)
        if code:
            message = C.create_string_buffer(4096)
            self.lib.AMGX_get_error_string(code, message, len(message))
            raise RuntimeError(f'AmgX {name}: {message.value.decode(errors="replace")} (code {code})')

    def finish(self):
        for obj in list(self.instances):
            obj.close()
        if self.resource is not None:
            self.call('resources_destroy', self.resource)
            self.call('config_destroy', self.resource_config)
        self.lib.AMGX_finalize()

    def resource_for(self, device_index, configuration):
        # AmgX Resources destruction releases GLOBAL pools/cuBLAS handles.
        # Independent Resources for simultaneous checkpoint drivers can free
        # each other's allocations. Share one process-lifetime resource.
        if self.resource is not None:
            if device_index != self.device_index:
                raise ValueError('AFSEI AmgX supports one CUDA device per process')
            return self.resource
        config, resource = C.c_void_p(), C.c_void_p()
        self.call('config_create', C.byref(config), configuration)
        device = C.c_int(device_index)
        try:
            self.call('resources_create', C.byref(resource), config, None, 1, C.byref(device))
        except BaseException:
            self.call('config_destroy', config)
            raise
        self.resource_config, self.resource = config, resource
        self.device_index = device_index
        return resource


class AmgXPCG:
    def __init__(self, matrix, options, *, profile='size2'):
        if profile not in ('size2','size4','size8','size8_two_level'):
            raise ValueError('unsupported AmgX profile')
        self.profile=profile
        if matrix.layout != torch.sparse_csr or not matrix.is_cuda or matrix.dtype != torch.float64:
            raise ValueError('AmgX EP requires a CUDA FP64 CSR matrix')
        self.matrix, self.options = matrix, options
        self.device, self.n = matrix.device, matrix.shape[0]
        self.handles = []
        self.setup_count = 0
        self.solve_count = 0
        global _api
        with torch.cuda.device(self.device):
            # One-time allocator ownership boundary: AmgX cudaMalloc cannot
            # reclaim PyTorch's inactive assembly cache. Live tensors stay owned.
            torch.cuda.empty_cache()
            if _api is None:
                _api = _API()
            self.api = _api
            self.api.instances.add(self)
            # int32 conversion and host CSR indices are startup-only. Values
            # stay on device. The single-GPU C API takes 32-bit CSR indices.
            if matrix.values().numel() > 2**31-1 or self.n >= 2**31-1:
                raise ValueError('AmgX single-GPU CSR exceeds int32 range')
            row = matrix.crow_indices().to(dtype=torch.int32, device='cpu').contiguous()
            col = matrix.col_indices().to(dtype=torch.int32, device='cpu').contiguous()
            self.values = matrix.values().contiguous()
            self.rhs_work = matrix.values().new_empty(self.n)
            self.x_work = torch.empty_like(self.rhs_work)
            self.tau = options.rtol if options.rtol > 0 else options.atol
            cfg = dict(config_version=2, determinism_flag=1, exception_handling=1,
                solver=dict(solver='PCG', scope='main', norm='L2',
                    convergence='ABSOLUTE', tolerance=self.tau*.25,
                    max_iters=options.max_iterations, monitor_residual=1,
                    print_solve_stats=0, store_res_history=0,
                    preconditioner=dict(solver='AMG', scope='amg',
                        algorithm='AGGREGATION', selector='SIZE_'+profile[4], interpolator='D2',
                        cycle='V', max_iters=1, max_levels=20, min_coarse_rows=32,
                        presweeps=1, postsweeps=1,
                        coarse_solver='DENSE_LU_SOLVER', monitor_residual=0,
                        print_solve_stats=0, print_grid_stats=0,
                        smoother=dict(solver='BLOCK_JACOBI', scope='jacobi',
                            relaxation_factor=.5, monitor_residual=0,
                            print_solve_stats=0))))
            if profile=='size8_two_level':
                preconditioner=cfg['solver']['preconditioner']
                preconditioner['max_levels']=2
                preconditioner['coarse_solver']=dict(solver='BLOCK_JACOBI',scope='coarse',
                    max_iters=4,relaxation_factor=.5,monitor_residual=0,print_solve_stats=0)
            try:
                configuration = json.dumps(cfg).encode()
                self.config = self._create('config', configuration)
                self.resource = self.api.resource_for(self.device.index, configuration)
                self.amatrix = self._create('matrix', self.resource, self.api.mode)
                self.rhs = self._create('vector', self.resource, self.api.mode)
                self.solution = self._create('vector', self.resource, self.api.mode)
                self.solver = self._create('solver', self.resource, self.api.mode, self.config)
                # AmgX's C API does not expose a PyTorch stream parameter.
                # Synchronize the producer stream at the ownership boundary.
                torch.cuda.current_stream(self.device).synchronize()
                self.api.call('matrix_upload_all', self.amatrix, self.n,
                    self.values.numel(), 1, 1, row.data_ptr(), col.data_ptr(),
                    self.values.data_ptr(), None)
                for vec in (self.rhs, self.solution):
                    self.api.call('vector_bind', vec, self.amatrix)
                self.api.call('solver_setup', self.solver, self.amatrix)
                self.setup_count = 1
            except BaseException:
                self.close()
                raise

    def _create(self, kind, *args):
        handle = C.c_void_p()
        self.api.call(kind+'_create', C.byref(handle), *args)
        self.handles.append((kind, handle))
        return handle

    @torch.no_grad()
    def solve(self, rhs, *, initial):
        if not self.handles:
            raise RuntimeError('AmgX solver is closed')
        if any(v.shape != (self.n,) or v.device != self.device or v.dtype != torch.float64
               for v in (rhs, initial)):
            raise ValueError('AmgX vectors must match the matrix device, size and FP64 dtype')
        with torch.cuda.device(self.device):
            mv = lambda v: torch.sparse.mm(self.matrix, v[:, None]).flatten()
            rhs_norm, residual_norm = torch.stack((torch.linalg.vector_norm(rhs),
                torch.linalg.vector_norm(rhs-mv(initial)))).tolist()
            if not isfinite(rhs_norm) or not isfinite(residual_norm):
                raise RuntimeError('nonfinite AmgX EP input/residual')
            tolerance = max(self.options.atol, self.options.rtol*rhs_norm)
            if residual_norm <= tolerance:
                return initial.clone(), SolveInfo(0, residual_norm, rhs_norm, tolerance)
            # Normalize so ABSOLUTE convergence reproduces the project's
            # max(atol, rtol*||b||) criterion, independent of initial residual.
            scale = tolerance/self.tau
            torch.div(rhs, scale, out=self.rhs_work)
            torch.div(initial, scale, out=self.x_work)
            torch.cuda.current_stream(self.device).synchronize()
            self.api.call('vector_upload', self.rhs, self.n, 1, self.rhs_work.data_ptr())
            self.api.call('vector_upload', self.solution, self.n, 1, self.x_work.data_ptr())
            self.api.call('solver_solve', self.solver, self.rhs, self.solution)
            self.solve_count += 1
            self.api.call('vector_download', self.solution, self.x_work.data_ptr())
            # Establish completion before PyTorch consumes AmgX output.
            torch.cuda.synchronize(self.device)
            result = self.x_work*scale  # independent accepted/trial-state storage
            residual_norm = torch.linalg.vector_norm(rhs-mv(result)).item()
            status, iterations = C.c_int(), C.c_int()
            self.api.call('solver_get_status', self.solver, C.byref(status))
            self.api.call('solver_get_iterations_number', self.solver, C.byref(iterations))
            if not isfinite(residual_norm) or residual_norm > tolerance or status.value in (1, 2):
                raise RuntimeError(f'AmgX EP solve failed: status={status.value}, '
                    f'true residual {residual_norm:.6e} > tolerance {tolerance:.6e}')
            return result, SolveInfo(iterations.value, residual_norm, rhs_norm, tolerance)

    def close(self):
        if not hasattr(self, 'api'):
            return
        while self.handles:
            kind, handle = self.handles.pop()
            self.api.call(kind+'_destroy', handle)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
