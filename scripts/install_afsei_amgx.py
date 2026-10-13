"""Build the pinned optional AmgX backend without changing the conda stack."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

REVISION = '91a8413ef267b1c32aff4014c02820e1c5897ac2'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=int, default=8)
    parser.add_argument('--cuda-root', type=Path, default=Path(sys.prefix))
    args = parser.parse_args()
    if sys.platform != 'linux' or args.jobs < 1:
        parser.error('Linux and a positive job count required')
    project = Path(__file__).resolve().parents[1]
    root = project/'.deps/amgx'
    source, build = root/'source', root/'build'
    root.mkdir(parents=True, exist_ok=True)
    run = lambda argv, **kw: subprocess.run([str(v) for v in argv], check=True, **kw)
    if not source.exists():
        run(['git', 'clone', 'https://github.com/NVIDIA/AMGX.git', source])
    actual = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != REVISION:
        run(['git', '-C', source, 'fetch', 'origin', REVISION])
        run(['git', '-C', source, 'checkout', '--detach', REVISION])
    env = dict(os.environ, PATH=str(args.cuda_root/'bin')+os.pathsep+os.environ['PATH'])
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('actual target CUDA GPU required to choose build architecture')
    major, minor = torch.cuda.get_device_capability()
    arch = str(major)+str(minor)
    run(['cmake', '-S', source, '-B', build, '-DCMAKE_BUILD_TYPE=Release',
        '-DCMAKE_NO_MPI=ON', '-DCUDA_ARCH='+arch, '-DCMAKE_CUDA_ARCHITECTURES='+arch,
        '-DCMAKE_CUDA_COMPILER='+str(args.cuda_root/'bin/nvcc'),
        '-DCUDAToolkit_ROOT='+str(args.cuda_root)], env=env)
    run(['cmake', '--build', build, '--target', 'amgxsh', '--parallel', args.jobs], env=env)
    library = next(build.rglob('libamgxsh.so'))
    run([shutil.which('cc') or 'gcc', '-shared', '-fPIC', '-O2',
        '-I'+str(source/'include'), project/'src/afsi_torch/ep/csrc/amgx_mode.c',
        '-o', library.with_name('libafsei_amgx_mode.so')])
    report = dict(revision=REVISION, cuda_arch=arch, library=str(library),
        torch_version=torch.__version__, cuda_version=torch.version.cuda,
        data_mode='dDDI', matrix_precision='float64', vector_precision='float64')
    (root/'installation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
