"""Fetch the pinned public TorchCor PDE source without installing dependencies."""
import hashlib
from pathlib import Path
import sys
import tarfile
import urllib.request
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from validation.torchcor_pde import COMMIT,SOURCE_SHA256


def main():
    project=Path(__file__).resolve().parents[1]
    parent=project/'.deps';parent.mkdir(exist_ok=True)
    root=parent/f'torchcor-{COMMIT}'
    def valid():
        return all((root/name).is_file() and hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
                   for name,digest in SOURCE_SHA256.items())
    if valid():
        print(f'Pinned TorchCor PDE source already present: {root}');return
    if root.exists():raise ValueError('existing reference differs; inspect it before replacing')
    request=urllib.request.Request(f'https://codeload.github.com/sagebei/torchcor/tar.gz/{COMMIT}',
                                   headers={'User-Agent':'AFSEI-PDE-comparison'})
    with urllib.request.urlopen(request,timeout=30) as response:
        archive=parent/f'torchcor-{COMMIT}.tar.gz';archive.write_bytes(response.read())
    with tarfile.open(archive) as source:source.extractall(parent,filter='data')
    if not valid():raise ValueError('pinned source integrity check failed')
    print(f'Pinned TorchCor PDE source ready: {root}')


if __name__=='__main__':main()
