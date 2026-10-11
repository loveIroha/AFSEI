"""Validate a discarded physical-pressure preload handoff on the original MAC."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from afsi_torch.ep import checkpoint
from afsi_torch.ep.mac_pressure_handoff import validate_static_handoff


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--steps', type=int, default=1)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if not torch.cuda.is_available() or not 1 <= a.steps <= 20:
        p.error('actual CUDA and 1..20 discarded static trials required')
    path = Path(a.checkpoint)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    driver, state, progress = checkpoint.load(path)
    preload = progress.get('preload', {})
    if (preload.get('preload_performed') is not True or preload.get('loading_history_steps') != 100
            or preload.get('strain_history_advanced_during_preload') is not True):
        p.error('a verified source 100 x 5ms MAC preload checkpoint is required')
    a.output.mkdir(parents=True, exist_ok=False)
    r = validate_static_handoff(driver, state, steps=a.steps)
    if sha != hashlib.sha256(path.read_bytes()).hexdigest():
        raise AssertionError('source checkpoint changed')
    r.update(checkpoint=str(path), source_sha256=sha, source_checkpoint_unchanged=True,
             gpu=torch.cuda.get_device_name(), grid_shape=driver.config.fsi.fluid.shape)
    (a.output/'report.json').write_text(json.dumps(r, indent=2, allow_nan=False)+'\n')
    print(json.dumps(r, indent=2), flush=True)


if __name__ == '__main__':
    main()
