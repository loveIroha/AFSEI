"""Retired added-solid LV experiment; retained name rejects stale commands.

Current scope permits only the original LV solid. Use the reusable circulation
controller in ep.lv_circulation after validating the internal basal fluid boundary.
"""


def run(c, output, *, device='cuda:0'):
    raise RuntimeError(
        'Retired LV housing experiment: the solid must contain only the original LV. '
        'The internal basal fluid boundary is not validated. '
        'No output or simulation was started. See configs/lv_source_flow_20261102/STATUS_zh.md.'
    )