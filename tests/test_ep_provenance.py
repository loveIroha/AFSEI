from pathlib import Path
import hashlib
import json
from afsi_torch.ep.initial import GPB_INITIAL, RESTING_VOLTAGE_V


def test_pinned_sources_and_explicit_unused_initial_state():
    folder = Path(__file__).resolve().parents[1]/'validation/ep_reference'
    record = json.loads((folder/'provenance.json').read_text())
    assert record['commit'] == '5bdde73c6be10cefadb005a0fcec5fdb6521f336'
    for name,digest in record['sha256'].items():
        assert hashlib.sha256((folder/name).read_bytes()).hexdigest() == digest
    assert record['source_initial_entries'] == 57
    assert len(GPB_INITIAL) == 57
    assert GPB_INITIAL[56] == 0.0  # missing full[57], unused and now deterministic
    assert RESTING_VOLTAGE_V == -0.0815455936324844
