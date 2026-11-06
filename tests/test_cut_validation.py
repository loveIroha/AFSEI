import pytest
from afsi_torch.ep.cut_validation import EventWindow


def test_event_window_counts_only_accepted_events_and_limits_exact_endpoint():
    w=EventWindow.from_config(dict(name='PHYSICAL_AV_OPENING',advance_ms=2.),5e-5)
    assert w.observe(10000,[],14000)==14000
    assert not w.completed(10000)
    assert w.observe(11000,['PHYSICAL_AV_OPENING'],14000)==11040
    assert w.observe(11002,['PHYSICAL_AV_OPENING'],14000)==11040
    assert w.first_tick==11000
    assert not w.completed(11039)
    assert w.completed(11040)


def test_event_window_cannot_report_success_when_duration_limit_truncates_window():
    w=EventWindow.from_config(dict(name='PHYSICAL_AV_OPENING',advance_ms=2.),5e-5)
    assert w.observe(11000,['PHYSICAL_AV_OPENING'],11010)==11010
    assert not w.completed(11010)


@pytest.mark.parametrize('spec',[
    dict(name='fake',advance_ms=2.),dict(name='PHYSICAL_AV_OPENING',advance_ms=-1.),
    dict(name='PHYSICAL_AV_OPENING',advance_ms=.012),dict(name='PHYSICAL_AV_OPENING',advance_ms=True)])
def test_event_window_rejects_invalid_or_off_grid_requests(spec):
    with pytest.raises(ValueError):EventWindow.from_config(spec,5e-5)
