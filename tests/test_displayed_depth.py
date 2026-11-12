import pytest
from ml4t.specs import OrderSide

from ml4t.backtest.execution.depth import DepthConsumption, DisplayedDepthBook

BUY, SELL = OrderSide.BUY, OrderSide.SELL


@pytest.mark.parametrize(
    ("policy", "after_first", "after_update"),
    [
        (DepthConsumption.REUSE, ((101, 8),), ((101, 8),)),
        (DepthConsumption.REFRESH_LEVEL, ((101, 2),), ((101, 8),)),
        (DepthConsumption.CARRY_CONSUMPTION, ((101, 2),), ((101, 3),)),
    ],
)
def test_ioc_consumption_controls_price_limited_refills(policy, after_first, after_update):
    book = DisplayedDepthBook(policy)
    book.update(4, [(SELL, 101, 10), (SELL, 102, 5), (BUY, 100, 8)], snapshot=True)
    assert book.best(SELL) == 101
    assert book.sweep(BUY, 8, limit=101) == ((101, 8),)
    assert book.sweep(BUY, 8, limit=101) == after_first
    book.update(5, [(SELL, 101, 13)])
    assert book.sweep(BUY, 8, limit=101) == after_update
    assert book.sweep(SELL, 10, limit=99) == ((100, 8),)


def test_gap_and_invalid_batch_leave_depth_and_sequence_intact():
    book = DisplayedDepthBook()
    book.update(4, [(SELL, 101, 5)], snapshot=True)
    with pytest.raises(ValueError, match="contiguous"):
        book.update(6, [(SELL, 101, 0)])
    with pytest.raises(ValueError, match="nonnegative"):
        book.update(5, [(SELL, 101, 0), (SELL, 102, -1)])
    assert book.sweep(BUY, 1) == ((101, 1),)
    book.update(5, [(SELL, 101, 0)])
    assert book.sweep(BUY, 10) == ()
    book.update(100, [(SELL, 101, 7)], snapshot=True)
    assert book.sweep(BUY, 10) == ((101, 7),)


def test_size_at_match_does_not_reset_when_unmatched_level_changes_back():
    book = DisplayedDepthBook(DepthConsumption.REFRESH_SIZE_AT_MATCH)
    book.update(1, [(SELL, 101, 10)], snapshot=True)
    assert book.sweep(BUY, 8) == ((101, 8),)
    book.update(2, [(SELL, 101, 12)])
    book.update(3, [(SELL, 101, 10)])
    assert book.sweep(BUY, 8) == ((101, 2),)
    book.update(4, [(SELL, 101, 12)])
    assert book.sweep(BUY, 8) == ((101, 8),)
