import os

import pytest
import vectorbtpro as vbt
from vectorbtpro.portfolio import nb
from vectorbtpro.portfolio.call_seq import build_call_seq, build_call_seq_nb
from vectorbtpro.portfolio.enums import *
from vectorbtpro.utils.random_ import set_seed

from tests.utils import *

seed = 42


# ############# Global ############# #


def setup_module():
    if os.environ.get("VBT_DISABLE_CACHING", "0") == "1":
        vbt.settings.caching["disable_machinery"] = True
    vbt.settings.pbar["disable"] = True
    vbt.settings.numba["check_func_suffix"] = True
    vbt.settings.portfolio["attach_call_seq"] = True


def teardown_module():
    vbt.settings.reset()


# ############# core ############# #


def assert_same_tuple(tup1, tup2):
    for i in range(len(tup1)):
        assert tup1[i] == tup2[i] or np.isnan(tup1[i]) and np.isnan(tup2[i])


def test_invalid_state():
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ExecState(
                cash=np.nan,
                position=100.0,
                debt=0.0,
                locked_cash=0.0,
                free_cash=100.0,
                val_price=10.0,
                value=1100.0,
            ),
            nb.order_nb(10, 10),
        )
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ExecState(
                cash=100.0,
                position=np.inf,
                debt=0.0,
                locked_cash=0.0,
                free_cash=100.0,
                val_price=10.0,
                value=1100.0,
            ),
            nb.order_nb(10, 10),
        )
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ExecState(
                cash=100.0,
                position=np.nan,
                debt=0.0,
                locked_cash=0.0,
                free_cash=100.0,
                val_price=10.0,
                value=1100.0,
            ),
            nb.order_nb(10, 10),
        )
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ExecState(
                cash=100.0,
                position=100.0,
                debt=np.nan,
                locked_cash=0.0,
                free_cash=100.0,
                val_price=10.0,
                value=1100.0,
            ),
            nb.order_nb(10, 10),
        )
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ExecState(
                cash=100.0,
                position=100.0,
                debt=-10.0,
                locked_cash=0.0,
                free_cash=100.0,
                val_price=10.0,
                value=1100.0,
            ),
            nb.order_nb(10, 10),
        )
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ExecState(
                cash=100.0,
                position=100.0,
                debt=0.0,
                locked_cash=np.nan,
                free_cash=100.0,
                val_price=10.0,
                value=1100.0,
            ),
            nb.order_nb(10, 10),
        )
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ExecState(
                cash=100.0,
                position=100.0,
                debt=0.0,
                locked_cash=-10.0,
                free_cash=100.0,
                val_price=10.0,
                value=1100.0,
            ),
            nb.order_nb(10, 10),
        )
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ExecState(
                cash=100.0,
                position=100.0,
                debt=0.0,
                locked_cash=0.0,
                free_cash=np.nan,
                val_price=10.0,
                value=1100.0,
            ),
            nb.order_nb(10, 10),
        )


def test_invalid_order():
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ExecState(
                cash=100.0,
                position=100.0,
                debt=0.0,
                locked_cash=0.0,
                free_cash=100.0,
                val_price=10.0,
                value=1100.0,
            ),
            nb.order_nb(10, -1),
        )
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ExecState(
                cash=100.0,
                position=100.0,
                debt=0.0,
                locked_cash=0.0,
                free_cash=100.0,
                val_price=10.0,
                value=1100.0,
            ),
            nb.order_nb(10, 10, size_type=-2),
        )
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ExecState(
                cash=100.0,
                position=100.0,
                debt=0.0,
                locked_cash=0.0,
                free_cash=100.0,
                val_price=10.0,
                value=1100.0,
            ),
            nb.order_nb(10, 10, size_type=20),
        )
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ExecState(
                cash=100.0,
                position=100.0,
                debt=0.0,
                locked_cash=0.0,
                free_cash=100.0,
                val_price=10.0,
                value=1100.0,
            ),
            nb.order_nb(10, 10, direction=-2),
        )
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ExecState(
                cash=100.0,
                position=100.0,
                debt=0.0,
                locked_cash=0.0,
                free_cash=100.0,
                val_price=10.0,
                value=1100.0,
            ),
            nb.order_nb(10, 10, direction=20),
        )
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ExecState(
                cash=100.0,
                position=100.0,
                debt=0.0,
                locked_cash=0.0,
                free_cash=100.0,
                val_price=10.0,
                value=1100.0,
            ),
            nb.order_nb(10, 10, fees=np.inf),
        )
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ExecState(
                cash=100.0,
                position=100.0,
                debt=0.0,
                locked_cash=0.0,
                free_cash=100.0,
                val_price=10.0,
                value=1100.0,
            ),
            nb.order_nb(10, 10, fees=np.nan),
        )
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ExecState(
                cash=100.0,
                position=100.0,
                debt=0.0,
                locked_cash=0.0,
                free_cash=100.0,
                val_price=10.0,
                value=1100.0,
            ),
            nb.order_nb(10, 10, fixed_fees=np.inf),
        )
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ExecState(
                cash=100.0,
                position=100.0,
                debt=0.0,
                locked_cash=0.0,
                free_cash=100.0,
                val_price=10.0,
                value=1100.0,
            ),
            nb.order_nb(10, 10, fixed_fees=np.nan),
        )
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ExecState(
                cash=100.0,
                position=100.0,
                debt=0.0,
                locked_cash=0.0,
                free_cash=100.0,
                val_price=10.0,
                value=1100.0,
            ),
            nb.order_nb(10, 10, slippage=np.inf),
        )
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ExecState(
                cash=100.0,
                position=100.0,
                debt=0.0,
                locked_cash=0.0,
                free_cash=100.0,
                val_price=10.0,
                value=1100.0,
            ),
            nb.order_nb(10, 10, slippage=-1),
        )
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ExecState(
                cash=100.0,
                position=100.0,
                debt=0.0,
                locked_cash=0.0,
                free_cash=100.0,
                val_price=10.0,
                value=1100.0,
            ),
            nb.order_nb(10, 10, min_size=np.inf),
        )
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ExecState(
                cash=100.0,
                position=100.0,
                debt=0.0,
                locked_cash=0.0,
                free_cash=100.0,
                val_price=10.0,
                value=1100.0,
            ),
            nb.order_nb(10, 10, min_size=-1),
        )
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ExecState(
                cash=100.0,
                position=100.0,
                debt=0.0,
                locked_cash=0.0,
                free_cash=100.0,
                val_price=10.0,
                value=1100.0,
            ),
            nb.order_nb(10, 10, max_size=0),
        )
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ExecState(
                cash=100.0,
                position=100.0,
                debt=0.0,
                locked_cash=0.0,
                free_cash=100.0,
                val_price=10.0,
                value=1100.0,
            ),
            nb.order_nb(10, 10, max_size=-10),
        )
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ExecState(
                cash=100.0,
                position=100.0,
                debt=0.0,
                locked_cash=0.0,
                free_cash=100.0,
                val_price=10.0,
                value=1100.0,
            ),
            nb.order_nb(10, 10, size_granularity=np.inf),
        )
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ExecState(
                cash=100.0,
                position=100.0,
                debt=0.0,
                locked_cash=0.0,
                free_cash=100.0,
                val_price=10.0,
                value=1100.0,
            ),
            nb.order_nb(10, 10, size_granularity=-10),
        )
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ExecState(
                cash=100.0,
                position=100.0,
                debt=0.0,
                locked_cash=0.0,
                free_cash=100.0,
                val_price=10.0,
                value=1100.0,
            ),
            nb.order_nb(10, 10, leverage=np.nan),
        )
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ExecState(
                cash=100.0,
                position=100.0,
                debt=0.0,
                locked_cash=0.0,
                free_cash=100.0,
                val_price=10.0,
                value=1100.0,
            ),
            nb.order_nb(10, 10, leverage=0),
        )
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ExecState(
                cash=100.0,
                position=100.0,
                debt=0.0,
                locked_cash=0.0,
                free_cash=100.0,
                val_price=10.0,
                value=1100.0,
            ),
            nb.order_nb(10, 10, leverage=-10),
        )
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ExecState(
                cash=100.0,
                position=100.0,
                debt=0.0,
                locked_cash=0.0,
                free_cash=100.0,
                val_price=10.0,
                value=1100.0,
            ),
            nb.order_nb(10, 10, leverage_mode=-2),
        )
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ExecState(
                cash=100.0,
                position=100.0,
                debt=0.0,
                locked_cash=0.0,
                free_cash=100.0,
                val_price=10.0,
                value=1100.0,
            ),
            nb.order_nb(10, 10, leverage_mode=20),
        )
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ExecState(
                cash=100.0,
                position=100.0,
                debt=0.0,
                locked_cash=0.0,
                free_cash=100.0,
                val_price=10.0,
                value=1100.0,
            ),
            nb.order_nb(10, 10, reject_prob=np.nan),
        )
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ExecState(
                cash=100.0,
                position=100.0,
                debt=0.0,
                locked_cash=0.0,
                free_cash=100.0,
                val_price=10.0,
                value=1100.0,
            ),
            nb.order_nb(10, 10, reject_prob=-1),
        )
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ExecState(
                cash=100.0,
                position=100.0,
                debt=0.0,
                locked_cash=0.0,
                free_cash=100.0,
                val_price=10.0,
                value=1100.0,
            ),
            nb.order_nb(10, 10, reject_prob=2),
        )
    order_result, account_state = nb.execute_order_nb(
        ExecState(
            cash=100.0,
            position=100.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=100.0,
            val_price=10.0,
            value=np.nan,
        ),
        nb.order_nb(1, 10, size_type=SizeType.TargetPercent),
    )
    np.testing.assert_array_equal(
        np.asarray(account_state),
        np.asarray(
            ExecState(
                cash=100.0,
                position=100.0,
                debt=0.0,
                locked_cash=0.0,
                free_cash=100.0,
                val_price=10.0,
                value=np.nan,
            )
        ),
    )
    assert_same_tuple(
        order_result,
        OrderResult(
            size=np.nan,
            price=np.nan,
            fees=np.nan,
            side=-1,
            status=1,
            status_info=3,
        ),
    )
    order_result, account_state = nb.execute_order_nb(
        ExecState(
            cash=100.0,
            position=100.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=100.0,
            val_price=10.0,
            value=-10.0,
        ),
        nb.order_nb(1, 10, size_type=SizeType.TargetPercent),
    )
    assert account_state == ExecState(
        cash=100.0,
        position=100.0,
        debt=0.0,
        locked_cash=0.0,
        free_cash=100.0,
        val_price=10.0,
        value=-10.0,
    )
    assert_same_tuple(
        order_result,
        OrderResult(
            size=np.nan,
            price=np.nan,
            fees=np.nan,
            side=-1,
            status=2,
            status_info=4,
        ),
    )
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ExecState(
                cash=100.0,
                position=100.0,
                debt=0.0,
                locked_cash=0.0,
                free_cash=100.0,
                val_price=np.inf,
                value=1100.0,
            ),
            nb.order_nb(10, 10, size_type=SizeType.Value),
        )
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ExecState(
                cash=100.0,
                position=100.0,
                debt=0.0,
                locked_cash=0.0,
                free_cash=100.0,
                val_price=-10.0,
                value=1100,
            ),
            nb.order_nb(10, 10, size_type=SizeType.Value),
        )
    order_result, account_state = nb.execute_order_nb(
        ExecState(
            cash=100.0,
            position=100.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=100.0,
            val_price=np.nan,
            value=1100.0,
        ),
        nb.order_nb(10, 10, size_type=SizeType.Value),
    )
    np.testing.assert_array_equal(
        np.asarray(account_state),
        np.asarray(
            ExecState(
                cash=100.0,
                position=100.0,
                debt=0.0,
                locked_cash=0.0,
                free_cash=100.0,
                val_price=np.nan,
                value=1100.0,
            )
        ),
    )
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ExecState(
                cash=100.0,
                position=100.0,
                debt=0.0,
                locked_cash=0.0,
                free_cash=100.0,
                val_price=np.inf,
                value=1100.0,
            ),
            nb.order_nb(10, 10, size_type=SizeType.TargetValue),
        )
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ExecState(
                cash=100.0,
                position=100.0,
                debt=0.0,
                locked_cash=0.0,
                free_cash=100.0,
                val_price=-10.0,
                value=1100,
            ),
            nb.order_nb(10, 10, size_type=SizeType.TargetValue),
        )
    order_result, account_state = nb.execute_order_nb(
        ExecState(
            cash=100.0,
            position=100.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=100.0,
            val_price=np.nan,
            value=1100.0,
        ),
        nb.order_nb(10, 10, size_type=SizeType.TargetValue),
    )
    np.testing.assert_array_equal(
        np.asarray(account_state),
        np.asarray(
            ExecState(
                cash=100.0,
                position=100.0,
                debt=0.0,
                locked_cash=0.0,
                free_cash=100.0,
                val_price=np.nan,
                value=1100.0,
            )
        ),
    )
    assert_same_tuple(
        order_result,
        OrderResult(
            size=np.nan,
            price=np.nan,
            fees=np.nan,
            side=-1,
            status=1,
            status_info=2,
        ),
    )
    order_result, account_state = nb.execute_order_nb(
        ExecState(
            cash=100.0,
            position=-10.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=100.0,
            val_price=10.0,
            value=1100.0,
        ),
        nb.order_nb(np.inf, 10, direction=Direction.ShortOnly),
    )
    assert account_state == ExecState(
        cash=200.0,
        position=-20.0,
        debt=100.0,
        locked_cash=100.0,
        free_cash=0.0,
        val_price=10.0,
        value=1100.0,
    )
    assert_same_tuple(
        order_result,
        OrderResult(
            size=10.0,
            price=10.0,
            fees=0.0,
            side=1,
            status=0,
            status_info=-1,
        ),
    )
    order_result, account_state = nb.execute_order_nb(
        ExecState(
            cash=100.0,
            position=-10.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=100.0,
            val_price=10.0,
            value=1100.0,
        ),
        nb.order_nb(-np.inf, 10, direction=Direction.Both),
    )
    assert account_state == ExecState(
        cash=200.0,
        position=-20.0,
        debt=100.0,
        locked_cash=100.0,
        free_cash=0.0,
        val_price=10.0,
        value=1100.0,
    )
    assert_same_tuple(
        order_result,
        OrderResult(
            size=10.0,
            price=10.0,
            fees=0.0,
            side=1,
            status=0,
            status_info=-1,
        ),
    )
    order_result, account_state = nb.execute_order_nb(
        ExecState(
            cash=100.0,
            position=10.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=100.0,
            val_price=10.0,
            value=1100.0,
        ),
        nb.order_nb(0, 10),
    )
    assert account_state == ExecState(
        cash=100.0,
        position=10.0,
        debt=0.0,
        locked_cash=0.0,
        free_cash=100.0,
        val_price=10.0,
        value=1100.0,
    )
    assert_same_tuple(
        order_result,
        OrderResult(
            size=np.nan,
            price=np.nan,
            fees=np.nan,
            side=-1,
            status=1,
            status_info=5,
        ),
    )
    order_result, account_state = nb.execute_order_nb(
        ExecState(
            cash=100.0,
            position=100.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=100.0,
            val_price=10.0,
            value=1100.0,
        ),
        nb.order_nb(15, 10, max_size=10, allow_partial=False),
    )
    assert account_state == ExecState(
        cash=100.0,
        position=100.0,
        debt=0.0,
        locked_cash=0.0,
        free_cash=100.0,
        val_price=10.0,
        value=1100.0,
    )
    assert_same_tuple(
        order_result,
        OrderResult(
            size=np.nan,
            price=np.nan,
            fees=np.nan,
            side=-1,
            status=2,
            status_info=8,
        ),
    )
    order_result, account_state = nb.execute_order_nb(
        ExecState(
            cash=100.0,
            position=100.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=100.0,
            val_price=10.0,
            value=1100.0,
        ),
        nb.order_nb(10, 10, reject_prob=1.0),
    )
    assert account_state == ExecState(
        cash=100.0,
        position=100.0,
        debt=0.0,
        locked_cash=0.0,
        free_cash=100.0,
        val_price=10.0,
        value=1100.0,
    )
    assert_same_tuple(
        order_result,
        OrderResult(
            size=np.nan,
            price=np.nan,
            fees=np.nan,
            side=-1,
            status=2,
            status_info=9,
        ),
    )
    order_result, account_state = nb.execute_order_nb(
        ExecState(
            cash=0.0,
            position=100.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=0.0,
            val_price=10.0,
            value=1100.0,
        ),
        nb.order_nb(10, 10, direction=Direction.LongOnly),
    )
    assert account_state == ExecState(
        cash=0.0,
        position=100.0,
        debt=0.0,
        locked_cash=0.0,
        free_cash=0.0,
        val_price=10.0,
        value=1100.0,
    )
    assert_same_tuple(
        order_result,
        OrderResult(
            size=np.nan,
            price=np.nan,
            fees=np.nan,
            side=-1,
            status=2,
            status_info=6,
        ),
    )
    order_result, account_state = nb.execute_order_nb(
        ExecState(
            cash=0.0,
            position=100.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=0.0,
            val_price=10.0,
            value=1100.0,
        ),
        nb.order_nb(10, 10, direction=Direction.Both),
    )
    assert account_state == ExecState(
        cash=0.0,
        position=100.0,
        debt=0.0,
        locked_cash=0.0,
        free_cash=0.0,
        val_price=10.0,
        value=1100.0,
    )
    assert_same_tuple(
        order_result,
        OrderResult(
            size=np.nan,
            price=np.nan,
            fees=np.nan,
            side=-1,
            status=2,
            status_info=6,
        ),
    )
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ExecState(
                cash=np.inf,
                position=100,
                debt=0.0,
                locked_cash=0.0,
                free_cash=np.inf,
                val_price=np.nan,
                value=1100.0,
            ),
            nb.order_nb(np.inf, 10, direction=Direction.LongOnly),
        )
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ExecState(
                cash=np.inf,
                position=100.0,
                debt=0.0,
                locked_cash=0.0,
                free_cash=np.inf,
                val_price=10.0,
                value=1100.0,
            ),
            nb.order_nb(np.inf, 10, direction=Direction.Both),
        )
    order_result, account_state = nb.execute_order_nb(
        ExecState(
            cash=100.0,
            position=0.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=100.0,
            val_price=10.0,
            value=1100.0,
        ),
        nb.order_nb(-10, 10, direction=Direction.ShortOnly),
    )
    assert account_state == ExecState(
        cash=100.0,
        position=0.0,
        debt=0.0,
        locked_cash=0.0,
        free_cash=100.0,
        val_price=10.0,
        value=1100.0,
    )
    assert_same_tuple(
        order_result,
        OrderResult(
            size=np.nan,
            price=np.nan,
            fees=np.nan,
            side=-1,
            status=2,
            status_info=7,
        ),
    )
    order_result, account_state = nb.execute_order_nb(
        ExecState(
            cash=np.inf,
            position=-100.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=np.inf,
            val_price=10.0,
            value=1100.0,
        ),
        nb.order_nb(-np.inf, 10, direction=Direction.ShortOnly),
    )
    assert account_state == ExecState(
        cash=np.inf,
        position=0.0,
        debt=0.0,
        locked_cash=0.0,
        free_cash=np.inf,
        val_price=10.0,
        value=1100.0,
    )
    assert_same_tuple(
        order_result,
        OrderResult(
            size=100.0,
            price=10.0,
            fees=0.0,
            side=0,
            status=0,
            status_info=-1,
        ),
    )
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ExecState(
                cash=np.inf,
                position=100.0,
                debt=0.0,
                locked_cash=0.0,
                free_cash=np.inf,
                val_price=10.0,
                value=1100.0,
            ),
            nb.order_nb(-np.inf, 10, direction=Direction.Both),
        )
    order_result, account_state = nb.execute_order_nb(
        ExecState(
            cash=100.0,
            position=0.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=100.0,
            val_price=10.0,
            value=1100.0,
        ),
        nb.order_nb(-10, 10, direction=Direction.LongOnly),
    )
    assert account_state == ExecState(
        cash=100.0,
        position=0.0,
        debt=0.0,
        locked_cash=0.0,
        free_cash=100.0,
        val_price=10.0,
        value=1100.0,
    )
    assert_same_tuple(
        order_result,
        OrderResult(
            size=np.nan,
            price=np.nan,
            fees=np.nan,
            side=-1,
            status=2,
            status_info=7,
        ),
    )
    order_result, account_state = nb.execute_order_nb(
        ExecState(
            cash=100.0,
            position=100.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=100.0,
            val_price=10.0,
            value=1100.0,
        ),
        nb.order_nb(10, 10, fixed_fees=100),
    )
    assert account_state == ExecState(
        cash=100.0,
        position=100.0,
        debt=0.0,
        locked_cash=0.0,
        free_cash=100.0,
        val_price=10.0,
        value=1100.0,
    )
    assert_same_tuple(
        order_result,
        OrderResult(
            size=np.nan,
            price=np.nan,
            fees=np.nan,
            side=-1,
            status=2,
            status_info=10,
        ),
    )
    order_result, account_state = nb.execute_order_nb(
        ExecState(
            cash=100.0,
            position=100.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=100.0,
            val_price=10.0,
            value=1100.0,
        ),
        nb.order_nb(10, 10, min_size=100),
    )
    assert account_state == ExecState(
        cash=100.0,
        position=100.0,
        debt=0.0,
        locked_cash=0.0,
        free_cash=100.0,
        val_price=10.0,
        value=1100.0,
    )
    assert_same_tuple(
        order_result,
        OrderResult(
            size=np.nan,
            price=np.nan,
            fees=np.nan,
            side=-1,
            status=1,
            status_info=11,
        ),
    )
    order_result, account_state = nb.execute_order_nb(
        ExecState(
            cash=100.0,
            position=100.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=100.0,
            val_price=10.0,
            value=1100.0,
        ),
        nb.order_nb(100, 10, allow_partial=False),
    )
    assert account_state == ExecState(
        cash=100.0,
        position=100.0,
        debt=0.0,
        locked_cash=0.0,
        free_cash=100.0,
        val_price=10.0,
        value=1100.0,
    )
    assert_same_tuple(
        order_result,
        OrderResult(
            size=np.nan,
            price=np.nan,
            fees=np.nan,
            side=-1,
            status=2,
            status_info=12,
        ),
    )
    order_result, account_state = nb.execute_order_nb(
        ExecState(
            cash=100.0,
            position=100.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=100.0,
            val_price=10.0,
            value=1100.0,
        ),
        nb.order_nb(-10, 10, min_size=100),
    )
    assert account_state == ExecState(
        cash=100.0,
        position=100.0,
        debt=0.0,
        locked_cash=0.0,
        free_cash=100.0,
        val_price=10.0,
        value=1100.0,
    )
    assert_same_tuple(
        order_result,
        OrderResult(
            size=np.nan,
            price=np.nan,
            fees=np.nan,
            side=-1,
            status=1,
            status_info=11,
        ),
    )
    order_result, account_state = nb.execute_order_nb(
        ExecState(
            cash=100.0,
            position=100.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=100.0,
            val_price=10.0,
            value=1100.0,
        ),
        nb.order_nb(-200, 10, direction=Direction.LongOnly, allow_partial=False),
    )
    assert account_state == ExecState(
        cash=100.0,
        position=100.0,
        debt=0.0,
        locked_cash=0.0,
        free_cash=100.0,
        val_price=10.0,
        value=1100.0,
    )
    assert_same_tuple(
        order_result,
        OrderResult(
            size=np.nan,
            price=np.nan,
            fees=np.nan,
            side=-1,
            status=2,
            status_info=12,
        ),
    )
    order_result, account_state = nb.execute_order_nb(
        ExecState(
            cash=100.0,
            position=100.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=100.0,
            val_price=10.0,
            value=1100.0,
        ),
        nb.order_nb(-10, 10, fixed_fees=1000),
    )
    assert account_state == ExecState(
        cash=100.0,
        position=100.0,
        debt=0.0,
        locked_cash=0.0,
        free_cash=100.0,
        val_price=10.0,
        value=1100.0,
    )
    assert_same_tuple(
        order_result,
        OrderResult(
            size=np.nan,
            price=np.nan,
            fees=np.nan,
            side=-1,
            status=2,
            status_info=10,
        ),
    )


def test_calculations():
    order_result, account_state = nb.execute_order_nb(
        ExecState(
            cash=100.0,
            position=0.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=100.0,
            val_price=10.0,
            value=100.0,
        ),
        nb.order_nb(10, 10, fees=0.1, fixed_fees=1, slippage=0.1),
    )
    assert account_state == ExecState(
        cash=0.0,
        position=8.18181818181818,
        debt=0.0,
        locked_cash=0.0,
        free_cash=0.0,
        val_price=10.0,
        value=100.0,
    )
    assert_same_tuple(
        order_result,
        OrderResult(
            size=8.18181818181818,
            price=11.0,
            fees=10.000000000000014,
            side=0,
            status=0,
            status_info=-1,
        ),
    )
    order_result, account_state = nb.execute_order_nb(
        ExecState(
            cash=100.0,
            position=0.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=100.0,
            val_price=10.0,
            value=100.0,
        ),
        nb.order_nb(100, 10, fees=0.1, fixed_fees=1, slippage=0.1),
    )
    assert account_state == ExecState(
        cash=0.0,
        position=8.18181818181818,
        debt=0.0,
        locked_cash=0.0,
        free_cash=0.0,
        val_price=10.0,
        value=100.0,
    )
    assert_same_tuple(
        order_result,
        OrderResult(
            size=8.18181818181818,
            price=11.0,
            fees=10.000000000000014,
            side=0,
            status=0,
            status_info=-1,
        ),
    )
    order_result, account_state = nb.execute_order_nb(
        ExecState(
            cash=100.0,
            position=0.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=100.0,
            val_price=10.0,
            value=100.0,
        ),
        nb.order_nb(-10, 10, fees=0.1, fixed_fees=1, slippage=0.1),
    )
    assert account_state == ExecState(
        cash=180.0,
        position=-10.0,
        debt=90.0,
        locked_cash=90.0,
        free_cash=0.0,
        val_price=10.0,
        value=100.0,
    )
    assert_same_tuple(
        order_result,
        OrderResult(
            size=10.0,
            price=9.0,
            fees=10.0,
            side=1,
            status=0,
            status_info=-1,
        ),
    )
    order_result, account_state = nb.execute_order_nb(
        ExecState(
            cash=100.0,
            position=0.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=100.0,
            val_price=10.0,
            value=100.0,
        ),
        nb.order_nb(-100, 10, fees=0.1, fixed_fees=1, slippage=0.1),
    )
    assert account_state == ExecState(
        cash=180.0,
        position=-10.0,
        debt=90.0,
        locked_cash=90.0,
        free_cash=0.0,
        val_price=10.0,
        value=100.0,
    )
    assert_same_tuple(
        order_result,
        OrderResult(
            size=10.0,
            price=9.0,
            fees=10.0,
            side=1,
            status=0,
            status_info=-1,
        ),
    )
    order_result, account_state = nb.execute_order_nb(
        ExecState(
            cash=100.0,
            position=0.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=100.0,
            val_price=10.0,
            value=100.0,
        ),
        nb.order_nb(-100, 10, fees=0.1, fixed_fees=1, slippage=0.1, leverage=np.inf),
    )
    assert account_state == ExecState(
        cash=909.0,
        position=-100.0,
        debt=900.0,
        locked_cash=9.0,
        free_cash=0.0,
        val_price=10.0,
        value=100.0,
    )
    assert_same_tuple(
        order_result,
        OrderResult(
            size=100.0,
            price=9.0,
            fees=91.0,
            side=1,
            status=0,
            status_info=-1,
        ),
    )
    order_result, account_state = nb.execute_order_nb(
        ExecState(
            cash=100.0,
            position=0.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=100.0,
            val_price=10.0,
            value=100.0,
        ),
        nb.order_nb(10, 10, fees=-0.1, fixed_fees=-1, slippage=0.1),
    )
    assert account_state == ExecState(
        cash=2.0,
        position=10.0,
        debt=0.0,
        locked_cash=0.0,
        free_cash=2.0,
        val_price=10.0,
        value=100.0,
    )
    assert_same_tuple(
        order_result,
        OrderResult(
            size=10.0,
            price=11.0,
            fees=-12.0,
            side=0,
            status=0,
            status_info=-1,
        ),
    )
    order_result, account_state = nb.execute_order_nb(
        ExecState(
            cash=100.0,
            position=0.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=100.0,
            val_price=10.0,
            value=100.0,
        ),
        nb.order_nb(10, 0, fees=0.1, fixed_fees=1, slippage=0.1),
    )
    assert account_state == ExecState(
        cash=99.0,
        position=10.0,
        debt=0.0,
        locked_cash=0.0,
        free_cash=99.0,
        val_price=10.0,
        value=100.0,
    )
    assert_same_tuple(
        order_result,
        OrderResult(
            size=10.0,
            price=0.0,
            fees=1.0,
            side=0,
            status=0,
            status_info=-1,
        ),
    )
    order_result, account_state = nb.execute_order_nb(
        ExecState(
            cash=100.0,
            position=10.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=100.0,
            val_price=10.0,
            value=100.0,
        ),
        nb.order_nb(-10, 0, fees=0.1, fixed_fees=1, slippage=0.1),
    )
    assert account_state == ExecState(
        cash=99.0,
        position=0.0,
        debt=0.0,
        locked_cash=0.0,
        free_cash=99.0,
        val_price=10.0,
        value=100.0,
    )
    assert_same_tuple(
        order_result,
        OrderResult(
            size=10.0,
            price=0.0,
            fees=1.0,
            side=1,
            status=0,
            status_info=-1,
        ),
    )

    order_result, account_state = nb.execute_order_nb(
        ExecState(
            cash=100.0,
            position=0.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=100.0,
            val_price=10.0,
            value=100.0,
        ),
        nb.order_nb(10, 10, size_type=SizeType.TargetAmount),
    )
    assert account_state == ExecState(
        cash=0.0,
        position=10.0,
        debt=0.0,
        locked_cash=0.0,
        free_cash=0.0,
        val_price=10.0,
        value=100.0,
    )
    assert_same_tuple(
        order_result,
        OrderResult(
            size=10.0,
            price=10.0,
            fees=0.0,
            side=0,
            status=0,
            status_info=-1,
        ),
    )
    order_result, account_state = nb.execute_order_nb(
        ExecState(
            cash=100.0,
            position=0.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=100.0,
            val_price=10.0,
            value=100.0,
        ),
        nb.order_nb(-10, 10, size_type=SizeType.TargetAmount),
    )
    assert account_state == ExecState(
        cash=200.0,
        position=-10.0,
        debt=100.0,
        locked_cash=100.0,
        free_cash=0.0,
        val_price=10.0,
        value=100.0,
    )
    assert_same_tuple(
        order_result,
        OrderResult(
            size=10.0,
            price=10.0,
            fees=0.0,
            side=1,
            status=0,
            status_info=-1,
        ),
    )

    order_result, account_state = nb.execute_order_nb(
        ExecState(
            cash=100.0,
            position=0.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=100.0,
            val_price=10.0,
            value=100.0,
        ),
        nb.order_nb(100, 10, size_type=SizeType.Value),
    )
    assert account_state == ExecState(
        cash=0.0,
        position=10.0,
        debt=0.0,
        locked_cash=0.0,
        free_cash=0.0,
        val_price=10.0,
        value=100.0,
    )
    assert_same_tuple(
        order_result,
        OrderResult(
            size=10.0,
            price=10.0,
            fees=0.0,
            side=0,
            status=0,
            status_info=-1,
        ),
    )
    order_result, account_state = nb.execute_order_nb(
        ExecState(
            cash=100.0,
            position=0.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=100.0,
            val_price=10.0,
            value=100.0,
        ),
        nb.order_nb(-100, 10, size_type=SizeType.Value),
    )
    assert account_state == ExecState(
        cash=200.0,
        position=-10.0,
        debt=100.0,
        locked_cash=100.0,
        free_cash=0.0,
        val_price=10.0,
        value=100.0,
    )
    assert_same_tuple(
        order_result,
        OrderResult(
            size=10.0,
            price=10.0,
            fees=0.0,
            side=1,
            status=0,
            status_info=-1,
        ),
    )

    order_result, account_state = nb.execute_order_nb(
        ExecState(
            cash=100.0,
            position=0.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=100.0,
            val_price=10.0,
            value=100.0,
        ),
        nb.order_nb(100, 10, size_type=SizeType.TargetValue),
    )
    assert account_state == ExecState(
        cash=0.0,
        position=10.0,
        debt=0.0,
        locked_cash=0.0,
        free_cash=0.0,
        val_price=10.0,
        value=100.0,
    )
    assert_same_tuple(
        order_result,
        OrderResult(
            size=10.0,
            price=10.0,
            fees=0.0,
            side=0,
            status=0,
            status_info=-1,
        ),
    )
    order_result, account_state = nb.execute_order_nb(
        ExecState(
            cash=100.0,
            position=0.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=100.0,
            val_price=10.0,
            value=100.0,
        ),
        nb.order_nb(-100, 10, size_type=SizeType.TargetValue),
    )
    assert account_state == ExecState(
        cash=200.0,
        position=-10.0,
        debt=100.0,
        locked_cash=100.0,
        free_cash=0.0,
        val_price=10.0,
        value=100.0,
    )
    assert_same_tuple(
        order_result,
        OrderResult(
            size=10.0,
            price=10.0,
            fees=0.0,
            side=1,
            status=0,
            status_info=-1,
        ),
    )

    order_result, account_state = nb.execute_order_nb(
        ExecState(
            cash=100.0,
            position=0.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=100.0,
            val_price=10.0,
            value=100.0,
        ),
        nb.order_nb(1, 10, size_type=SizeType.TargetPercent),
    )
    assert account_state == ExecState(
        cash=0.0,
        position=10.0,
        debt=0.0,
        locked_cash=0.0,
        free_cash=0.0,
        val_price=10.0,
        value=100.0,
    )
    assert_same_tuple(
        order_result,
        OrderResult(
            size=10.0,
            price=10.0,
            fees=0.0,
            side=0,
            status=0,
            status_info=-1,
        ),
    )
    order_result, account_state = nb.execute_order_nb(
        ExecState(
            cash=100.0,
            position=0.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=100.0,
            val_price=10.0,
            value=100.0,
        ),
        nb.order_nb(-1, 10, size_type=SizeType.TargetPercent),
    )
    assert account_state == ExecState(
        cash=200.0,
        position=-10.0,
        debt=100.0,
        locked_cash=100.0,
        free_cash=0.0,
        val_price=10.0,
        value=100.0,
    )
    assert_same_tuple(
        order_result,
        OrderResult(
            size=10.0,
            price=10.0,
            fees=0.0,
            side=1,
            status=0,
            status_info=-1,
        ),
    )

    order_result, account_state = nb.execute_order_nb(
        ExecState(
            cash=50.0,
            position=5.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=50.0,
            val_price=10.0,
            value=100.0,
        ),
        nb.order_nb(1, 10, size_type=SizeType.Percent),
    )
    assert account_state == ExecState(
        cash=0.0,
        position=10.0,
        debt=0.0,
        locked_cash=0.0,
        free_cash=0.0,
        val_price=10.0,
        value=100.0,
    )
    assert_same_tuple(
        order_result,
        OrderResult(
            size=5.0,
            price=10.0,
            fees=0.0,
            side=0,
            status=0,
            status_info=-1,
        ),
    )
    order_result, account_state = nb.execute_order_nb(
        ExecState(
            cash=50.0,
            position=5.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=50.0,
            val_price=10.0,
            value=100.0,
        ),
        nb.order_nb(0.5, 10, size_type=SizeType.Percent),
    )
    assert account_state == ExecState(
        cash=25.0,
        position=7.5,
        debt=0.0,
        locked_cash=0.0,
        free_cash=25.0,
        val_price=10.0,
        value=100.0,
    )
    assert_same_tuple(
        order_result,
        OrderResult(
            size=2.5,
            price=10.0,
            fees=0.0,
            side=0,
            status=0,
            status_info=-1,
        ),
    )
    order_result, account_state = nb.execute_order_nb(
        ExecState(
            cash=50.0,
            position=5.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=50.0,
            val_price=10.0,
            value=100.0,
        ),
        nb.order_nb(-0.5, 10, size_type=SizeType.Percent),
    )
    assert account_state == ExecState(
        cash=150.0,
        position=-5.0,
        debt=50.0,
        locked_cash=50.0,
        free_cash=50.0,
        val_price=10.0,
        value=100.0,
    )
    assert_same_tuple(
        order_result,
        OrderResult(
            size=10.0,
            price=10.0,
            fees=0.0,
            side=1,
            status=0,
            status_info=-1,
        ),
    )
    order_result, account_state = nb.execute_order_nb(
        ExecState(
            cash=50.0,
            position=5.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=50.0,
            val_price=10.0,
            value=100.0,
        ),
        nb.order_nb(-1, 10, size_type=SizeType.Percent),
    )
    assert account_state == ExecState(
        cash=200.0,
        position=-10.0,
        debt=100.0,
        locked_cash=100.0,
        free_cash=0.0,
        val_price=10.0,
        value=100.0,
    )
    assert_same_tuple(
        order_result,
        OrderResult(
            size=15.0,
            price=10.0,
            fees=0.0,
            side=1,
            status=0,
            status_info=-1,
        ),
    )
    order_result, account_state = nb.execute_order_nb(
        ExecState(
            cash=50.0,
            position=0.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=50.0,
            val_price=10.0,
            value=100.0,
        ),
        nb.order_nb(1, 10, size_type=SizeType.Percent),
    )
    assert account_state == ExecState(
        cash=0.0,
        position=5.0,
        debt=0.0,
        locked_cash=0.0,
        free_cash=0.0,
        val_price=10.0,
        value=100.0,
    )
    assert_same_tuple(
        order_result,
        OrderResult(
            size=5.0,
            price=10.0,
            fees=0.0,
            side=0,
            status=0,
            status_info=-1,
        ),
    )
    order_result, account_state = nb.execute_order_nb(
        ExecState(
            cash=50.0,
            position=0.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=50.0,
            val_price=10.0,
            value=100.0,
        ),
        nb.order_nb(0.5, 10, size_type=SizeType.Percent),
    )
    assert account_state == ExecState(
        cash=25.0,
        position=2.5,
        debt=0.0,
        locked_cash=0.0,
        free_cash=25.0,
        val_price=10.0,
        value=100.0,
    )
    assert_same_tuple(
        order_result,
        OrderResult(
            size=2.5,
            price=10.0,
            fees=0.0,
            side=0,
            status=0,
            status_info=-1,
        ),
    )
    order_result, account_state = nb.execute_order_nb(
        ExecState(
            cash=50.0,
            position=0.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=50.0,
            val_price=10.0,
            value=100.0,
        ),
        nb.order_nb(-0.5, 10, size_type=SizeType.Percent),
    )
    assert account_state == ExecState(
        cash=75.0,
        position=-2.5,
        debt=25.0,
        locked_cash=25.0,
        free_cash=25.0,
        val_price=10.0,
        value=100.0,
    )
    assert_same_tuple(
        order_result,
        OrderResult(
            size=2.5,
            price=10.0,
            fees=0.0,
            side=1,
            status=0,
            status_info=-1,
        ),
    )
    order_result, account_state = nb.execute_order_nb(
        ExecState(
            cash=50.0,
            position=0.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=50.0,
            val_price=10.0,
            value=100.0,
        ),
        nb.order_nb(-1, 10, size_type=SizeType.Percent),
    )
    assert account_state == ExecState(
        cash=100.0,
        position=-5.0,
        debt=50.0,
        locked_cash=50.0,
        free_cash=0.0,
        val_price=10.0,
        value=100.0,
    )
    assert_same_tuple(
        order_result,
        OrderResult(
            size=5.0,
            price=10.0,
            fees=0.0,
            side=1,
            status=0,
            status_info=-1,
        ),
    )
    order_result, account_state = nb.execute_order_nb(
        ExecState(
            cash=50.0,
            position=-5.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=50.0,
            val_price=10.0,
            value=100.0,
        ),
        nb.order_nb(1, 10, size_type=SizeType.Percent),
    )
    assert account_state == ExecState(
        cash=0.0,
        position=0.0,
        debt=0.0,
        locked_cash=0.0,
        free_cash=0.0,
        val_price=10.0,
        value=100.0,
    )
    assert_same_tuple(
        order_result,
        OrderResult(
            size=5.0,
            price=10.0,
            fees=0.0,
            side=0,
            status=0,
            status_info=-1,
        ),
    )
    order_result, account_state = nb.execute_order_nb(
        ExecState(
            cash=50.0,
            position=-5.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=50.0,
            val_price=10.0,
            value=100.0,
        ),
        nb.order_nb(0.5, 10, size_type=SizeType.Percent),
    )
    assert account_state == ExecState(
        cash=0.0,
        position=0.0,
        debt=0.0,
        locked_cash=0.0,
        free_cash=0.0,
        val_price=10.0,
        value=100.0,
    )
    assert_same_tuple(
        order_result,
        OrderResult(
            size=5.0,
            price=10.0,
            fees=0.0,
            side=0,
            status=0,
            status_info=-1,
        ),
    )
    order_result, account_state = nb.execute_order_nb(
        ExecState(
            cash=50.0,
            position=-5.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=50.0,
            val_price=10.0,
            value=100.0,
        ),
        nb.order_nb(-0.5, 10, size_type=SizeType.Percent),
    )
    assert account_state == ExecState(
        cash=75.0,
        position=-7.5,
        debt=25.0,
        locked_cash=25.0,
        free_cash=25.0,
        val_price=10.0,
        value=100.0,
    )
    assert_same_tuple(
        order_result,
        OrderResult(
            size=2.5,
            price=10.0,
            fees=0.0,
            side=1,
            status=0,
            status_info=-1,
        ),
    )
    order_result, account_state = nb.execute_order_nb(
        ExecState(
            cash=50.0,
            position=-5.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=50.0,
            val_price=10.0,
            value=100.0,
        ),
        nb.order_nb(-1, 10, size_type=SizeType.Percent),
    )
    assert account_state == ExecState(
        cash=100.0,
        position=-10.0,
        debt=50.0,
        locked_cash=50.0,
        free_cash=0.0,
        val_price=10.0,
        value=100.0,
    )
    assert_same_tuple(
        order_result,
        OrderResult(
            size=5.0,
            price=10.0,
            fees=0.0,
            side=1,
            status=0,
            status_info=-1,
        ),
    )

    order_result, account_state = nb.execute_order_nb(
        ExecState(
            cash=100.0,
            position=0.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=100.0,
            val_price=10.0,
            value=100.0,
        ),
        nb.order_nb(np.inf, 10),
    )
    assert account_state == ExecState(
        cash=0.0,
        position=10.0,
        debt=0.0,
        locked_cash=0.0,
        free_cash=0.0,
        val_price=10.0,
        value=100.0,
    )
    assert_same_tuple(
        order_result,
        OrderResult(
            size=10.0,
            price=10.0,
            fees=0.0,
            side=0,
            status=0,
            status_info=-1,
        ),
    )
    order_result, account_state = nb.execute_order_nb(
        ExecState(
            cash=100.0,
            position=-5.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=100.0,
            val_price=10.0,
            value=100.0,
        ),
        nb.order_nb(np.inf, 10),
    )
    assert account_state == ExecState(
        cash=0.0,
        position=5.0,
        debt=0.0,
        locked_cash=0.0,
        free_cash=0.0,
        val_price=10.0,
        value=100.0,
    )
    assert_same_tuple(
        order_result,
        OrderResult(
            size=10.0,
            price=10.0,
            fees=0.0,
            side=0,
            status=0,
            status_info=-1,
        ),
    )
    order_result, account_state = nb.execute_order_nb(
        ExecState(
            cash=100.0,
            position=0.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=100.0,
            val_price=10.0,
            value=100.0,
        ),
        nb.order_nb(-np.inf, 10),
    )
    assert account_state == ExecState(
        cash=200.0,
        position=-10.0,
        debt=100.0,
        locked_cash=100.0,
        free_cash=0.0,
        val_price=10.0,
        value=100.0,
    )
    assert_same_tuple(
        order_result,
        OrderResult(
            size=10.0,
            price=10.0,
            fees=0.0,
            side=1,
            status=0,
            status_info=-1,
        ),
    )
    order_result, account_state = nb.execute_order_nb(
        ExecState(
            cash=150.0,
            position=-5.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=150.0,
            val_price=10.0,
            value=100.0,
        ),
        nb.order_nb(-np.inf, 10),
    )
    assert account_state == ExecState(
        cash=300.0,
        position=-20.0,
        debt=150.0,
        locked_cash=150.0,
        free_cash=0.0,
        val_price=10.0,
        value=100.0,
    )
    assert_same_tuple(
        order_result,
        OrderResult(
            size=15.0,
            price=10.0,
            fees=0.0,
            side=1,
            status=0,
            status_info=-1,
        ),
    )

    order_result, account_state = nb.execute_order_nb(
        ExecState(
            cash=100.0,
            position=0.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=50.0,
            val_price=10.0,
            value=100.0,
        ),
        nb.order_nb(10, 10),
    )
    assert account_state == ExecState(
        cash=50.0,
        position=5.0,
        debt=0.0,
        locked_cash=0.0,
        free_cash=0.0,
        val_price=10.0,
        value=100.0,
    )
    assert_same_tuple(
        order_result,
        OrderResult(
            size=5.0,
            price=10.0,
            fees=0.0,
            side=0,
            status=0,
            status_info=-1,
        ),
    )
    order_result, account_state = nb.execute_order_nb(
        ExecState(
            cash=1000.0,
            position=-5.0,
            debt=50.0,
            locked_cash=50.0,
            free_cash=50.0,
            val_price=10.0,
            value=100.0,
        ),
        nb.order_nb(10, 17.5),
    )
    assert account_state == ExecState(
        cash=850.0,
        position=3.5714285714285716,
        debt=0.0,
        locked_cash=0.0,
        free_cash=0.0,
        val_price=10.0,
        value=100.0,
    )
    assert_same_tuple(
        order_result,
        OrderResult(
            size=8.5714285714285716,
            price=17.5,
            fees=0.0,
            side=0,
            status=0,
            status_info=-1,
        ),
    )
    order_result, account_state = nb.execute_order_nb(
        ExecState(
            cash=150.0,
            position=-5.0,
            debt=50.0,
            locked_cash=50.0,
            free_cash=50.0,
            val_price=10.0,
            value=100.0,
        ),
        nb.order_nb(10, 100),
    )
    assert account_state == ExecState(
        cash=0.0,
        position=-3.5,
        debt=35.0,
        locked_cash=35.0,
        free_cash=-70.0,
        val_price=10.0,
        value=100.0,
    )
    assert_same_tuple(
        order_result,
        OrderResult(
            size=1.5,
            price=100.0,
            fees=0.0,
            side=0,
            status=0,
            status_info=-1,
        ),
    )

    order_result, account_state = nb.execute_order_nb(
        ExecState(
            cash=0.0,
            position=10.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=-50.0,
            val_price=10.0,
            value=100.0,
        ),
        nb.order_nb(-20, 10),
    )
    assert account_state == ExecState(
        cash=150.0,
        position=-5.0,
        debt=50.0,
        locked_cash=50.0,
        free_cash=0.0,
        val_price=10.0,
        value=100.0,
    )
    assert_same_tuple(
        order_result,
        OrderResult(
            size=15.0,
            price=10.0,
            fees=0.0,
            side=1,
            status=0,
            status_info=-1,
        ),
    )
    order_result, account_state = nb.execute_order_nb(
        ExecState(
            cash=0.0,
            position=1.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=-50.0,
            val_price=10.0,
            value=100.0,
        ),
        nb.order_nb(-10, 10),
    )
    assert account_state == ExecState(
        cash=10.0,
        position=0.0,
        debt=0.0,
        locked_cash=0.0,
        free_cash=-40.0,
        val_price=10.0,
        value=100.0,
    )
    assert_same_tuple(
        order_result,
        OrderResult(
            size=1.0,
            price=10.0,
            fees=0.0,
            side=1,
            status=0,
            status_info=-1,
        ),
    )
    order_result, account_state = nb.execute_order_nb(
        ExecState(
            cash=0.0,
            position=0.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=-100.0,
            val_price=10.0,
            value=100.0,
        ),
        nb.order_nb(-10, 10),
    )
    assert account_state == ExecState(
        cash=0.0,
        position=0.0,
        debt=0.0,
        locked_cash=0.0,
        free_cash=-100.0,
        val_price=10.0,
        value=100.0,
    )
    assert_same_tuple(
        order_result,
        OrderResult(
            size=np.nan,
            price=np.nan,
            fees=np.nan,
            side=-1,
            status=2,
            status_info=6,
        ),
    )
    order_result, account_state = nb.execute_order_nb(
        ExecState(
            cash=0.0,
            position=0.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=100.0,
            val_price=10.0,
            value=100.0,
        ),
        nb.order_nb(-20, 10, fees=0.1, slippage=0.1, fixed_fees=1.0),
    )
    assert account_state == ExecState(
        cash=80.0,
        position=-10.0,
        debt=90.0,
        locked_cash=90.0,
        free_cash=0.0,
        val_price=10.0,
        value=100.0,
    )
    assert_same_tuple(
        order_result,
        OrderResult(
            size=10.0,
            price=9.0,
            fees=10.0,
            side=1,
            status=0,
            status_info=-1,
        ),
    )


def test_approx_order_value_nb():
    assert (
        nb.approx_order_value_nb(
            ExecState(
                cash=100.0,
                position=0.0,
                debt=0.0,
                locked_cash=0.0,
                free_cash=100.0,
                val_price=15.0,
                value=100.0,
            ),
            np.inf,
            SizeType.Amount,
            Direction.Both,
        )
        == np.inf
    )
    assert (
        nb.approx_order_value_nb(
            ExecState(
                cash=100.0,
                position=0.0,
                debt=0.0,
                locked_cash=0.0,
                free_cash=100.0,
                val_price=15.0,
                value=100.0,
            ),
            -np.inf,
            SizeType.Amount,
            Direction.Both,
        )
        == np.inf
    )
    assert (
        nb.approx_order_value_nb(
            ExecState(
                cash=100.0,
                position=0.0,
                debt=0.0,
                locked_cash=0.0,
                free_cash=100.0,
                val_price=15.0,
                value=100.0,
            ),
            10,
            SizeType.Amount,
            Direction.Both,
        )
        == 150.0
    )
    assert (
        nb.approx_order_value_nb(
            ExecState(
                cash=100.0,
                position=0.0,
                debt=0.0,
                locked_cash=0.0,
                free_cash=100.0,
                val_price=15.0,
                value=100.0,
            ),
            -10,
            SizeType.Amount,
            Direction.Both,
        )
        == 150.0
    )
    assert (
        nb.approx_order_value_nb(
            ExecState(
                cash=0.0,
                position=10.0,
                debt=0.0,
                locked_cash=0.0,
                free_cash=0.0,
                val_price=15.0,
                value=100.0,
            ),
            -5,
            SizeType.Amount,
            Direction.LongOnly,
        )
        == -75.0
    )
    assert (
        nb.approx_order_value_nb(
            ExecState(
                cash=0.0,
                position=10.0,
                debt=0.0,
                locked_cash=0.0,
                free_cash=0.0,
                val_price=15.0,
                value=100.0,
            ),
            -np.inf,
            SizeType.Amount,
            Direction.LongOnly,
        )
        == -150.0
    )
    assert (
        nb.approx_order_value_nb(
            ExecState(
                cash=0.0,
                position=10.0,
                debt=0.0,
                locked_cash=0.0,
                free_cash=0.0,
                val_price=15.0,
                value=100.0,
            ),
            -15,
            SizeType.Amount,
            Direction.Both,
        )
        == -75.0
    )
    assert (
        nb.approx_order_value_nb(
            ExecState(
                cash=0.0,
                position=10.0,
                debt=0.0,
                locked_cash=0.0,
                free_cash=0.0,
                val_price=15.0,
                value=100.0,
            ),
            -np.inf,
            SizeType.Amount,
            Direction.Both,
        )
        == np.inf
    )
    assert (
        nb.approx_order_value_nb(
            ExecState(
                cash=200.0,
                position=-10.0,
                debt=100.0,
                locked_cash=100.0,
                free_cash=0.0,
                val_price=10.0,
                value=100.0,
            ),
            -5,
            SizeType.Amount,
            Direction.ShortOnly,
        )
        == -50.0
    )
    assert (
        nb.approx_order_value_nb(
            ExecState(
                cash=200.0,
                position=-10.0,
                debt=100.0,
                locked_cash=100.0,
                free_cash=0.0,
                val_price=10.0,
                value=100.0,
            ),
            -np.inf,
            SizeType.Amount,
            Direction.ShortOnly,
        )
        == -100.0
    )
    assert (
        nb.approx_order_value_nb(
            ExecState(
                cash=200.0,
                position=-10.0,
                debt=100.0,
                locked_cash=100.0,
                free_cash=0.0,
                val_price=10.0,
                value=100.0,
            ),
            15,
            SizeType.Amount,
            Direction.Both,
        )
        == -50.0
    )
    assert (
        nb.approx_order_value_nb(
            ExecState(
                cash=200.0,
                position=-10.0,
                debt=100.0,
                locked_cash=100.0,
                free_cash=0.0,
                val_price=10.0,
                value=100.0,
            ),
            np.inf,
            SizeType.Amount,
            Direction.Both,
        )
        == np.inf
    )


def test_build_call_seq_nb():
    group_lens = np.array([1, 2, 3, 4])
    np.testing.assert_array_equal(
        build_call_seq_nb((10, 10), group_lens, CallSeqType.Default),
        build_call_seq((10, 10), group_lens, CallSeqType.Default),
    )
    np.testing.assert_array_equal(
        build_call_seq_nb((10, 10), group_lens, CallSeqType.Reversed),
        build_call_seq((10, 10), group_lens, CallSeqType.Reversed),
    )
    set_seed(seed)
    out1 = build_call_seq_nb((10, 10), group_lens, CallSeqType.Random)
    set_seed(seed)
    out2 = build_call_seq((10, 10), group_lens, CallSeqType.Random)
    np.testing.assert_array_equal(out1, out2)
