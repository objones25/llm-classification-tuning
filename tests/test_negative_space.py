import pytest

from llm_reward.negative_space import CheckFailed, bounded, check_finite, check_shape, require


def test_require_passes_silently_when_true():
    require(1 < 2)


def test_require_raises_check_failed_with_message():
    with pytest.raises(CheckFailed, match="ordering broken"):
        require(2 < 1, "ordering broken")


def test_bounded_yields_items_under_the_limit():
    assert list(bounded(range(3), limit=5)) == [0, 1, 2]


def test_bounded_raises_once_the_limit_is_exceeded():
    with pytest.raises(CheckFailed, match="exceeded its bound of 3"):
        list(bounded(range(10), limit=3, name="retries"))


def test_check_shape_returns_named_dim_bindings():
    class _Fake:
        shape = (2, 3)

    assert check_shape(_Fake(), (2, "n"), name="x") == {"n": 3}


def test_check_finite_raises_on_nan():
    with pytest.raises(CheckFailed, match="not finite"):
        check_finite(float("nan"), name="loss")
