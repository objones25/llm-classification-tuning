"""Runtime checks that survive `python -O`. Standard library only.

Use for programmer errors (a state your own code should have made impossible).
For operating errors — bad user input, missing file, network failure — raise a
typed exception instead and handle it.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Sequence
from typing import Any, NoReturn, TypeVar

__all__ = ["CheckFailed", "require", "unreachable", "bounded", "check_shape", "check_finite"]

T = TypeVar("T")


class CheckFailed(AssertionError):
    """A contract was violated. Subclasses AssertionError so pytest.raises(AssertionError) works."""


def require(condition: object, message: str = "") -> None:
    if not condition:
        raise CheckFailed(message or "requirement failed")


def unreachable(message: str = "") -> NoReturn:
    raise CheckFailed(message or "reached unreachable code")


def bounded(iterable: Iterable[T], limit: int, name: str = "loop") -> Iterator[T]:
    require(limit >= 1, f"{name}: bound must be at least 1, got {limit}")
    count = 0
    for item in iterable:
        count += 1
        if count > limit:
            raise CheckFailed(f"{name} exceeded its bound of {limit} iterations")
        yield item


def check_shape(array: Any, spec: Sequence[object], name: str = "array") -> dict[str, int]:
    shape = getattr(array, "shape", None)
    require(shape is not None, f"{name} has no .shape attribute")
    shape = tuple(int(d) for d in shape)
    require(
        len(shape) == len(spec),
        f"{name}: expected {len(spec)} dims {tuple(spec)}, got {len(shape)} {shape}",
    )
    bindings: dict[str, int] = {}
    for axis, (actual, expected) in enumerate(zip(shape, spec)):
        if expected is None or expected == -1:
            continue
        if isinstance(expected, str):
            if expected in bindings:
                require(
                    bindings[expected] == actual,
                    f"{name}: dim '{expected}' is {bindings[expected]} elsewhere but {actual} "
                    f"at axis {axis}; full shape {shape}",
                )
            else:
                bindings[expected] = actual
        else:
            require(
                actual == int(expected),
                f"{name}: axis {axis} expected {expected}, got {actual}; full shape {shape}",
            )
    return bindings


def check_finite(value: Any, name: str = "value") -> None:
    isfinite = getattr(value, "isfinite", None)
    if callable(isfinite):
        result = isfinite()
        allf = getattr(result, "all", None)
        ok = bool(allf()) if callable(allf) else bool(result)
        require(ok, f"{name} is not finite: {value}")
        return
    require(math.isfinite(float(value)), f"{name} is not finite: {value}")
