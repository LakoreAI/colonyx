"""Property-based tests for the bounds-validation invariant.

Bounds validation: every algorithm validates input
bounds correctly" as a key testing standard, but until now that was only
exercised by a handful of fixed-case examples per algorithm (see
``test_pso.py::test_binding_respects_bounds`` and siblings). Hypothesis lets
us check the same invariant -- the best solution AutoColony returns always
lies inside the requested per-dimension bounds -- across many randomly
generated bound boxes and dimensionalities instead of a few hand-picked ones.
"""

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from colonyx import AutoColony

# Every continuous mode AutoColony exposes; ACO is excluded because it takes
# a distance matrix rather than bounds (see docs/algorithms/aco.md).
_CONTINUOUS_MODES = ("pso", "abc", "gwo", "fa", "sa", "cs", "ba", "gso", "bfo", "de", "cmaes")


def _sphere(x):
    return sum(xi * xi for xi in x)


@st.composite
def _bounds(draw, max_dim=4):
    n_dims = draw(st.integers(min_value=1, max_value=max_dim))
    boxes = []
    for _ in range(n_dims):
        low = draw(st.floats(min_value=-50.0, max_value=40.0, allow_nan=False, allow_infinity=False))
        high = draw(st.floats(min_value=low + 0.5, max_value=low + 50.0, allow_nan=False, allow_infinity=False))
        boxes.append((low, high))
    return boxes


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(mode=st.sampled_from(_CONTINUOUS_MODES), bounds=_bounds())
def test_predict_stays_within_bounds_for_random_boxes(mode, bounds):
    optimizer = AutoColony(mode=mode, n_iterations=15, random_state=0)
    optimizer.fit(_sphere, bounds=bounds)
    solution = optimizer.predict()

    assert len(solution) == len(bounds)
    for value, (low, high) in zip(solution, bounds):
        assert low - 1e-9 <= value <= high + 1e-9, (
            f"mode={mode!r} returned {value} outside bounds ({low}, {high})"
        )
