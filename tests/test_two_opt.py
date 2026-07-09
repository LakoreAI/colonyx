from __future__ import annotations

from colonyx._colonyx import two_opt


def test_two_opt_improves_crossing_tour():
    distance_matrix = [
        [0.0, 1.0, 2.0, 1.0],
        [1.0, 0.0, 1.0, 2.0],
        [2.0, 1.0, 0.0, 1.0],
        [1.0, 2.0, 1.0, 0.0],
    ]

    improved_tour, improved_length = two_opt([0, 2, 1, 3], distance_matrix)

    assert improved_tour != [0, 2, 1, 3]
    assert improved_length <= 4.0

