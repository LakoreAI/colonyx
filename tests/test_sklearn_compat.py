"""Tests for sklearn compatibility and serialization."""

import pickle

import numpy as np
import pytest
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from colonyx import AutoColony


def sphere(x):
    return sum(value * value for value in x)


def test_pickle_roundtrip_preserves_optimization_state():
    optimizer = AutoColony(mode="pso", n_iterations=120, random_state=42)
    optimizer.fit(sphere, bounds=[(-5, 5), (-5, 5), (-5, 5)])

    restored = pickle.loads(pickle.dumps(optimizer))

    assert restored.predict() == optimizer.predict()
    assert restored.score() == optimizer.score()


def test_pipeline_with_sklearn_compatibility_mode():
    rng = np.random.default_rng(7)
    features = rng.normal(size=(40, 4))
    targets = rng.normal(size=40)

    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("optimizer", AutoColony(mode="pso", n_iterations=10, random_state=3)),
        ]
    )

    pipeline.fit(features, targets)
    predictions = pipeline.predict(features)

    assert predictions.shape == (40,)
    assert np.isfinite(predictions).all()


def test_grid_search_runs_with_auto_mode():
    rng = np.random.default_rng(11)
    features = rng.normal(size=(25, 3))
    targets = rng.normal(size=25)

    search = GridSearchCV(
        AutoColony(),
        param_grid={
            "mode": ["auto", "pso"],
            "n_iterations": [5, 10],
            "random_state": [1],
        },
        cv=2,
    )

    search.fit(features, targets)

    assert search.best_estimator_ is not None
    assert isinstance(search.best_score_, float)


def test_randomized_search_runs_with_pso_mode():
    rng = np.random.default_rng(13)
    features = rng.normal(size=(24, 3))
    targets = rng.normal(size=24)

    search = RandomizedSearchCV(
        AutoColony(mode="pso"),
        param_distributions={
            "n_iterations": [5, 10, 15],
            "random_state": [1, 2, 3],
            "w": [0.5, 0.7, 0.9],
        },
        n_iter=2,
        cv=2,
        random_state=5,
    )

    search.fit(features, targets)

    assert search.best_estimator_ is not None
    assert isinstance(search.best_score_, float)


def test_column_transformer_pipeline_runs():
    rng = np.random.default_rng(17)
    features = rng.normal(size=(30, 4))
    targets = rng.normal(size=30)

    pipeline = Pipeline(
        [
            (
                "compose",
                ColumnTransformer(
                    [
                        ("first_half", StandardScaler(), [0, 1]),
                        ("second_half", StandardScaler(), [2, 3]),
                    ]
                ),
            ),
            ("optimizer", AutoColony(mode="auto", n_iterations=10, random_state=2)),
        ]
    )

    pipeline.fit(features, targets)
    prediction = pipeline.predict(features)

    assert prediction.shape == (30,)
    assert np.isfinite(prediction).all()


def test_optimization_metrics_are_exposed():
    optimizer = AutoColony(mode="pso", n_iterations=40, random_state=4)
    optimizer.fit(sphere, bounds=[(-5, 5), (-5, 5), (-5, 5)])

    metrics = optimizer.optimization_metrics()

    assert set(metrics) == {"best_score", "convergence_rate", "diversity", "robustness"}
    assert np.isfinite(list(metrics.values())).all()


def test_default_search_spaces_are_mode_specific():
    grids = AutoColony.default_param_grids()
    distributions = AutoColony.default_param_distributions()

    assert isinstance(grids, list)
    assert {entry["mode"][0] for entry in grids} == {"aco", "pso", "abc", "gwo", "fa", "sa", "cs", "ba", "gso", "de"}
    assert set(distributions) == {"aco", "pso", "abc", "gwo", "fa", "sa", "cs", "ba", "gso", "de"}


def test_recommend_algorithm_uses_problem_shape():
    optimizer = AutoColony()

    assert optimizer.recommend_algorithm(np.eye(4))["mode"] == "aco"
    assert optimizer.recommend_algorithm(sphere, bounds=[(-5, 5), (-5, 5)])["mode"] == "pso"
    assert optimizer.recommend_algorithm(sphere, bounds=[(-5, 5)] * 6)["mode"] == "abc"


def test_suggest_parameters_matches_recommended_mode():
    optimizer = AutoColony()
    suggestions = optimizer.suggest_parameters(sphere, bounds=[(-5, 5)] * 6)

    assert suggestions["mode"] == "abc"
    assert "n_bees" in suggestions
    assert "limit" in suggestions


def test_parameter_mapping_and_help_are_mode_specific():
    optimizer = AutoColony(mode="pso")

    assert optimizer.parameter_mapping()["n_particles"] == "n_particles"
    assert "PSO params" in optimizer.parameter_help()


def test_auto_mode_chooses_abc_for_higher_dimensional_objective():
    optimizer = AutoColony(mode="auto", n_iterations=40, random_state=6)
    optimizer.fit(sphere, bounds=[(-5, 5)] * 6)

    assert optimizer._algorithm_mode == "abc"


def test_parameter_conflicts_are_recorded_for_explicit_modes():
    optimizer = AutoColony(mode="aco", n_particles=99)
    optimizer.resolve_parameter_conflicts("aco")

    assert "n_particles" in optimizer.parameter_conflicts_


def test_optimization_cv_strategy_returns_kfold_for_regression():
    rng = np.random.default_rng(21)
    features = rng.normal(size=(18, 3))
    targets = rng.normal(size=18)

    splitter = AutoColony.optimization_cv_strategy(features, targets, n_splits=4)

    assert splitter.__class__.__name__ == "KFold"
    search = GridSearchCV(
        AutoColony(),
        param_grid={"mode": ["auto"], "n_iterations": [5]},
        cv=splitter,
    )
    search.fit(features, targets)


def test_optimization_cv_strategy_returns_stratified_for_classification_like_targets():
    rng = np.random.default_rng(22)
    features = rng.normal(size=(20, 3))
    targets = np.array([0, 1] * 10)

    splitter = AutoColony.optimization_cv_strategy(features, targets, n_splits=5)

    assert splitter.__class__.__name__ == "StratifiedKFold"
