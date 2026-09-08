---
title: Rust Usage
description: How colonyx's Rust core works, why the optimization loops are written in Rust rather than Python, and how to use colonyx directly as a Rust crate.
---

# Rust usage

`colonyx` exposes its optimization core as a Rust library and its Python
extension from the same codebase: `src/algorithms/` and `src/core/` contain
the actual algorithm implementations and shared problem/solution types, and
`src/bindings.rs` wraps them with [PyO3](https://pyo3.rs/) so the exact same
code compiles into both a standalone Rust crate and the `colonyx._colonyx`
extension module that `colonyx/__init__.py` re-exports from. [maturin](https://www.maturin.rs/)
builds that extension module.

## Why the optimization core is written in Rust

Every algorithm here runs its inner loop — evaluate a population's fitness, update positions/velocities, sort or select, repeat — for potentially thousands of iterations across dozens of candidate solutions, which means tens of thousands to millions of small numeric operations per `fit()` call. That's exactly the workload where a native compiled loop with no per-iteration interpreter overhead, and no per-object allocation for things like particle positions, meaningfully outperforms the equivalent pure-Python loop; Rust also gets memory safety without a garbage collector, so continuous allocation/deallocation of populations across iterations doesn't add GC pause overhead into a hot loop the way it would in Python. This is why colonyx's algorithms live in Rust rather than Python from the ground up, instead of writing them in Python first and rewriting only proven bottlenecks later: for this workload, the hot path essentially *is* the whole algorithm. The tradeoff is that colonyx as a Rust crate has to expose Python-callable objectives through PyO3's callback mechanism (see the note on GIL and Python objectives below) whenever your objective function itself is written in Python rather than in Rust — that boundary crossing has real cost per call, which is worth knowing about if your objective function is cheap and called extremely often.

## Add the crate

In a Rust project, add `colonyx` as a dependency:

```toml
[dependencies]
colonyx = "0.3"
```

## Continuous optimization

Use `Bounds`, `ContinuousProblem`, and an optimizer such as `ParticleSwarm`:

```rust
use colonyx::algorithms::base::Optimizer;
use colonyx::algorithms::pso::ParticleSwarm;
use colonyx::core::{Bounds, ContinuousProblem};

fn main() {
    let bounds = Bounds::uniform(3, -5.0, 5.0).unwrap();
    let mut optimizer = ParticleSwarm::new(30, 100, 0.7, 1.5, 1.5, bounds);
    optimizer.set_random_seed(Some(42));

    let problem = ContinuousProblem {
        name: "sphere".to_string(),
        dimensions: 3,
        objective_function: Box::new(|x: &[f64]| x.iter().map(|value| value * value).sum()),
    };

    optimizer.fit(&problem).unwrap();

    let best = optimizer.predict().unwrap();
    println!("best position: {:?}", best.variables);
    println!("best score: {:?}", optimizer.score().unwrap());
}
```

## Discrete optimization

Use `AntColony` with a distance matrix for TSP-style problems:

```rust
use colonyx::algorithms::aco::{AcoVariant, AntColony};
use colonyx::algorithms::base::Optimizer;
use colonyx::core::DiscreteProblem;

fn main() {
    let mut optimizer = AntColony::new(
        20,
        100,
        1.0,
        3.0,
        0.5,
        100.0,
        true,
        AcoVariant::Basic,
        0.9,
        2.0,
        0.1,
        10.0,
    );

    let problem = DiscreteProblem {
        name: "tsp".to_string(),
        distance_matrix: vec![
            vec![0.0, 1.0, 9.0, 9.0],
            vec![1.0, 0.0, 1.0, 9.0],
            vec![9.0, 1.0, 0.0, 1.0],
            vec![9.0, 9.0, 1.0, 0.0],
        ],
    };

    optimizer.fit(&problem).unwrap();
    println!("best tour: {:?}", optimizer.predict().unwrap().variables);
    println!("tour length: {:?}", optimizer.score().unwrap());
}
```

## Multi-objective optimization

`Nsga2Optimizer` and `MopsoOptimizer` implement `MultiObjectiveOptimizer`
(`fit(&dyn MultiObjectiveProblem)` / `pareto_front() -> Vec<ParetoPoint>`),
the vector-objective counterpart to `Optimizer`, so both can be held
polymorphically:

```rust
use colonyx::algorithms::{MopsoOptimizer, MultiObjectiveOptimizer, Nsga2Optimizer};
use colonyx::core::{Bounds, MultiObjectiveContinuousProblem};

let bounds = Bounds::uniform(1, -2.0, 3.0).unwrap();
let problem = MultiObjectiveContinuousProblem {
    name: "biobjective".to_string(),
    dimensions: 1,
    objective_function: Box::new(|x: &[f64]| vec![x[0], (x[0] - 1.0).powi(2)]),
};

let mut optimizers: Vec<Box<dyn MultiObjectiveOptimizer>> = vec![
    Box::new(Nsga2Optimizer::new(30, 30, 0.8, 0.2, 0.5, 30, bounds.clone())),
    Box::new(MopsoOptimizer::new(20, 30, 0.7, 1.5, 1.5, 0.1, 20, bounds)),
];
for optimizer in optimizers.iter_mut() {
    optimizer.fit(&problem).unwrap();
    println!("{} points on the front", optimizer.pareto_front().len());
}
```

`BinaryParticleSwarm` implements the plain `Optimizer` trait the same way
(its `fit_with_objective` method is unchanged and still callable directly).

## Parallel fitness evaluation

`algorithms::base::evaluate_population(problem, population)` evaluates a
batch of candidates via [Rayon](https://docs.rs/rayon) when there's more
than one. It's used for population initialization across the continuous
algorithms and for CMA-ES's per-generation evaluation, since CMA-ES samples
a whole generation before selecting on it — no candidate's fitness depends
on another's within the same batch, so parallelizing it changes nothing
about the result.

!!! warning "Only parallelize independent batches"
    `evaluate_population` is safe only where each candidate's fitness is
    independent of every other candidate's in the same batch. Algorithms
    with steady-state, per-individual acceptance (DE, ABC's employed/onlooker
    phases) consume one candidate's result before generating the next, so
    parallelizing those loops would change the algorithm's actual behavior,
    not just its speed — don't do it there.

!!! note "GIL and Python objectives"
    When the objective is a Python callable (via `bindings::make_objective`),
    each evaluation re-acquires the GIL. The pyo3 binding releases the GIL
    for the whole `fit()` call (`py.allow_threads`) so Rayon's worker threads
    can each acquire it in turn — without that release, a worker blocking on
    the GIL while the calling thread holds it deadlocks. Real speedup from
    this still depends on the Python objective itself releasing the GIL
    while it runs (e.g. NumPy/SciPy-backed code); a pure-Python objective
    just serializes on the GIL again across threads. A native Rust `Problem`
    has no such limit.

## Notes

- The Python extension still lives at `colonyx._colonyx`.
- `colonyx::core` contains shared problem and solution types.
- `colonyx::algorithms` contains the Rust implementations.

## Building the extension yourself

If you're contributing to colonyx or just want to build from source rather than installing a wheel from PyPI, `maturin develop` builds the extension module and installs it into your active Python environment in one step — see [Release](release.md#local-build) for the exact commands, including the `RUSTFLAGS` needed on macOS for `cargo build` without maturin.

## Where to next

- [API Reference](api.md#rust-side-traits) for the `Optimizer`/`MultiObjectiveOptimizer`/`Problem` trait signatures.
- [Release](release.md) for building, packaging, and publishing both the PyPI wheel and the crates.io crate.
- [Algorithms overview](algorithms.md) for what each algorithm actually does, independent of which language you call it from.
