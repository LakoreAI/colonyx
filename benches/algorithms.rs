//! Micro-benchmarks for the hot optimization loops in `colonyx`.
//!
//! Run with `cargo bench`. These track the Rust-side performance story the
//! project is built on (see `docs/rust.md`) so a regression in the inner
//! fitness-evaluation/update loop shows up here before it reaches users.

use colonyx::algorithms::aco::{AcoVariant, AntColony};
use colonyx::algorithms::{Optimizer, ParticleSwarm};
use colonyx::core::{Bounds, ContinuousProblem, DiscreteProblem};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

fn sphere(dimensions: usize) -> ContinuousProblem {
    ContinuousProblem {
        name: "sphere".to_string(),
        dimensions,
        objective_function: Box::new(|x: &[f64]| x.iter().map(|&xi| xi * xi).sum()),
    }
}

/// A fully-connected random-ish distance matrix, deterministic so the
/// benchmark workload doesn't vary between runs.
fn distance_matrix(n: usize) -> DiscreteProblem {
    let mut matrix = vec![vec![0.0; n]; n];
    for (i, row) in matrix.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            if i != j {
                *cell = 1.0 + ((i * 31 + j * 17) % 97) as f64;
            }
        }
    }
    DiscreteProblem { name: "random".to_string(), distance_matrix: matrix }
}

fn bench_pso(c: &mut Criterion) {
    let mut group = c.benchmark_group("pso_sphere");
    for dim in [2usize, 10, 30] {
        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |b, &dim| {
            b.iter(|| {
                let bounds = Bounds::uniform(dim, -5.0, 5.0).unwrap();
                let mut pso = ParticleSwarm::new(30, 100, 0.7, 1.5, 1.5, bounds);
                pso.set_random_seed(Some(42));
                pso.fit(&sphere(dim)).unwrap();
                pso.score().unwrap()
            });
        });
    }
    group.finish();
}

fn bench_aco(c: &mut Criterion) {
    let mut group = c.benchmark_group("aco_tsp");
    for n in [10usize, 30, 60] {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter(|| {
                let mut aco = AntColony::new(
                    20,
                    50,
                    1.0,
                    2.0,
                    0.5,
                    1.0,
                    true,
                    AcoVariant::Basic,
                    0.9,
                    2.0,
                    1e-4,
                    10.0,
                );
                aco.set_random_seed(Some(42));
                aco.fit(&distance_matrix(n)).unwrap();
                aco.score().unwrap()
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_pso, bench_aco);
criterion_main!(benches);
