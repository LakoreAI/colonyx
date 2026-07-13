# Rust usage

`colonyx` exposes its optimization core as a Rust library and its Python
extension from the same codebase.

## Add the crate

In a Rust project, add `colonyx` as a dependency:

```toml
[dependencies]
colonyx = "0.1.1"
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

## Notes

- The Python extension still lives at `colonyx._colonyx`.
- `colonyx::core` contains shared problem and solution types.
- `colonyx::algorithms` contains the Rust implementations.
