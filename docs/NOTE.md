
## Algorithms

| Order | Algorithm | Type       | Use Case                     | Notes                 |
| ----- | --------- | ---------- | ---------------------------- | --------------------- |
| 1     | ACO       | Discrete   | TSP, routing, scheduling     | Good starter          |
| 2     | PSO       | Continuous | Function optimization        | Widely used           |
| 3     | ABC       | Continuous | Optimization, feature select | Slightly more complex |
| 4     | Firefly   | Continuous | Multimodal optimization      | Easy to add after PSO |
| 5     | Glowworm  | Continuous | Multi-solution discovery     | Experimental          |
| 6     | Bacterial | Mixed      | Bio-inspired optimization    | Complex               |


## Directory

```

colonyx/
├── src/                        # Rust core implementation
│   ├── lib.rs                  # Main library entry point
│   ├── algorithms/             # Algorithm implementations
│   │   ├── mod.rs             # Algorithms module
│   │   ├── base.rs            # Base traits and types
│   │   ├── aco.rs             # Ant Colony Optimization
│   │   ├── pso.rs             # Particle Swarm Optimization
│   │   └── abc.rs             # Artificial Bee Colony
│   ├── core/                  # Core optimization structures
│   │   ├── mod.rs
│   │   ├── problem.rs         # Problem definitions
│   │   └── solution.rs        # Solution representations
│   ├── utils/                 # Utilities
│   │   ├── mod.rs
│   │   └── math.rs            # Math utilities
│   └── bindings.rs            # Python bindings
│
├── colonyx/                   # Python package
│   ├── __init__.py           # Main exports
│   ├── auto.py               # AutoColony class
│   ├── base.py               # Base classes
│   ├── utils.py              # Python utilities
│   └── datasets.py           # Benchmark datasets
│
├── examples/                  # Usage examples
├── tests/                     # Tests
└── docs/                      # Documentation

```