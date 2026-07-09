
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
│   ├── lib.rs                  # Library entry + #[pymodule] registration
│   ├── bindings.rs             # Python bindings (pyo3 wrappers)
│   ├── algorithms/
│   │   ├── mod.rs              # Algorithms module
│   │   ├── base.rs             # Optimizer trait + error types
│   │   ├── aco.rs              # Ant Colony Optimization (discrete/TSP)
│   │   ├── pso.rs              # Particle Swarm Optimization (continuous)
│   │   └── abc.rs              # Artificial Bee Colony (continuous)
│   └── core/                   # Core optimization structures
│       ├── mod.rs
│       ├── problem.rs          # Problem definitions
│       ├── solution.rs         # Solution / SolutionSet
│       └── bounds.rs           # Variable bounds
│
├── colonyx/                    # Python package
│   ├── __init__.py             # Exports + compiled-core re-export
│   └── auto.py                 # AutoColony facade
│
├── tests/                      # pytest suite
│   ├── test_aco.py
│   ├── test_pso.py
│   └── test_abc.py
│
└── docs/                       # Documentation

```

> **Implemented and tested** (Rust core → pyo3 bindings → `AutoColony`):
> ACO (`mode='aco'`, discrete/TSP), PSO (`mode='pso'`, continuous) and
> ABC (`mode='abc'`, continuous).
>
> **Planned (not yet implemented):** `colonyx/datasets.py` (benchmark problems)
> and an `examples/` gallery. Firefly, Glowworm and Bacterial follow the same
> pattern as the continuous algorithms.